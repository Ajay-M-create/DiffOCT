"""Measured-space noise correction and pixel-level fitting exclusions."""
import numpy as np


def correct_and_mask(measured_db, noise, sensitivity, confocal, original_fit_mask):
    """Correct one (z,x) frame; reject individual invalid measurements.

    All depth curves use shape (z,). Invalid targets receive a finite placeholder
    and MUST be indexed with loss_mask. No intensity flooring or extra smoothing.
    The original fitting band remains available as transport support: missing
    observations do not establish that the optical path contains no scatterers.
    """
    measured_db = np.asarray(measured_db, dtype=np.float64)
    fit = np.asarray(original_fit_mask, dtype=bool)
    if measured_db.ndim != 2 or fit.shape != measured_db.shape:
        raise ValueError("Expected measured_db and mask with matching (z,x) shape")
    curves = [np.asarray(a, dtype=np.float64) for a in (noise, sensitivity, confocal)]
    if any(a.shape != (measured_db.shape[0],) for a in curves):
        raise ValueError("Noise, sensitivity and confocal must each have shape (z,)")
    noise, sensitivity, confocal = curves
    if not all(np.isfinite(a).all() for a in curves):
        raise ValueError("Calibration contains nonfinite values")
    if np.any(noise < 0) or np.any(sensitivity <= 0) or np.any(confocal <= 0):
        raise ValueError("Noise must be nonnegative; S and H must be positive")
    with np.errstate(over="ignore", invalid="ignore"):
        intensity = np.power(10.0, measured_db / 10.0)
        signal = intensity - noise[:, None]
        corrected = signal / (sensitivity * confocal)[:, None]
    finite = np.isfinite(measured_db) & np.isfinite(corrected)
    loss_mask = fit & finite & (corrected > 0)
    target = np.zeros_like(measured_db)
    valid_measurements = finite & (corrected > 0)
    target[valid_measurements] = 10.0 * np.log10(corrected[valid_measurements])
    # The supplied fit-band start defines the surface for both pilot variants.
    # Full-depth fitting continues beyond the CF band; IntMask is not used.
    z = np.arange(fit.shape[0])[:, None]
    nonempty = fit.any(axis=0)
    first = fit.argmax(axis=0)
    below_surface = (z >= first[None, :]) & nonempty[None, :]
    full_loss = below_surface & valid_measurements
    last_valid = fit.shape[0] - 1 - full_loss[::-1].argmax(axis=0)
    full_transport = below_surface & (z <= last_valid[None, :]) & full_loss.any(axis=0)[None, :]
    return {
        "target_db": target,
        "original_fit_mask": fit.copy(),
        "loss_mask": loss_mask,
        "comparison_mask": loss_mask.copy(),
        "valid_measurement_mask": valid_measurements,
        "full_depth_loss_mask": full_loss,
        "full_depth_transport_mask": full_transport,
        "transport_mask": fit.copy(),
        "excluded_mask": fit & ~loss_mask,
        "nonpositive_mask": fit & finite & (corrected <= 0),
        "nonfinite_mask": fit & ~finite,
    }
