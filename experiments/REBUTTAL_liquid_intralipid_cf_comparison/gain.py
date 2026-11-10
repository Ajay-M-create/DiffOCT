"""Convert the CF natural-log intercept to a fixed DiffOCT gain per A-line."""
import numpy as np

FORMULA = "g_db = (10/ln(10))*G_prime - 10*log10(mu_CF) - (20/ln(10))*mu_CF*dz_mm*z_start_absolute_zero_based"


def fixed_cf_gain(g_prime, mu_cf, original_fit_mask, dz_mm):
    g_prime = np.asarray(g_prime, dtype=np.float64)
    mu_cf = np.asarray(mu_cf, dtype=np.float64)
    mask = np.asarray(original_fit_mask, dtype=bool)
    if mask.ndim != 2 or g_prime.shape != (mask.shape[1],) or mu_cf.shape != g_prime.shape:
        raise ValueError("Expected CF arrays (x,) and original fit mask (z,x)")
    valid = np.isfinite(g_prime) & np.isfinite(mu_cf) & (mu_cf > 0) & mask.any(axis=0)
    if not valid.any():
        raise ValueError("No valid CF gain in this frame; cannot compute median fallback")
    starts = mask.argmax(axis=0)
    gain = np.empty_like(g_prime)
    gain[valid] = (10 / np.log(10)) * g_prime[valid] - 10 * np.log10(mu_cf[valid]) - (20 / np.log(10)) * mu_cf[valid] * dz_mm * starts[valid]
    gain[~valid] = np.median(gain[valid])
    if not np.isfinite(gain).all():
        raise ValueError("Converted gain is not finite")
    return gain, ~valid
