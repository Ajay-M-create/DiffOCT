"""Numerical checks for calibration, missing observations and independent batches."""
import json
import math
import numpy as np
import torch

from preprocessing import correct_and_mask
from model import forward_db, objective
from gain import fixed_cf_gain


def validate():
    torch.set_num_threads(2)
    dz = 0.00735294117647059
    depth, width = 40, 5
    mu_true, gain_true = 0.7, 100.0
    z = np.arange(depth)
    expected = gain_true + 10 * np.log10(mu_true) - (20 / np.log(10)) * mu_true * z * dz
    noise = np.linspace(1.2e8, 1.6e8, depth)
    sensitivity = np.linspace(0.6, 0.9, depth)
    confocal = 1 / (1 + ((z - 22) / 12) ** 2)
    measured = np.broadcast_to((10 ** (expected / 10) * sensitivity * confocal + noise)[:, None], (depth, width))
    mask = np.ones((depth, width), dtype=bool)
    correction = correct_and_mask(10 * np.log10(measured), noise, sensitivity, confocal, mask)
    calibration_error = float(np.max(np.abs(correction["target_db"] - expected[:, None])))
    assert calibration_error < 1e-10, calibration_error
    transport = torch.ones((1, depth, width), dtype=torch.bool)
    mu = torch.full((1, depth, width), mu_true, dtype=torch.float64)
    analytic_error = float((forward_db(mu, transport, dz) + gain_true - torch.tensor(expected)[None, :, None]).abs().max())
    assert analytic_error < 1e-10, analytic_error
    # An interior invalid pixel remains part of the path but has no data residual.
    damaged = measured.copy()
    damaged[10, 2] = 0.5 * noise[10]
    partial = correct_and_mask(10 * np.log10(damaged), noise, sensitivity, confocal, mask)
    assert partial["loss_mask"].sum() == mask.sum() - 1
    assert partial["loss_mask"].any(axis=0).all()
    assert partial["transport_mask"].all()
    assert np.isfinite(partial["target_db"]).all()
    valid = torch.tensor(partial["loss_mask"])[None]
    target = torch.tensor(partial["target_db"])[None]
    gain = torch.full((1, width), gain_true, dtype=torch.float64)
    base = objective(mu, target, valid, transport, dz, gain)
    altered = target.clone()
    altered[~valid] = 1e6
    changed = objective(mu, altered, valid, transport, dz, gain)
    assert torch.allclose(base[0], changed[0], atol=1e-12, rtol=0)
    transported = mu.clone()
    transported[0, 10, 2] = 1.4
    delta = forward_db(transported, transport, dz) - forward_db(mu, transport, dz)
    assert abs(float(delta[0, 11, 2]) + 20 / math.log(10) * 0.7 * dz) < 1e-12
    # Independent and batched objectives must give the same gradients.
    generator = torch.Generator().manual_seed(42)
    batch = (0.2 + torch.rand((2, depth, width), generator=generator, dtype=torch.float64)).requires_grad_()
    targets = target.repeat(2, 1, 1)
    valids = valid.repeat(2, 1, 1)
    transports = transport.repeat(2, 1, 1)
    gains = torch.tensor([[98.,99.,100.,101.,102.], [107.,106.,105.,104.,103.]], dtype=torch.float64)
    loss = objective(batch, targets, valids, transports, dz, gains)[0].sum()
    grad_batch = torch.autograd.grad(loss, batch)[0]
    independent = []
    for i in range(2):
        single = batch[i:i+1].detach().clone().requires_grad_()
        item = objective(single, targets[i:i+1], valids[i:i+1], transports[i:i+1], dz, gains[i:i+1])[0].sum()
        independent.append(torch.autograd.grad(item, single)[0])
    gradient_error = float((grad_batch - torch.cat(independent)).abs().max())
    assert gradient_error < 1e-10 and torch.isfinite(grad_batch).all()
    # Recover a homogeneous synthetic coefficient with independently known gain.
    scalar = torch.tensor(1.8, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([scalar], lr=0.02)
    clean_target = torch.tensor(expected)[None, :, None].expand(1, depth, width)
    for _ in range(1200):
        optimizer.zero_grad()
        prediction = forward_db(scalar.expand_as(mu), transport, dz) + gain_true
        ((prediction - clean_target) ** 2).mean().backward()
        optimizer.step()
        with torch.no_grad():
            scalar.clamp_(1e-6, 20)
    recovery_error = abs(float(scalar.detach()) - mu_true)
    assert recovery_error < 1e-6, recovery_error
    float32_error = float((forward_db(mu.float(), transport, dz).double() - forward_db(mu, transport, dz)).abs().max())
    assert float32_error < 1e-3
    empty = correct_and_mask(10 * np.log10(measured), noise, sensitivity, confocal, np.zeros_like(mask))
    assert not empty["loss_mask"].any() and np.isfinite(empty["target_db"]).all()
    assert not empty["full_depth_loss_mask"].any()
    short_band = np.zeros_like(mask)
    short_band[3:15] = True
    domains = correct_and_mask(10 * np.log10(damaged), noise, sensitivity, confocal, short_band)
    assert domains["comparison_mask"].sum() == 12 * width - 1
    assert np.array_equal(domains["comparison_mask"], domains["loss_mask"])
    assert domains["full_depth_loss_mask"].sum() == (depth - 3) * width - 1
    assert not domains["full_depth_loss_mask"][:3].any()
    assert domains["full_depth_transport_mask"][10, 2]
    assert np.all(domains["comparison_mask"] <= domains["full_depth_loss_mask"])
    # Different absolute origins, gains and coefficients must recover the CF lines.
    band = np.zeros((depth, width), dtype=bool)
    coefficients = np.array([.5, .7, 1., 1.5, 2.])
    intercepts = np.array([24., 25., 26., 27., 28.])
    for x in range(width):
        band[2 + 2*x:35, x] = True
    fixed, fallback = fixed_cf_gain(intercepts, coefficients, band, dz)
    assert not fallback.any()
    constant_mu = torch.tensor(np.broadcast_to(coefficients, band.shape).copy())[None]
    converted = forward_db(constant_mu, torch.tensor(band)[None], dz).numpy()[0] + fixed[None, :]
    expected_cf = (10 / np.log(10)) * (intercepts[None, :] - 2 * coefficients[None, :] * dz * np.arange(depth)[:, None])
    gain_conversion_error = float(np.max(np.abs(converted[band] - expected_cf[band])))
    assert gain_conversion_error < 1e-10
    broken_intercepts = intercepts.copy(); broken_intercepts[1] = np.nan
    broken_mu = coefficients.copy(); broken_mu[3] = 0
    fallback_gain, fallback = fixed_cf_gain(broken_intercepts, broken_mu, band, dz)
    assert np.array_equal(fallback, [False, True, False, True, False])
    np.testing.assert_equal(fallback_gain[fallback], np.median(fallback_gain[~fallback]))
    # Shrinking mu cannot be compensated by a gain update in the new objective.
    clean_signal = torch.pow(10., (clean_target - clean_target.mean()) / 10.)
    clean_signal /= clean_signal.mean()
    scale_losses = [float(objective(c * clean_signal, clean_target, transport, transport, dz, gain)[0])
                    for c in (.7, .07, .007, .0007)]
    assert all(b > a for a, b in zip(scale_losses, scale_losses[1:])), scale_losses
    assert torch.equal(objective(mu, target, valid, transport, dz, gain)[2], gain)
    return dict(calibration_max_error_db=calibration_error, analytic_forward_max_error_db=analytic_error,
                batched_gradient_max_error=gradient_error, homogeneous_recovery_error_mm_inv=recovery_error,
                float32_max_error_db=float32_error, invalid_pixel_loss_invariance=True,
                missing_observation_preserves_transport=True, partial_alines_retained=True,
                empty_mask_valid=True, matched_comparison_mask_across_domains=True,
                cf_gain_conversion_max_error_db=gain_conversion_error, gain_fallback_verified=True,
                shrinking_mu_fixed_gain_objectives=scale_losses, fixed_gain_unchanged=True)


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
