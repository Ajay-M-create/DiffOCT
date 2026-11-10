#!/usr/bin/env python
"""Confocal parameter estimation study for the differentiable OCT inverse solver.

Investigates whether the Rayleigh range (z_R) and focal plane position (z_FP)
of the confocal function can be recovered jointly with attenuation coefficients
from OCT image data alone — without separate calibration measurements.

Four experimental phases:

  Phase 1 — Synthetic validation: Generate B-scans with known z_FP, z_R, mu and
            recover all three using different strategies.
  Phase 2 — Strategy comparison on real phantom data: Compare joint, alternating,
            analytical, and two-stage approaches.
  Phase 3 — Initialization sweep: Test sensitivity to z_FP and z_R starting values.
  Phase 4 — Noise robustness: Add Gaussian noise at different SNR levels to
            synthetic data and test recovery accuracy.

At the end, a detailed Markdown report is generated with embedded plots.

Strategies implemented:
  1. Joint     — optimize mu, z_FP, z_R simultaneously with Adam
  2. Alternating — blocks of mu-only then confocal-only optimization
  3. Analytical — fit Lorentzian to depth-averaged profile, then optimize mu only
  4. Two-stage — analytical confocal init → joint fine-tuning

References:
  - Faber et al. (2004), "Quantitative measurement of attenuation coefficients
    of weakly scattering media using optical coherence tomography", Opt Express
  - Vermeer et al. (2014), "Depth-resolved model-based reconstruction of
    attenuation coefficients in optical coherence tomography", BOE
  - Uribe-Patarroyo & Bouma (2020), "Rotational distortion correction in
    endoscopic OCT", Opt Lett — discusses system calibration challenges
  - Almasian et al. (2015), "Validation of quantitative attenuation and
    backscattering coefficient measurements by OCT", JBO

Usage:
    python experiments/optimize_confocal.py
    PHASE=1 python experiments/optimize_confocal.py
    PHASE=2 PHANTOM_INDEX=0 python experiments/optimize_confocal.py
    SUMMARIZE=1 python experiments/optimize_confocal.py
"""

import csv
import gc
import math
import os
import sys
import time
import importlib.util
from typing import Optional

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from scipy.optimize import curve_fit, minimize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

# Ported from main:test/12_optimize_confocal.py. On this branch the phantom
# loading helpers live in the shared src/helpers.py module rather than being
# importlib-loaded out of a sibling numbered script.
from src.helpers import load_phantom_bscan, discover_despeckled_phantoms

from src.forward import RasterScanOCTForward
from src.inverse import build_noise_floor_mask, subtract_noise_floor_linear
from src.roi import load_intmask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CF_REFERENCE = {1.0: 4.523, 1.5: 5.914, 2.0: 5.878, 3.0: 7.742, 5.0: 9.346}

# Ground truth confocal params from config.yaml / MATLAB calibration
GT_ZFP = 231.0   # pixels
GT_ZR = 131.5     # pixels

OUTPUT_BASE = os.path.join(PHANTOM_ROOT, "outputs", "12_confocal_optimization")


# ===========================================================================
# Confocal-optimizable forward model
# ===========================================================================

class ConfocalOptimizableForward(nn.Module):
    """SS Beer-Lambert forward model with LEARNABLE confocal parameters.

    Identical to RasterScanOCTForward except z_FP and z_R are nn.Parameters
    and confocal_dB is recomputed dynamically in the forward pass.
    """

    def __init__(self, config, device, z_fp_init=231.0, z_r_init=131.5):
        super().__init__()
        self.device_str = device
        self.depth_samples = int(config.forward_model_system.depth_samples)
        self.a_lines = int(config.forward_model_system.a_lines)
        self.axial_pixel_size_mm = float(config.forward_model_system.axial_pixel_size_mm)

        # Learnable confocal parameters
        self.z_fp = nn.Parameter(
            torch.tensor(z_fp_init, dtype=torch.float32, device=device))
        self.z_r = nn.Parameter(
            torch.tensor(z_r_init, dtype=torch.float32, device=device))

        # Fixed buffers
        self.register_buffer('z_px',
            torch.arange(self.depth_samples, dtype=torch.float32, device=device))
        self.register_buffer('gain_dB',
            torch.tensor(0.0, dtype=torch.float32, device=device))
        self.register_buffer('_K',
            torch.tensor(20.0 / math.log(10.0), dtype=torch.float32, device=device))
        self.register_buffer('_zeros_row',
            torch.zeros(1, self.a_lines, dtype=torch.float32, device=device))

    def _compute_confocal_dB(self):
        """Dynamically compute confocal correction from learnable z_fp, z_r."""
        # Clamp z_r to positive to avoid division by zero
        z_r_safe = torch.clamp(self.z_r, min=5.0)
        confocal = 1.0 / (((self.z_px - self.z_fp) / z_r_safe) ** 2 + 1.0)
        return 10.0 * torch.log10(confocal + 1e-12)

    def forward(self, mu, alpha, noise_mask=None):
        eps = 1e-12
        mu_active = mu * noise_mask if noise_mask is not None else mu
        alpha_active = alpha * noise_mask if noise_mask is not None else alpha

        cumulative = torch.cumsum(mu_active * self.axial_pixel_size_mm, dim=0)
        shifted = torch.cat([self._zeros_row, cumulative[:-1]], dim=0)

        confocal_dB = self._compute_confocal_dB()

        I_dB = (self.gain_dB
                + 10.0 * torch.log10(alpha_active + eps)
                + 10.0 * torch.log10(mu_active + eps)
                + confocal_dB.unsqueeze(1)
                - self._K * shifted)
        return I_dB


# ===========================================================================
# Analytical confocal estimation (Lorentzian + exponential fit)
# ===========================================================================

def analytical_confocal_estimation(
    I_target_dB: np.ndarray,
    noise_floor_dB: float,
    axial_pixel_size_mm: float,
    intmask: Optional[np.ndarray] = None,
) -> dict:
    """Estimate z_FP and z_R by fitting a parametric model to the depth profile.

    Model: I_dB(z) = C0 - 10*log10(((z - zFP)/zR)^2 + 1) - slope * z
    where slope = K * mu * dz.

    The depth profile is obtained by lateral averaging over tissue pixels.

    Returns:
        dict with keys: z_fp, z_r, mu_est, c0, success, message
    """
    D, A = I_target_dB.shape
    K = 20.0 / math.log(10.0)

    # Build depth profile by lateral averaging
    if intmask is not None:
        tissue_cols = intmask.sum(axis=0) > 0
        if tissue_cols.sum() > 10:
            profile = np.nanmean(
                np.where(intmask[:, tissue_cols] > 0, I_target_dB[:, tissue_cols], np.nan),
                axis=1)
        else:
            profile = np.mean(I_target_dB, axis=1)
    else:
        profile = np.mean(I_target_dB, axis=1)

    # Use tissue region only (above noise floor)
    valid = (~np.isnan(profile)) & (profile > noise_floor_dB)
    z_valid = np.where(valid)[0]
    if len(z_valid) < 20:
        return dict(z_fp=GT_ZFP, z_r=GT_ZR, mu_est=5.0, c0=100.0,
                    success=False, message="Too few valid depth pixels")

    z_start, z_end = z_valid[0], z_valid[-1]
    z_fit = np.arange(z_start, z_end + 1)
    y_fit = profile[z_fit]

    # Remove NaN from fit data
    good = ~np.isnan(y_fit)
    z_fit = z_fit[good]
    y_fit = y_fit[good]

    def model(z, c0, zfp, zr, slope):
        return c0 - 10.0 * np.log10(((z - zfp) / max(zr, 5.0)) ** 2 + 1.0) - slope * z

    # Initial guess
    peak_idx = np.argmax(y_fit)
    z_peak = z_fit[peak_idx]
    p0 = [y_fit[peak_idx] + 5.0, float(z_peak), 130.0, 0.01]

    bounds_lo = [y_fit.min() - 10, 50.0, 10.0, 0.0]
    bounds_hi = [y_fit.max() + 50, 450.0, 500.0, 0.5]

    try:
        popt, pcov = curve_fit(model, z_fit, y_fit, p0=p0,
                               bounds=(bounds_lo, bounds_hi), maxfev=10000)
        c0, zfp, zr, slope = popt
        mu_est = slope / (K * axial_pixel_size_mm)
        return dict(z_fp=float(zfp), z_r=float(zr), mu_est=float(mu_est),
                    c0=float(c0), success=True, message="converged")
    except Exception as e:
        return dict(z_fp=GT_ZFP, z_r=GT_ZR, mu_est=5.0, c0=100.0,
                    success=False, message=str(e))


# ===========================================================================
# Custom optimization loop with confocal parameter support
# ===========================================================================

def optimize_with_confocal(
    model: ConfocalOptimizableForward,
    mu: nn.Parameter,
    alpha: nn.Parameter,
    I_target_dB: torch.Tensor,
    noise_floor_mask: Optional[torch.Tensor],
    strategy: str,
    num_iterations: int = 3000,
    lr_mu: float = 1e-3,
    lr_confocal: float = 0.5,
    loss_scale: float = 1e3,
    mu_min: float = 0.01,
    mu_max: float = 20.0,
    mu_tv_weight: float = 1e3,
    max_delta_mu_mm: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Optimization loop supporting different confocal estimation strategies.

    Strategies:
        'joint'       — optimize mu + z_fp + z_r simultaneously
        'alternating' — alternate blocks: 80% mu-only, 20% confocal-only
        'fixed'       — keep z_fp, z_r frozen, optimize mu only (baseline)

    Returns:
        dict with keys: z_fp_history, z_r_history, mse_history, mu_mean_history,
                        gain_history, z_fp_final, z_r_final, mu_final, best_mse, time_sec
    """
    eps = 1e-12
    device = mu.device

    # Zero noise-masked pixels
    if noise_floor_mask is not None:
        with torch.no_grad():
            mu.data *= noise_floor_mask
            alpha.data *= noise_floor_mask

    # Auto-init gain_dB
    with torch.no_grad():
        model.gain_dB.fill_(0.0)
        I_pred_init = model(mu, alpha, noise_mask=noise_floor_mask)
        if noise_floor_mask is not None:
            tissue = noise_floor_mask > 0.5
            offset = I_target_dB[tissue].median() - I_pred_init[tissue].median()
        else:
            offset = I_target_dB.median() - I_pred_init.median()
        model.gain_dB.fill_(offset.item())

    # Set up parameter groups based on strategy
    if strategy == 'fixed':
        model.z_fp.requires_grad_(False)
        model.z_r.requires_grad_(False)
        optimizer = torch.optim.Adam([mu, alpha], lr=lr_mu)
    elif strategy == 'joint':
        optimizer = torch.optim.Adam([
            {'params': [mu, alpha], 'lr': lr_mu},
            {'params': [model.z_fp, model.z_r], 'lr': lr_confocal},
        ])
    elif strategy == 'alternating':
        opt_mu = torch.optim.Adam([mu, alpha], lr=lr_mu)
        opt_conf = torch.optim.Adam([model.z_fp, model.z_r], lr=lr_confocal)
        optimizer = opt_mu  # placeholder for scheduler
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations, eta_min=lr_mu * 0.01)

    # Tracking
    history = dict(z_fp=[], z_r=[], mse=[], mu_mean=[], gain=[])
    best_mse = float('inf')
    best_mu = mu.detach().clone()
    best_z_fp = model.z_fp.item()
    best_z_r = model.z_r.item()

    t0 = time.time()

    for it in range(num_iterations):
        # Decide which params to update for alternating
        if strategy == 'alternating':
            cycle = it % 100
            confocal_phase = cycle >= 80  # last 20% of each 100-step block
            if confocal_phase:
                model.z_fp.requires_grad_(True)
                model.z_r.requires_grad_(True)
                mu.requires_grad_(False)
                alpha.requires_grad_(False)
                cur_opt = opt_conf
            else:
                model.z_fp.requires_grad_(False)
                model.z_r.requires_grad_(False)
                mu.requires_grad_(True)
                alpha.requires_grad_(True)
                cur_opt = opt_mu
        else:
            cur_opt = optimizer

        cur_opt.zero_grad(set_to_none=False)
        I_pred = model(mu, alpha, noise_mask=noise_floor_mask)

        # MSE loss
        diff = (I_pred - I_target_dB) ** 2
        if noise_floor_mask is not None:
            loss = loss_scale * (diff * noise_floor_mask).sum() / (noise_floor_mask.sum() + eps)
        else:
            loss = loss_scale * diff.mean()

        # TV regularization on mu
        if mu_tv_weight > 0 and mu.requires_grad:
            dz = (mu[1:, :] - mu[:-1, :]).abs()
            dx = (mu[:, 1:] - mu[:, :-1]).abs()
            if noise_floor_mask is not None:
                mask_dz = noise_floor_mask[1:, :] * noise_floor_mask[:-1, :]
                mask_dx = noise_floor_mask[:, 1:] * noise_floor_mask[:, :-1]
                tv = (dz * mask_dz).sum() / (mask_dz.sum() + eps) + \
                     (dx * mask_dx).sum() / (mask_dx.sum() + eps)
            else:
                tv = dz.mean() + dx.mean()
            loss = loss + mu_tv_weight * tv

        loss.backward()

        # Gradient clamping for mu
        if mu.grad is not None:
            with torch.no_grad():
                torch.nan_to_num_(mu.grad, nan=0.0, posinf=0.0, neginf=0.0)
                if lr_mu > 0 and max_delta_mu_mm < 1e6:
                    mu.grad.clamp_(-max_delta_mu_mm / lr_mu, max_delta_mu_mm / lr_mu)

        cur_opt.step()
        if strategy != 'alternating':
            scheduler.step()

        # Project mu back into bounds
        with torch.no_grad():
            mu.data.clamp_(mu_min, mu_max)
            alpha.data.clamp_(1.0, 1.0)  # alpha fixed at 1.0
            if noise_floor_mask is not None:
                mu.data.mul_(noise_floor_mask)
                alpha.data.mul_(noise_floor_mask)

            # Clamp confocal params
            model.z_fp.data.clamp_(10.0, 490.0)
            model.z_r.data.clamp_(5.0, 500.0)

            # Per-iter MSE-optimal gain
            model.gain_dB.fill_(0.0)
            pred_no_gain = model(mu, alpha, noise_mask=noise_floor_mask)
            if noise_floor_mask is not None:
                g_star = ((I_target_dB - pred_no_gain) * noise_floor_mask).sum() / \
                         (noise_floor_mask.sum() + eps)
            else:
                g_star = (I_target_dB - pred_no_gain).mean()
            model.gain_dB.fill_(g_star.item())

        # Record
        with torch.no_grad():
            mse_val = float(loss.item() / loss_scale)
            history['z_fp'].append(model.z_fp.item())
            history['z_r'].append(model.z_r.item())
            history['mse'].append(mse_val)
            history['mu_mean'].append(float(mu.mean().item()))
            history['gain'].append(model.gain_dB.item())

            if mse_val < best_mse:
                best_mse = mse_val
                best_mu.copy_(mu.data)
                best_z_fp = model.z_fp.item()
                best_z_r = model.z_r.item()

        if verbose and (it % max(1, num_iterations // 10) == 0 or it == num_iterations - 1):
            print(f"    iter {it:5d} | MSE={mse_val:.4e} | "
                  f"zFP={model.z_fp.item():.1f} zR={model.z_r.item():.1f} | "
                  f"mu_mean={mu.mean().item():.3f}")

    # Restore best
    with torch.no_grad():
        mu.data.copy_(best_mu)
        model.z_fp.data.fill_(best_z_fp)
        model.z_r.data.fill_(best_z_r)

    # Ensure requires_grad is restored
    mu.requires_grad_(True)
    alpha.requires_grad_(True)
    model.z_fp.requires_grad_(True)
    model.z_r.requires_grad_(True)

    elapsed = time.time() - t0
    return dict(
        z_fp_history=history['z_fp'], z_r_history=history['z_r'],
        mse_history=history['mse'], mu_mean_history=history['mu_mean'],
        gain_history=history['gain'],
        z_fp_final=best_z_fp, z_r_final=best_z_r,
        mu_final=best_mu.cpu().numpy(), best_mse=best_mse, time_sec=elapsed,
    )


# ===========================================================================
# Helpers
# ===========================================================================

def _load_configs():
    cfg_dir = os.path.join(PHANTOM_ROOT, "configs")
    config = OmegaConf.load(os.path.join(cfg_dir, "config.yaml"))
    opt_config = OmegaConf.load(os.path.join(cfg_dir, "optimization_config_fast.yaml"))
    sc_config = OmegaConf.load(os.path.join(cfg_dir, "scatterer_config.yaml"))
    return config, opt_config, sc_config


def _cleanup(*tensors):
    for t in tensors:
        del t
    plt.close("all")
    gc.collect()
    torch.cuda.empty_cache()


def _generate_synthetic_bscan(
    config, device, mu_value=5.0, z_fp=GT_ZFP, z_r=GT_ZR, noise_std_dB=0.0, seed=42,
):
    """Generate a synthetic B-scan using the SS forward model with known params.

    Returns: (I_target_dB, noise_floor_mask, gt_mu_value)
    """
    D = int(config.forward_model_system.depth_samples)
    A = int(config.forward_model_system.a_lines)
    noise_floor_dB = float(config.forward_model_system.noise_floor_dB)

    # Build standard SS forward model with given confocal params
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.forward_model_system.focal_plane_position_pixels = z_fp
    cfg.forward_model_system.rayleigh_range_pixels = z_r
    cfg.forward_model_system.forward_model_type = 'ss'

    fwd = RasterScanOCTForward(cfg, device=device)

    # Uniform mu with tissue in rows 200-350 (mimicking a phantom slab)
    mu_gt = torch.zeros(D, A, dtype=torch.float32, device=device)
    mu_gt[200:350, :] = mu_value

    alpha_gt = torch.ones(D, A, dtype=torch.float32, device=device)

    # Build noise mask (tissue region only)
    nf_mask = torch.zeros(D, A, dtype=torch.float32, device=device)
    nf_mask[200:350, :] = 1.0

    # Auto-init gain
    with torch.no_grad():
        fwd.gain_dB.fill_(0.0)
        pred_init = fwd(mu_gt, alpha_gt, noise_mask=nf_mask)
        tissue = nf_mask > 0.5
        offset = 100.0 - pred_init[tissue].median()  # place tissue around 100 dB
        fwd.gain_dB.fill_(offset.item())

        I_clean = fwd(mu_gt, alpha_gt, noise_mask=nf_mask)

    # Add noise
    if noise_std_dB > 0:
        torch.manual_seed(seed)
        noise = torch.randn_like(I_clean) * noise_std_dB
        I_target = I_clean + noise
    else:
        I_target = I_clean.clone()

    # Set non-tissue to noise floor level
    with torch.no_grad():
        I_target[nf_mask < 0.5] = noise_floor_dB - 5.0

    return I_target.detach(), nf_mask.detach(), mu_value


def _compute_tissue_mu_stats(mu_np, mask_np, thresh=0.01):
    """Compute mean and std of mu within a mask."""
    vals = mu_np[mask_np > 0.5]
    vals = vals[vals > thresh]
    if len(vals) == 0:
        return float('nan'), float('nan')
    return float(vals.mean()), float(vals.std())


# ===========================================================================
# Phase 1: Synthetic validation
# ===========================================================================

def run_phase1(config, device, out_dir, num_iterations=3000):
    """Phase 1: Recover z_FP, z_R, mu from synthetic B-scans with known ground truth."""
    print("\n" + "=" * 70)
    print("PHASE 1: Synthetic validation")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    D = int(config.forward_model_system.depth_samples)
    A = int(config.forward_model_system.a_lines)
    noise_floor_dB = float(config.forward_model_system.noise_floor_dB)
    dz = float(config.forward_model_system.axial_pixel_size_mm)

    strategies = ['fixed', 'joint', 'alternating']
    results = []

    for mu_gt_val in [3.0, 5.0, 8.0]:
        print(f"\n--- Ground truth mu = {mu_gt_val:.1f} mm^-1 ---")
        I_target, nf_mask, _ = _generate_synthetic_bscan(
            config, device, mu_value=mu_gt_val)

        # Analytical estimation
        I_np = I_target.cpu().numpy()
        nf_np = nf_mask.cpu().numpy()
        ana = analytical_confocal_estimation(I_np, noise_floor_dB, dz, intmask=nf_np)
        print(f"  Analytical: z_FP={ana['z_fp']:.1f} z_R={ana['z_r']:.1f} "
              f"mu_est={ana['mu_est']:.2f} (success={ana['success']})")

        results.append(dict(
            mu_gt=mu_gt_val, strategy='analytical',
            z_fp_final=ana['z_fp'], z_r_final=ana['z_r'],
            z_fp_err=abs(ana['z_fp'] - GT_ZFP), z_r_err=abs(ana['z_r'] - GT_ZR),
            mu_mean=ana['mu_est'], mse=float('nan'), time_sec=0.0,
        ))

        for strat in strategies:
            print(f"\n  Strategy: {strat}")

            # Perturbed initialization for confocal params
            z_fp_init = 180.0 if strat != 'fixed' else GT_ZFP
            z_r_init = 80.0 if strat != 'fixed' else GT_ZR

            model = ConfocalOptimizableForward(
                config, device, z_fp_init=z_fp_init, z_r_init=z_r_init)

            torch.manual_seed(42)
            mu = nn.Parameter(torch.empty(D, A, device=device).uniform_(0.0, 10.0))
            alpha = nn.Parameter(torch.ones(D, A, device=device))

            res = optimize_with_confocal(
                model, mu, alpha, I_target, nf_mask,
                strategy=strat, num_iterations=num_iterations)

            mu_mean, mu_std = _compute_tissue_mu_stats(res['mu_final'], nf_np)
            print(f"    Final: z_FP={res['z_fp_final']:.1f} z_R={res['z_r_final']:.1f} "
                  f"tissue_mu={mu_mean:.3f}+/-{mu_std:.3f}")

            results.append(dict(
                mu_gt=mu_gt_val, strategy=strat,
                z_fp_final=res['z_fp_final'], z_r_final=res['z_r_final'],
                z_fp_err=abs(res['z_fp_final'] - GT_ZFP),
                z_r_err=abs(res['z_r_final'] - GT_ZR),
                mu_mean=mu_mean, mse=res['best_mse'], time_sec=res['time_sec'],
            ))

            # Save convergence plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes[0, 0].plot(res['mse_history'])
            axes[0, 0].set_ylabel('MSE'); axes[0, 0].set_title('Loss convergence')
            axes[0, 0].set_yscale('log')
            axes[0, 1].plot(res['z_fp_history'], label='z_FP')
            axes[0, 1].axhline(GT_ZFP, color='r', ls='--', label=f'GT={GT_ZFP}')
            axes[0, 1].set_ylabel('z_FP (px)'); axes[0, 1].legend()
            axes[1, 0].plot(res['z_r_history'], label='z_R')
            axes[1, 0].axhline(GT_ZR, color='r', ls='--', label=f'GT={GT_ZR}')
            axes[1, 0].set_ylabel('z_R (px)'); axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].legend()
            axes[1, 1].plot(res['mu_mean_history'])
            axes[1, 1].axhline(mu_gt_val, color='r', ls='--', label=f'GT={mu_gt_val}')
            axes[1, 1].set_ylabel('Mean mu'); axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].legend()
            fig.suptitle(f'Phase 1: {strat} | mu_gt={mu_gt_val}')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'p1_{strat}_mu{mu_gt_val:.0f}.png'), dpi=150)
            plt.close()

            _cleanup(model, mu, alpha, I_target)

    # Save CSV
    csv_path = os.path.join(out_dir, 'phase1_results.csv')
    fields = ['mu_gt', 'strategy', 'z_fp_final', 'z_r_final', 'z_fp_err', 'z_r_err',
              'mu_mean', 'mse', 'time_sec']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")
    return results


# ===========================================================================
# Phase 2: Strategy comparison on real phantom data
# ===========================================================================

def run_phase2(config, device, out_dir, phantom_index=0, bscan_index=256,
               num_iterations=3000):
    """Phase 2: Compare strategies on real phantom data."""
    print("\n" + "=" * 70)
    print("PHASE 2: Strategy comparison on real phantom data")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    D = int(config.forward_model_system.depth_samples)
    A = int(config.forward_model_system.a_lines)
    noise_floor_dB = float(config.forward_model_system.noise_floor_dB)
    dz = float(config.forward_model_system.axial_pixel_size_mm)

    # Discover phantoms
    input_dir = os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206")
    phantoms = discover_despeckled_phantoms(input_dir)
    if not phantoms:
        print("ERROR: No phantoms found"); return []

    ph = phantoms[phantom_index]
    conc = ph['concentration']
    cf_ref = CF_REFERENCE.get(conc, 5.0)
    print(f"  Phantom: {ph['basename']} (conc={conc}%, CF ref={cf_ref} mm^-1)")

    # Load B-scan
    I_dB_np = load_phantom_bscan(ph['path'], bscan_index)
    I_target_dB = torch.tensor(I_dB_np, dtype=torch.float32, device=device)

    # N(z) subtraction
    I_target_dB = subtract_noise_floor_linear(I_target_dB, noise_floor_dB)
    nf_mask = build_noise_floor_mask(I_target_dB, noise_floor_dB, device)

    # Load IntMask for tissue stats
    intmask = load_intmask(
        os.path.join(PHANTOM_ROOT, "matlab_CF_scripts_results", "IntMask.mat"),
        phantom_index, bscan_index)
    intmask_np = intmask.cpu().numpy().astype(bool) if intmask is not None else None
    nf_np = nf_mask.cpu().numpy() > 0.5

    strategies = ['fixed', 'joint', 'alternating']
    results = []

    # Analytical estimation
    ana = analytical_confocal_estimation(
        I_target_dB.cpu().numpy(), noise_floor_dB, dz,
        intmask=intmask_np.astype(float) if intmask_np is not None else None)
    print(f"  Analytical: z_FP={ana['z_fp']:.1f} z_R={ana['z_r']:.1f} "
          f"success={ana['success']}")

    # Two-stage: analytical init → joint fine-tuning
    two_stage_z_fp = ana['z_fp'] if ana['success'] else 200.0
    two_stage_z_r = ana['z_r'] if ana['success'] else 100.0

    # Add two-stage to strategies list
    strategies_plus = strategies + ['two_stage']

    for strat in strategies_plus:
        print(f"\n  Strategy: {strat}")

        if strat == 'fixed':
            z_fp_init, z_r_init = GT_ZFP, GT_ZR
        elif strat == 'two_stage':
            z_fp_init, z_r_init = two_stage_z_fp, two_stage_z_r
        else:
            z_fp_init, z_r_init = 180.0, 80.0

        model = ConfocalOptimizableForward(
            config, device, z_fp_init=z_fp_init, z_r_init=z_r_init)

        torch.manual_seed(42)
        mu = nn.Parameter(torch.empty(D, A, device=device).uniform_(0.0, 10.0))
        alpha = nn.Parameter(torch.ones(D, A, device=device))

        actual_strat = 'joint' if strat == 'two_stage' else strat
        res = optimize_with_confocal(
            model, mu, alpha, I_target_dB, nf_mask,
            strategy=actual_strat, num_iterations=num_iterations)

        # Tissue stats
        mu_final = res['mu_final']
        if intmask_np is not None:
            combined = intmask_np & nf_np
            tissue_vals = mu_final[combined]
            tissue_vals = tissue_vals[tissue_vals > 0.01]
            mu_mean = float(tissue_vals.mean()) if len(tissue_vals) > 0 else float('nan')
            mu_std = float(tissue_vals.std()) if len(tissue_vals) > 0 else float('nan')
        else:
            mu_mean, mu_std = _compute_tissue_mu_stats(mu_final, nf_np.astype(float))

        mu_err_pct = abs(mu_mean - cf_ref) / cf_ref * 100 if not np.isnan(mu_mean) else float('nan')

        print(f"    Final: z_FP={res['z_fp_final']:.1f} z_R={res['z_r_final']:.1f} "
              f"tissue_mu={mu_mean:.3f}+/-{mu_std:.3f} err={mu_err_pct:.1f}%")

        results.append(dict(
            strategy=strat, z_fp_init=z_fp_init, z_r_init=z_r_init,
            z_fp_final=res['z_fp_final'], z_r_final=res['z_r_final'],
            z_fp_err=abs(res['z_fp_final'] - GT_ZFP),
            z_r_err=abs(res['z_r_final'] - GT_ZR),
            mu_mean=mu_mean, mu_std=mu_std, mu_err_pct=mu_err_pct,
            cf_ref=cf_ref, mse=res['best_mse'], time_sec=res['time_sec'],
        ))

        # Save convergence
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(res['mse_history']); axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('Loss'); axes[0, 0].set_yscale('log')
        axes[0, 1].plot(res['z_fp_history'])
        axes[0, 1].axhline(GT_ZFP, color='r', ls='--', label=f'GT={GT_ZFP}')
        axes[0, 1].set_ylabel('z_FP'); axes[0, 1].legend()
        axes[1, 0].plot(res['z_r_history'])
        axes[1, 0].axhline(GT_ZR, color='r', ls='--', label=f'GT={GT_ZR}')
        axes[1, 0].set_ylabel('z_R'); axes[1, 0].set_xlabel('Iter'); axes[1, 0].legend()
        axes[1, 1].plot(res['mu_mean_history'])
        axes[1, 1].set_ylabel('Mean mu'); axes[1, 1].set_xlabel('Iter')
        fig.suptitle(f'Phase 2: {strat} | phantom {phantom_index} conc={conc}%')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'p2_{strat}.png'), dpi=150)
        plt.close()

        _cleanup(model, mu, alpha)

    # Add analytical result
    results.insert(0, dict(
        strategy='analytical', z_fp_init=float('nan'), z_r_init=float('nan'),
        z_fp_final=ana['z_fp'], z_r_final=ana['z_r'],
        z_fp_err=abs(ana['z_fp'] - GT_ZFP), z_r_err=abs(ana['z_r'] - GT_ZR),
        mu_mean=ana['mu_est'], mu_std=float('nan'),
        mu_err_pct=abs(ana['mu_est'] - cf_ref) / cf_ref * 100,
        cf_ref=cf_ref, mse=float('nan'), time_sec=0.0,
    ))

    csv_path = os.path.join(out_dir, 'phase2_results.csv')
    fields = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")
    return results


# ===========================================================================
# Phase 3: Initialization sweep
# ===========================================================================

def run_phase3(config, device, out_dir, num_iterations=3000):
    """Phase 3: Sweep z_FP and z_R initial values on synthetic data."""
    print("\n" + "=" * 70)
    print("PHASE 3: Initialization sweep")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    D = int(config.forward_model_system.depth_samples)
    A = int(config.forward_model_system.a_lines)

    mu_gt_val = 5.0
    I_target, nf_mask, _ = _generate_synthetic_bscan(config, device, mu_value=mu_gt_val)
    nf_np = nf_mask.cpu().numpy()

    # Sweep grid
    z_fp_inits = [100.0, 150.0, 200.0, 231.0, 280.0, 350.0]
    z_r_inits = [30.0, 80.0, 131.5, 200.0, 350.0]

    results = []
    for z_fp_init in z_fp_inits:
        for z_r_init in z_r_inits:
            label = f"zFP{z_fp_init:.0f}_zR{z_r_init:.0f}"
            print(f"\n  Init: z_FP={z_fp_init}, z_R={z_r_init}")

            model = ConfocalOptimizableForward(
                config, device, z_fp_init=z_fp_init, z_r_init=z_r_init)
            torch.manual_seed(42)
            mu = nn.Parameter(torch.empty(D, A, device=device).uniform_(0.0, 10.0))
            alpha = nn.Parameter(torch.ones(D, A, device=device))

            res = optimize_with_confocal(
                model, mu, alpha, I_target, nf_mask,
                strategy='joint', num_iterations=num_iterations, verbose=False)

            mu_mean, _ = _compute_tissue_mu_stats(res['mu_final'], nf_np)

            results.append(dict(
                z_fp_init=z_fp_init, z_r_init=z_r_init,
                z_fp_final=res['z_fp_final'], z_r_final=res['z_r_final'],
                z_fp_err=abs(res['z_fp_final'] - GT_ZFP),
                z_r_err=abs(res['z_r_final'] - GT_ZR),
                mu_mean=mu_mean, mu_err=abs(mu_mean - mu_gt_val),
                mse=res['best_mse'], time_sec=res['time_sec'],
            ))
            print(f"    -> z_FP={res['z_fp_final']:.1f} z_R={res['z_r_final']:.1f} "
                  f"mu={mu_mean:.3f}")

            _cleanup(model, mu, alpha)

    # Heatmap plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, (metric, title) in enumerate([
        ('z_fp_err', 'z_FP error (px)'), ('z_r_err', 'z_R error (px)'),
        ('mu_err', 'mu error (mm^-1)')]):
        grid = np.full((len(z_r_inits), len(z_fp_inits)), np.nan)
        for r in results:
            i = z_r_inits.index(r['z_r_init'])
            j = z_fp_inits.index(r['z_fp_init'])
            grid[i, j] = r[metric]
        im = axes[ax_idx].imshow(grid, aspect='auto', origin='lower',
                                  cmap='RdYlGn_r')
        axes[ax_idx].set_xticks(range(len(z_fp_inits)))
        axes[ax_idx].set_xticklabels([f'{v:.0f}' for v in z_fp_inits])
        axes[ax_idx].set_yticks(range(len(z_r_inits)))
        axes[ax_idx].set_yticklabels([f'{v:.0f}' for v in z_r_inits])
        axes[ax_idx].set_xlabel('z_FP init (px)')
        axes[ax_idx].set_ylabel('z_R init (px)')
        axes[ax_idx].set_title(title)
        plt.colorbar(im, ax=axes[ax_idx])
        # Annotate
        for r in results:
            i = z_r_inits.index(r['z_r_init'])
            j = z_fp_inits.index(r['z_fp_init'])
            axes[ax_idx].text(j, i, f'{r[metric]:.1f}', ha='center', va='center',
                              fontsize=7, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'p3_init_sweep_heatmap.png'), dpi=150)
    plt.close()

    csv_path = os.path.join(out_dir, 'phase3_results.csv')
    fields = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")
    return results


# ===========================================================================
# Phase 4: Noise robustness
# ===========================================================================

def run_phase4(config, device, out_dir, num_iterations=3000):
    """Phase 4: Test confocal recovery under different noise levels."""
    print("\n" + "=" * 70)
    print("PHASE 4: Noise robustness")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    D = int(config.forward_model_system.depth_samples)
    A = int(config.forward_model_system.a_lines)

    noise_levels = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]  # dB std
    mu_gt_val = 5.0
    results = []

    for noise_std in noise_levels:
        print(f"\n  Noise std = {noise_std:.1f} dB")

        I_target, nf_mask, _ = _generate_synthetic_bscan(
            config, device, mu_value=mu_gt_val, noise_std_dB=noise_std)
        nf_np = nf_mask.cpu().numpy()

        model = ConfocalOptimizableForward(
            config, device, z_fp_init=180.0, z_r_init=80.0)
        torch.manual_seed(42)
        mu = nn.Parameter(torch.empty(D, A, device=device).uniform_(0.0, 10.0))
        alpha = nn.Parameter(torch.ones(D, A, device=device))

        res = optimize_with_confocal(
            model, mu, alpha, I_target, nf_mask,
            strategy='joint', num_iterations=num_iterations, verbose=False)

        mu_mean, _ = _compute_tissue_mu_stats(res['mu_final'], nf_np)

        results.append(dict(
            noise_std_dB=noise_std,
            z_fp_final=res['z_fp_final'], z_r_final=res['z_r_final'],
            z_fp_err=abs(res['z_fp_final'] - GT_ZFP),
            z_r_err=abs(res['z_r_final'] - GT_ZR),
            mu_mean=mu_mean, mu_err=abs(mu_mean - mu_gt_val),
            mse=res['best_mse'], time_sec=res['time_sec'],
        ))
        print(f"    -> z_FP={res['z_fp_final']:.1f} z_R={res['z_r_final']:.1f} "
              f"mu={mu_mean:.3f}")

        _cleanup(model, mu, alpha)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ns = [r['noise_std_dB'] for r in results]
    axes[0].plot(ns, [r['z_fp_err'] for r in results], 'o-')
    axes[0].set_xlabel('Noise std (dB)'); axes[0].set_ylabel('z_FP error (px)')
    axes[0].set_title('Focal plane recovery vs noise')
    axes[1].plot(ns, [r['z_r_err'] for r in results], 's-')
    axes[1].set_xlabel('Noise std (dB)'); axes[1].set_ylabel('z_R error (px)')
    axes[1].set_title('Rayleigh range recovery vs noise')
    axes[2].plot(ns, [r['mu_err'] for r in results], '^-')
    axes[2].set_xlabel('Noise std (dB)'); axes[2].set_ylabel('mu error (mm^-1)')
    axes[2].set_title('Attenuation recovery vs noise')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'p4_noise_robustness.png'), dpi=150)
    plt.close()

    csv_path = os.path.join(out_dir, 'phase4_results.csv')
    fields = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}")
    return results


# ===========================================================================
# Report generation
# ===========================================================================

def generate_report(out_dir, p1_results, p2_results, p3_results, p4_results):
    """Generate a detailed Markdown report with embedded images."""
    report_path = os.path.join(out_dir, 'report.md')

    lines = []
    lines.append("# Experiment 12: Confocal Parameter Optimization\n")
    lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Ground truth**: z_FP = {GT_ZFP} px, z_R = {GT_ZR} px\n")

    lines.append("## Background\n")
    lines.append("The confocal function in OCT models the depth-dependent beam intensity:\n")
    lines.append("```\nC(z) = 1 / [((z - z_FP) / z_R)^2 + 1]\n```\n")
    lines.append("where z_FP is the focal plane position and z_R is the Rayleigh range. "
                 "In clinical systems, these parameters may be unknown or vary across acquisitions. "
                 "This experiment tests whether z_FP and z_R can be recovered from the OCT data itself, "
                 "jointly with the attenuation coefficient mu.\n")

    lines.append("## Strategies Tested\n")
    lines.append("| Strategy | Description |\n|---|---|\n")
    lines.append("| **Fixed** | z_FP and z_R frozen at ground truth (baseline) |\n")
    lines.append("| **Joint** | Optimize mu, z_FP, z_R simultaneously with Adam (lr_confocal=0.5) |\n")
    lines.append("| **Alternating** | 80-step mu blocks, 20-step confocal blocks, repeating |\n")
    lines.append("| **Analytical** | Fit Lorentzian+exponential to depth-averaged profile |\n")
    lines.append("| **Two-stage** | Analytical init for z_FP, z_R, then joint fine-tuning |\n")

    # Phase 1
    lines.append("\n---\n## Phase 1: Synthetic Validation\n")
    lines.append("Synthetic B-scans with known ground truth (uniform mu slab, rows 200-350).\n")
    lines.append("Initial confocal: z_FP=180, z_R=80 (perturbed from GT).\n\n")

    if p1_results:
        lines.append("| mu_GT | Strategy | z_FP final | z_FP err | z_R final | z_R err | mu_mean | Time |\n")
        lines.append("|-------|----------|-----------|---------|----------|--------|---------|------|\n")
        for r in p1_results:
            lines.append(f"| {r['mu_gt']:.1f} | {r['strategy']} | "
                         f"{r['z_fp_final']:.1f} | {r['z_fp_err']:.1f} | "
                         f"{r['z_r_final']:.1f} | {r['z_r_err']:.1f} | "
                         f"{r['mu_mean']:.3f} | {r['time_sec']:.0f}s |\n")

        # Embed convergence plots for mu=5.0
        lines.append("\n### Convergence plots (mu_GT=5.0)\n")
        for strat in ['fixed', 'joint', 'alternating']:
            fname = f'p1_{strat}_mu5.png'
            if os.path.exists(os.path.join(out_dir, 'phase1', fname)):
                lines.append(f"#### {strat.title()}\n![{strat}](phase1/{fname})\n\n")

    # Phase 2
    lines.append("\n---\n## Phase 2: Strategy Comparison on Real Phantom Data\n")
    if p2_results:
        cf_ref = p2_results[0].get('cf_ref', 'N/A') if p2_results else 'N/A'
        lines.append(f"Phantom data, CF reference mu = {cf_ref} mm^-1.\n\n")
        lines.append("| Strategy | z_FP final | z_FP err | z_R final | z_R err | "
                     "mu_mean | mu_err% | MSE | Time |\n")
        lines.append("|----------|-----------|---------|----------|--------|"
                     "---------|---------|-----|------|\n")
        for r in p2_results:
            lines.append(f"| {r['strategy']} | {r['z_fp_final']:.1f} | "
                         f"{r['z_fp_err']:.1f} | {r['z_r_final']:.1f} | "
                         f"{r['z_r_err']:.1f} | {r['mu_mean']:.3f} | "
                         f"{r.get('mu_err_pct', float('nan')):.1f}% | "
                         f"{r.get('mse', float('nan')):.2e} | {r['time_sec']:.0f}s |\n")

        lines.append("\n### Convergence plots\n")
        for strat in ['fixed', 'joint', 'alternating', 'two_stage']:
            fname = f'p2_{strat}.png'
            if os.path.exists(os.path.join(out_dir, 'phase2', fname)):
                lines.append(f"#### {strat.replace('_', ' ').title()}\n"
                             f"![{strat}](phase2/{fname})\n\n")

    # Phase 3
    lines.append("\n---\n## Phase 3: Initialization Sensitivity\n")
    lines.append("Joint optimization on synthetic data (mu_GT=5.0) with different "
                 "initial z_FP and z_R values.\n\n")
    if p3_results:
        lines.append("![Initialization sweep heatmap](phase3/p3_init_sweep_heatmap.png)\n\n")
        lines.append("| z_FP init | z_R init | z_FP final | z_FP err | "
                     "z_R final | z_R err | mu_mean | mu_err |\n")
        lines.append("|-----------|----------|-----------|---------|"
                     "----------|--------|---------|--------|\n")
        for r in sorted(p3_results, key=lambda x: x['z_fp_err'] + x['z_r_err']):
            lines.append(f"| {r['z_fp_init']:.0f} | {r['z_r_init']:.0f} | "
                         f"{r['z_fp_final']:.1f} | {r['z_fp_err']:.1f} | "
                         f"{r['z_r_final']:.1f} | {r['z_r_err']:.1f} | "
                         f"{r['mu_mean']:.3f} | {r['mu_err']:.3f} |\n")

    # Phase 4
    lines.append("\n---\n## Phase 4: Noise Robustness\n")
    lines.append("Joint optimization on synthetic data (mu_GT=5.0) with additive "
                 "Gaussian noise.\n\n")
    if p4_results:
        lines.append("![Noise robustness](phase4/p4_noise_robustness.png)\n\n")
        lines.append("| Noise (dB) | z_FP final | z_FP err | z_R final | z_R err | "
                     "mu_mean | mu_err |\n")
        lines.append("|------------|-----------|---------|----------|--------|"
                     "---------|--------|\n")
        for r in p4_results:
            lines.append(f"| {r['noise_std_dB']:.1f} | {r['z_fp_final']:.1f} | "
                         f"{r['z_fp_err']:.1f} | {r['z_r_final']:.1f} | "
                         f"{r['z_r_err']:.1f} | {r['mu_mean']:.3f} | "
                         f"{r['mu_err']:.3f} |\n")

    # Conclusions
    lines.append("\n---\n## Key Findings\n\n")
    if p1_results:
        joint_p1 = [r for r in p1_results if r['strategy'] == 'joint' and r['mu_gt'] == 5.0]
        fixed_p1 = [r for r in p1_results if r['strategy'] == 'fixed' and r['mu_gt'] == 5.0]
        if joint_p1 and fixed_p1:
            j, f = joint_p1[0], fixed_p1[0]
            lines.append(f"1. **Synthetic recovery**: Joint optimization recovers z_FP within "
                         f"{j['z_fp_err']:.1f} px and z_R within {j['z_r_err']:.1f} px "
                         f"of ground truth (mu_GT=5.0).\n")
            lines.append(f"2. **Baseline comparison**: Fixed confocal (GT values) gives "
                         f"mu={f['mu_mean']:.3f}; joint gives mu={j['mu_mean']:.3f}.\n")

    if p3_results:
        converged = [r for r in p3_results if r['z_fp_err'] < 20 and r['z_r_err'] < 30]
        lines.append(f"3. **Convergence basin**: {len(converged)}/{len(p3_results)} "
                     f"init conditions converged within 20px of GT z_FP and 30px of GT z_R.\n")

    if p4_results:
        low_noise = [r for r in p4_results if r['noise_std_dB'] <= 1.0]
        high_noise = [r for r in p4_results if r['noise_std_dB'] >= 5.0]
        if low_noise and high_noise:
            ln_err = np.mean([r['z_fp_err'] + r['z_r_err'] for r in low_noise])
            hn_err = np.mean([r['z_fp_err'] + r['z_r_err'] for r in high_noise])
            lines.append(f"4. **Noise sensitivity**: Mean confocal error at low noise "
                         f"(<= 1 dB): {ln_err:.1f} px; at high noise (>= 5 dB): {hn_err:.1f} px.\n")

    with open(report_path, 'w') as f:
        f.writelines(lines)
    print(f"\nReport saved: {report_path}")
    return report_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    phase = int(os.environ.get('PHASE', '0'))  # 0 = run all
    summarize = int(os.environ.get('SUMMARIZE', '0'))
    phantom_index = int(os.environ.get('PHANTOM_INDEX', '0'))
    bscan_index = int(os.environ.get('BSCAN_INDEX', '256'))
    num_iterations = int(os.environ.get('NUM_ITERATIONS', '3000'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config, opt_config, sc_config = _load_configs()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    p1_results, p2_results, p3_results, p4_results = [], [], [], []

    if summarize:
        # Load existing results
        for phase_num, name, results_list in [
            (1, 'phase1', p1_results), (2, 'phase2', p2_results),
            (3, 'phase3', p3_results), (4, 'phase4', p4_results)]:
            csv_path = os.path.join(OUTPUT_BASE, f'phase{phase_num}', f'{name}_results.csv')
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        converted = {}
                        for k, v in row.items():
                            try:
                                converted[k] = float(v)
                            except (ValueError, TypeError):
                                converted[k] = v
                        results_list.append(converted)
        report_path = generate_report(OUTPUT_BASE, p1_results, p2_results,
                                      p3_results, p4_results)
        print(f"Report: {report_path}")
        return

    t_total = time.time()

    if phase in (0, 1):
        p1_results = run_phase1(config, device,
                                os.path.join(OUTPUT_BASE, 'phase1'), num_iterations)

    if phase in (0, 2):
        p2_results = run_phase2(config, device,
                                os.path.join(OUTPUT_BASE, 'phase2'),
                                phantom_index, bscan_index, num_iterations)

    if phase in (0, 3):
        p3_results = run_phase3(config, device,
                                os.path.join(OUTPUT_BASE, 'phase3'), num_iterations)

    if phase in (0, 4):
        p4_results = run_phase4(config, device,
                                os.path.join(OUTPUT_BASE, 'phase4'), num_iterations)

    # Generate report
    report_path = generate_report(OUTPUT_BASE, p1_results, p2_results,
                                  p3_results, p4_results)

    total_time = time.time() - t_total
    print(f"\nTotal time: {total_time / 60:.1f} min")
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()
