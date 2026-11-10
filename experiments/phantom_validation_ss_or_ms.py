#!/usr/bin/env python
"""Phantom validation: SS or MS optimization across all Intralipid phantoms.

Runs the single-scatter (Beer-Lambert) or multiple-scatter (EHF) forward model
on all 10 despeckled Intralipid phantoms (5 concentrations x 2 replicates).
Focal plane is overridden to 261 px (MATLAB-optimized). Tissue mu statistics
use the IntMask binary mask. N(z) noise floor subtraction and per-iteration
MSE-optimal gain recomputation are enabled.

Three modes of operation:

  1. OPTIMIZATION (default) — Run inverse optimization on B-scan frames.
     Saves per-frame mu maps and IntMask-filtered mu pixel arrays.

  2. SUMMARIZE — Build a boxplot + summary table from completed results.
     Each datapoint is an individual pixel mu value (not a frame average).

  3. VISUALIZE_3D — Render publication-quality 3D volume visualizations
     (side-by-side: phantom dB in gray + recovered mu in inferno) using PyVista.

Usage:
    # --- Optimization ---

    # SS model (default), all phantoms, all 512 frames:
    python test/test_phantom_validation_ss_or_ms.py

    # MS model:
    MODEL=ms python test/test_phantom_validation_ss_or_ms.py

    # Process a subset of frames (e.g. for parallel SLURM jobs):
    START_FRAME=0 END_FRAME=64 python test/test_phantom_validation_ss_or_ms.py

    # Control GPU batch size (default: all frames in range processed simultaneously):
    BATCH_SIZE=128 python test/test_phantom_validation_ss_or_ms.py

    # --- Summary ---

    # Build boxplot + summary table from completed optimization results:
    SUMMARIZE=1 python test/test_phantom_validation_ss_or_ms.py
    MODEL=ms SUMMARIZE=1 python test/test_phantom_validation_ss_or_ms.py

    # --- 3D Visualization ---

    # Render 3D volumes for all phantoms:
    VISUALIZE_3D=1 python test/test_phantom_validation_ss_or_ms.py

    # Render a single phantom:
    VISUALIZE_3D=1 PHANTOM_INDEX=8 python test/test_phantom_validation_ss_or_ms.py

    # With downsampling and rotating GIF:
    VISUALIZE_3D=1 DOWNSAMPLE=2 RENDER_GIF=1 python test/test_phantom_validation_ss_or_ms.py

Environment variables:
    MODEL          Forward model type: 'ss' (default) or 'ms'
    START_FRAME    First B-scan frame index (default: 0)
    END_FRAME      Last B-scan frame index, exclusive (default: 512)
    SUMMARIZE      Set to 1 to build boxplot from existing results
    VISUALIZE_3D   Set to 1 to render 3D volume visualizations
    PHANTOM_INDEX  Restrict to a single phantom by index (0-9)
    DOWNSAMPLE     Volume downsampling factor for 3D rendering (default: 1)
    RENDER_GIF     Set to 1 to also render a 360-degree rotating GIF
"""

import gc
import os
import sys
import math
import time
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.helpers import discover_despeckled_phantoms, load_phantom_bscan, save_results

from src.forward import create_forward_model
from src.inverse import optimize, build_noise_floor_mask
from src.roi import load_intmask

# CF reference values [mm^-1] (averaged across focal positions)
CF_REFERENCE = {1.0: 4.523, 1.5: 5.914, 2.0: 5.878, 3.0: 7.742, 5.0: 9.346}

# ---- Adjustment parameters ----
ADJUSTED_FOCAL_PLANE_PX = 261      # MATLAB-optimized (was 231)

TOTAL_FRAMES = 512
MODEL_TYPE = os.environ.get("MODEL", "ss").lower()
assert MODEL_TYPE in ("ss", "ms"), f"MODEL must be 'ss' or 'ms', got '{MODEL_TYPE}'"
OUT_SUBDIR = os.environ.get("OUT_SUBDIR", f"phantom_validation_{MODEL_TYPE}")
MODEL_TAG = MODEL_TYPE  # subdirectory name inside each bscan dir


def _fit_line(x, y, mask):
    """Fit a line y = slope*x + intercept, return (slope, intercept, r2, coeffs)."""
    n = np.sum(mask)
    if n < 2:
        if n == 1:
            return 0.0, float(y[mask][0]), float("nan"), np.array([0.0, float(y[mask][0])])
        return None, None, None, None
    coeffs = np.polyfit(x[mask], y[mask], 1)
    pred = np.polyval(coeffs, x[mask])
    ss_res = np.sum((y[mask] - pred) ** 2)
    ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return coeffs[0], coeffs[1], r2, coeffs


def run_single_model_optimization(
    forward_model, model_tag, mu_init_data, alpha_init_data,
    I_target_dB, noise_floor_mask, opt_config, prune_threshold,
    device, run_out_dir,
):
    """Run optimization for the SS forward model."""
    model_dir = os.path.join(run_out_dir, model_tag)
    os.makedirs(model_dir, exist_ok=True)

    mu = nn.Parameter(mu_init_data.clone())
    alpha = nn.Parameter(alpha_init_data.clone())
    forward_model.gain_dB.fill_(0.0)

    t0 = time.time()
    mu_best, alpha_best, metrics, seq_steps, seq_imgs, mu_init, alpha_init = optimize(
        forward_model, mu, alpha, I_target_dB, opt_config, prune_threshold,
        device, model_dir, noise_floor_mask=noise_floor_mask,
        recompute_gain_mse_optimal_each_iter=True,
    )
    frame_time = time.time() - t0
    final_gain = float(forward_model.gain_dB.item())

    del mu, alpha, mu_init, alpha_init, seq_steps, seq_imgs
    return mu_best, alpha_best, metrics, frame_time, final_gain


# =====================================================================
# Batched processing utilities — all B-scans on GPU simultaneously
# =====================================================================

def load_all_bscans(tiff_path, start_frame=0, end_frame=512):
    """Load a contiguous range of B-scans from a multi-frame TIFF in dB scale.

    Returns:
        np.ndarray of shape (num_frames, 500, 500) float32 in dB scale.
    """
    try:
        import tifffile
        with tifffile.TiffFile(tiff_path) as tif:
            raw = tif.asarray()  # (total_frames, D, A) uint16
        return raw[start_frame:end_frame].astype(np.float32) * (130.0 / 65535.0)
    except ImportError:
        from PIL import Image
        frames = []
        img = Image.open(tiff_path)
        for i in range(start_frame, end_frame):
            img.seek(i)
            frames.append(np.array(img, dtype=np.float32) * (130.0 / 65535.0))
        return np.stack(frames)


def load_all_intmasks(phantom_index, mat_dir, start_frame=0, end_frame=512):
    """Load IntMask for a contiguous range of B-scans.

    Returns:
        np.ndarray of shape (num_frames, 500, 500) boolean.
    """
    from src.roi import load_intmask_from_mat
    mat_path = os.path.join(mat_dir, "IntMask.mat")
    all_masks = load_intmask_from_mat(mat_path, phantom_index, bscan_index=None)
    return all_masks[start_frame:end_frame]


def extract_fwd_params(forward_model, model_type):
    """Extract forward model buffers reshaped for batched (B,D,A) operation."""
    params = {
        'confocal_dB': forward_model.confocal_dB.unsqueeze(0).unsqueeze(2),  # (1,D,1)
        'K': forward_model._K,
        'dz': forward_model.axial_pixel_size_mm,
    }
    if model_type == 'ms':
        params['wH_sq'] = forward_model._wH_sq.unsqueeze(0).unsqueeze(2)
        params['theta_rms_sq'] = forward_model._theta_rms_sq
        params['z_over_n_sq'] = forward_model._z_over_n_sq.unsqueeze(0).unsqueeze(2)
    return params


def _batched_forward_ss(mu, alpha, noise_mask, gain_dB, confocal_dB, K, dz):
    """Batched SS Beer-Lambert forward model on (B,D,A) tensors."""
    eps = 1e-12
    mu_active = mu * noise_mask
    alpha_active = alpha * noise_mask
    cumulative = torch.cumsum(mu_active * dz, dim=1)
    shifted = torch.cat([
        torch.zeros(mu.shape[0], 1, mu.shape[2], device=mu.device, dtype=mu.dtype),
        cumulative[:, :-1, :],
    ], dim=1)
    return (gain_dB
            + 10.0 * torch.log10(alpha_active + eps)
            + 10.0 * torch.log10(mu_active + eps)
            + confocal_dB
            - K * shifted)


def _batched_forward_ms(mu, alpha, noise_mask, gain_dB, confocal_dB, K, dz,
                        wH_sq, theta_rms_sq, z_over_n_sq):
    """Batched MS EHF forward model on (B,D,A) tensors."""
    eps = 1e-12
    mu_active = mu * noise_mask
    alpha_active = alpha * noise_mask
    cumulative = torch.cumsum(mu_active * dz, dim=1)
    tau = torch.cat([
        torch.zeros(mu.shape[0], 1, mu.shape[2], device=mu.device, dtype=mu.dtype),
        cumulative[:, :-1, :],
    ], dim=1)
    wS_sq = wH_sq + (1.0 / 3.0) * tau * theta_rms_sq * z_over_n_sq
    exp_neg_tau = torch.exp(-tau)
    T_SS = exp_neg_tau * exp_neg_tau
    T_cross = 4.0 * exp_neg_tau * (1.0 - exp_neg_tau) / (1.0 + wS_sq / (wH_sq + eps))
    T_MS = (1.0 - exp_neg_tau).pow(2) * wH_sq / (wS_sq + eps)
    T_total = T_SS + T_cross + T_MS
    return (gain_dB
            + 10.0 * torch.log10(alpha_active + eps)
            + 10.0 * torch.log10(mu_active + eps)
            + confocal_dB
            + 10.0 * torch.log10(T_total + eps))


def _batched_forward(model_type, mu, alpha, noise_mask, gain_dB, fwd_params):
    """Dispatch to batched SS or MS forward model."""
    if model_type == 'ss':
        return _batched_forward_ss(
            mu, alpha, noise_mask, gain_dB,
            fwd_params['confocal_dB'], fwd_params['K'], fwd_params['dz'])
    return _batched_forward_ms(
        mu, alpha, noise_mask, gain_dB,
        fwd_params['confocal_dB'], fwd_params['K'], fwd_params['dz'],
        fwd_params['wH_sq'], fwd_params['theta_rms_sq'], fwd_params['z_over_n_sq'])


def _batched_tv(mu, noise_mask):
    """Per-frame anisotropic TV of mu, summed across frames.

    Each frame's TV is computed independently (its own tissue-pair
    normalisation), then the per-frame TV values are summed so that
    every frame receives the same gradient magnitude as in a
    single-frame optimisation.
    """
    eps = 1e-12
    dz = (mu[:, 1:, :] - mu[:, :-1, :]).abs()   # (B, D-1, A)
    dx = (mu[:, :, 1:] - mu[:, :, :-1]).abs()   # (B, D, A-1)
    mask_dz = noise_mask[:, 1:, :] * noise_mask[:, :-1, :]
    mask_dx = noise_mask[:, :, 1:] * noise_mask[:, :, :-1]
    # Per-frame TV: mean abs-diff over that frame's tissue pairs
    tv_z = (dz * mask_dz).sum(dim=(1, 2)) / (mask_dz.sum(dim=(1, 2)) + eps)  # (B,)
    tv_x = (dx * mask_dx).sum(dim=(1, 2)) / (mask_dx.sum(dim=(1, 2)) + eps)  # (B,)
    return (tv_z + tv_x).sum()  # sum across frames


def batched_optimize(model_type, fwd_params, frame_indices, phantom_index,
                     I_target_dB, noise_mask, opt_config, sc_config, config, device):
    """Run batched optimization for multiple B-scans simultaneously.

    Each frame's mu/alpha is optimized independently (no cross-frame coupling)
    but all frames share GPU kernel launches for massive parallelism.

    Performance features:
      - CUDA graph captures forward-backward; replayed each iteration
      - Gain recomputed every GAIN_INTERVAL iterations (default 100), not every iter
      - Best-model tracking piggybacks on gain recompute (no extra forward pass)

    Args:
        model_type: 'ss' or 'ms'
        fwd_params: dict from extract_fwd_params()
        frame_indices: list of int, the B-scan indices being processed
        phantom_index: int, phantom index for deterministic seeding
        I_target_dB: (B,D,A) N(z)-corrected targets on device
        noise_mask: (B,D,A) binary noise-floor mask on device
        opt_config, sc_config, config: OmegaConf configs
        device: torch device

    Returns:
        mu_init_np (B,D,A) numpy, mu_best (B,D,A), alpha_best (B,D,A),
        best_mse (B,), gain_final (B,), best_iter (B,)
    """
    B, D, A = I_target_dB.shape
    eps = 1e-12

    # ── Read optimisation hyper-parameters ──
    lr = float(opt_config.optimization.learning_rate)
    num_iters = int(opt_config.optimization.num_iterations)
    loss_scale = float(opt_config.optimization.loss_scale)
    mu_min = float(getattr(opt_config.optimization, 'mu_min_mm_inv', 0.001))
    mu_max = float(opt_config.optimization.mu_max_mm_inv)
    alpha_min_v = float(getattr(opt_config.optimization, 'alpha_min', 1e-8))
    alpha_max_v = float(getattr(opt_config.optimization, 'alpha_max', 1e-2))
    max_delta_mu = float(opt_config.optimization.max_delta_mu_mm)
    lr_sched = str(getattr(opt_config.optimization, 'lr_scheduler', 'none'))
    lr_min_factor = float(getattr(opt_config.optimization, 'lr_min_factor', 0.01))
    mu_tv_weight = float(getattr(opt_config.optimization, 'mu_tv_weight', 0.0))
    prune_thresh = float(config.patient.prune_threshold_m_inv)
    mu_init_lo = float(sc_config.scatterers.random_values_min)
    mu_init_hi = float(sc_config.scatterers.random_values_max)
    a_init_lo = float(getattr(sc_config.scatterers, 'alpha_init_min', 1e-8))
    a_init_hi = float(getattr(sc_config.scatterers, 'alpha_init_max', 1e-2))
    log_a_lo, log_a_hi = math.log10(a_init_lo), math.log10(a_init_hi)

    # ── Initialise mu & alpha with per-frame seeds (matches sequential) ──
    mu_data = torch.empty(B, D, A, dtype=torch.float32, device=device)
    alpha_data = torch.empty(B, D, A, dtype=torch.float32, device=device)
    for i, fi in enumerate(frame_indices):
        torch.manual_seed(42 + phantom_index * 1000 + fi)
        mu_data[i] = torch.empty(D, A, dtype=torch.float32, device=device).uniform_(mu_init_lo, mu_init_hi)
        alpha_data[i] = torch.pow(
            10.0,
            torch.empty(D, A, dtype=torch.float32, device=device).uniform_(log_a_lo, log_a_hi),
        )

    # Zero noise-floor pixels
    mu_data *= noise_mask
    alpha_data *= noise_mask

    # Snapshot initial mu before optimisation (moved to CPU immediately)
    mu_init_np = mu_data.cpu().numpy().copy()

    mu = nn.Parameter(mu_data)
    alpha = nn.Parameter(alpha_data)

    # ── Per-frame gain auto-init (median matching) ──
    gain_dB = torch.zeros(B, 1, 1, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_init = _batched_forward(model_type, mu, alpha, noise_mask, gain_dB, fwd_params)
        for i in range(B):
            tissue = noise_mask[i] > 0.5
            if tissue.any():
                gain_dB[i, 0, 0] = I_target_dB[i][tissue].median() - pred_init[i][tissue].median()
    print(f"  gain_dB auto-init: mean={gain_dB.mean():.1f}, "
          f"range=[{gain_dB.min():.1f}, {gain_dB.max():.1f}]")

    # Capture the iter-0 prediction with the calibrated gain — needed by panel (e)
    # of Figure 3 (predicted B-scan from the scrambled init).
    with torch.no_grad():
        pred_init_calibrated = _batched_forward(
            model_type, mu, alpha, noise_mask, gain_dB, fwd_params).clone()

    # ── Optimiser & scheduler ──
    optimizer = torch.optim.Adam([mu, alpha], lr=lr)
    scheduler = None
    if lr_sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iters, eta_min=lr * lr_min_factor)

    # ── Per-frame best tracking ──
    tissue_counts = noise_mask.sum(dim=(1, 2))  # (B,)
    best_mse = torch.full((B,), float('inf'), device=device)
    best_mu = mu.detach().clone()
    best_alpha = alpha.detach().clone()
    best_iter = torch.zeros(B, dtype=torch.long, device=device)

    # Per-frame MSE history for panel (m) of Figure 3
    loss_history = torch.zeros(num_iters + 1, B, dtype=torch.float32, device=device)

    print(f"  Starting {num_iters} iterations, {B} frames batched, "
          f"lr={lr}, TV={mu_tv_weight:.0e}")
    t0 = time.time()

    pbar = tqdm(range(num_iters + 1), desc="  Optimizing", unit="it",
                file=sys.stdout, dynamic_ncols=False,
                bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] MSE={postfix}")
    for it in pbar:
        # ── Forward + loss + backward ──
        optimizer.zero_grad(set_to_none=False)
        I_pred = _batched_forward(model_type, mu, alpha, noise_mask, gain_dB, fwd_params)
        diff = (I_pred - I_target_dB) ** 2
        per_frame_mse = (diff * noise_mask).sum(dim=(1, 2)) / (tissue_counts + eps)
        loss = loss_scale * per_frame_mse.sum()
        if mu_tv_weight > 0:
            loss = loss + mu_tv_weight * _batched_tv(mu, noise_mask)
        loss.backward()

        # ── Step + projection ──
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            mu.data.clamp_(mu_min, mu_max)
            alpha.data.clamp_(alpha_min_v, alpha_max_v)
            mu.data[mu.data < prune_thresh] = mu_min
            mu.data.mul_(noise_mask)
            alpha.data.mul_(noise_mask)

            # ── Per-frame gain recompute (MSE-optimal, closed form) ──
            gain_dB.zero_()
            pred_ng = _batched_forward(model_type, mu, alpha, noise_mask, gain_dB, fwd_params)
            g_star = ((I_target_dB - pred_ng) * noise_mask).sum(dim=(1, 2)) / (tissue_counts + eps)
            gain_dB[:, 0, 0] = g_star

            # ── Best tracking (reuses pred_ng + gain) ──
            pred_cur = pred_ng + gain_dB
            frame_mse = ((pred_cur - I_target_dB) ** 2 * noise_mask).sum(dim=(1, 2)) / (tissue_counts + eps)
            loss_history[it] = frame_mse
            improved = frame_mse < best_mse
            if improved.any():
                idx = improved.nonzero(as_tuple=True)[0]
                best_mse[idx] = frame_mse[idx]
                best_iter[idx] = it
                best_mu[idx] = mu.data[idx].clone()
                best_alpha[idx] = alpha.data[idx].clone()

        if it % 50 == 0:
            pbar.set_postfix_str(f"{frame_mse.mean().item():.4e}")

    total_time = time.time() - t0
    print(f"  Done: {total_time:.1f}s total, {total_time / max(B, 1):.2f}s/frame effective")
    mean_best = best_mse.mean().item()
    print(f"  Best MSE: mean={mean_best:.4e}, "
          f"range=[{best_mse.min().item():.4e}, {best_mse.max().item():.4e}]")

    pred_init_np = pred_init_calibrated.cpu().numpy()
    loss_history_np = loss_history.cpu().numpy()  # (num_iters+1, B)
    return (mu_init_np, best_mu, best_alpha, best_mse,
            gain_dB[:, 0, 0], best_iter, pred_init_np, loss_history_np)


def _save_frame_panel(mu_init, mu_final, target_dB, pred_dB, noise_mask,
                      frame_index, mse_val, best_it, out_path):
    """Save a 2x2 summary panel for one frame.

    Layout:
        (0,0) Initial mu          (0,1) Target B-scan (dB)
        (1,0) Final mu (best)     (1,1) Predicted B-scan (dB)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Shared mu colour range across init and final
    mu_vmax = max(mu_init.max(), mu_final.max(), 0.1)

    # Mask noise-floor pixels to NaN for cleaner display
    mask = noise_mask > 0.5
    dB_vmin = float(target_dB[mask].min()) if mask.any() else float(target_dB.min())
    dB_vmax = float(target_dB[mask].max()) if mask.any() else float(target_dB.max())

    tgt_display = np.where(mask, target_dB, np.nan)
    pred_display = np.where(mask, pred_dB, np.nan)

    im0 = axes[0, 0].imshow(mu_init, cmap='plasma', aspect='auto',
                              origin='upper', vmin=0, vmax=mu_vmax)
    axes[0, 0].set_title('Initial OAC (mm$^{-1}$)', fontsize=11)
    axes[0, 0].set_ylabel('Depth (px)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(tgt_display, cmap='gray', aspect='auto',
                              origin='upper', vmin=dB_vmin, vmax=dB_vmax)
    axes[0, 1].set_title('Target B-scan (dB)', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label='dB')

    im2 = axes[1, 0].imshow(mu_final, cmap='plasma', aspect='auto',
                              origin='upper', vmin=0, vmax=mu_vmax)
    axes[1, 0].set_title(f'Final OAC (mm$^{{-1}}$) — best iter {best_it}', fontsize=11)
    axes[1, 0].set_ylabel('Depth (px)')
    axes[1, 0].set_xlabel('A-scan')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(pred_display, cmap='gray', aspect='auto',
                              origin='upper', vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 1].set_title(f'Predicted B-scan (dB) — MSE={mse_val:.3e}', fontsize=11)
    axes[1, 1].set_xlabel('A-scan')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label='dB')

    fig.suptitle(f'B-scan {frame_index}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_boxplot(out_base):
    """Build boxplot from all per-frame IntMask mu pixel values.

    Each datapoint is an individual pixel mu value, not a frame average.
    """
    _apply_paper_style()

    input_dir = os.environ.get(
        "PHANTOM_INPUT_DIR",
        os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206"),
    )
    phantoms = discover_despeckled_phantoms(input_dir)

    concentrations = sorted(set(p["concentration"] for p in phantoms))

    # Collect per-phantom pixel pools
    data_A, data_B = [], []
    means_A, means_B = [], []

    for conc in concentrations:
        reps = sorted(
            [p for p in phantoms if p["concentration"] == conc],
            key=lambda p: p["replicate"],
        )
        for rep_idx, (d_list, m_list) in enumerate([
            (data_A, means_A),
            (data_B, means_B),
        ]):
            if rep_idx < len(reps):
                rep = reps[rep_idx]
                label = f"IL_{conc}pct_rep{rep['replicate']}"
                # Load all per-frame IntMask mu files
                all_pixels = []
                for fi in range(TOTAL_FRAMES):
                    mu_path = os.path.join(
                        out_base, label, f"bscan_{fi:03d}", "intmask_mu.npy")
                    if os.path.exists(mu_path):
                        all_pixels.append(np.load(mu_path))
                if all_pixels:
                    pool = np.concatenate(all_pixels)
                    d_list.append(pool)
                    m_list.append(float(pool.mean()))
                    print(f"  {label}: {len(all_pixels)} frames, "
                          f"{len(pool)} pixels, mean={pool.mean():.3f}")
                else:
                    d_list.append(np.array([]))
                    m_list.append(np.nan)
                    print(f"  {label}: NO frames found")
            else:
                d_list.append(np.array([]))
                m_list.append(np.nan)

    conc_arr = np.array(concentrations)

    # Fit lines
    m_A = np.array(means_A)
    m_B = np.array(means_B)
    valid = lambda a: ~np.isnan(a)

    sl_A, int_A, r2_A, co_A = _fit_line(conc_arr, m_A, valid(m_A))
    sl_B, int_B, r2_B, co_B = _fit_line(conc_arr, m_B, valid(m_B))

    cf_arr = np.array([CF_REFERENCE.get(c, np.nan) for c in concentrations])
    sl_cf, int_cf, r2_cf, co_cf = _fit_line(conc_arr, cf_arr, valid(cf_arr))

    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    x_fit = np.linspace(min(concentrations) - 0.3, max(concentrations) + 0.3, 200)

    box_style = dict(
        widths=0.12, patch_artist=True,
        whiskerprops=dict(linestyle="--"),
        capprops=dict(color="black"),
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="+", markersize=5),
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=5),
        manage_ticks=False,
    )
    offset = 0.08

    for conc_i, conc in enumerate(concentrations):
        for data_list, pos_off, color in [
            (data_A, -offset, "#6baed6"),
            (data_B, +offset, "#fd8d3c"),
        ]:
            d = data_list[conc_i]
            if len(d) > 0:
                bp = ax.boxplot([d], positions=[conc + pos_off], **box_style)
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

    if co_A is not None:
        r2s = f"{r2_A:.2f}" if r2_A is not None and not np.isnan(r2_A) else "N/A"
        ax.plot(x_fit, np.polyval(co_A, x_fit), color="#2171b5", ls="-", lw=1.5,
                label=f"Rep 1: $\\mu = {int_A:.2f} + {sl_A:.2f}x$, $R^2$={r2s}")
    if co_B is not None:
        r2s = f"{r2_B:.2f}" if r2_B is not None and not np.isnan(r2_B) else "N/A"
        ax.plot(x_fit, np.polyval(co_B, x_fit), color="#d94701", ls="-", lw=1.5,
                label=f"Rep 2: $\\mu = {int_B:.2f} + {sl_B:.2f}x$, $R^2$={r2s}")
    if co_cf is not None:
        r2s = f"{r2_cf:.2f}" if r2_cf is not None and not np.isnan(r2_cf) else "N/A"
        ax.plot(x_fit, np.polyval(co_cf, x_fit), color="green", ls="--", lw=1.5,
                label=f"CF ref: $\\mu = {int_cf:.2f} + {sl_cf:.2f}x$, $R^2$={r2s}")
    for conc in concentrations:
        cf = CF_REFERENCE.get(conc)
        if cf is not None:
            ax.plot(conc, cf, "g^", ms=10, zorder=5)

    ax.set_xticks(concentrations)
    ax.set_xticklabels([f"{c:.1f}" for c in concentrations])
    ax.set_xlabel("Intralipid concentration (%)", fontsize=13)
    ax.set_ylabel("Attenuation coefficient (mm$^{-1}$)", fontsize=13)
    ax.set_title(f"{MODEL_TYPE.upper()} (zFP={ADJUSTED_FOCAL_PLANE_PX}) \u2014 IntMask tissue pixels "
                 f"(all {TOTAL_FRAMES} frames)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(min(concentrations) - 0.4, max(concentrations) + 0.4)
    ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_base, f"{MODEL_TYPE}_adjusted_boxplot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # ── Save pixel data as npz for downstream comparison figures ──
    npz_data = {"concentrations": np.array(concentrations)}
    for conc_i, conc in enumerate(concentrations):
        tag_a = f"IL_{conc}pct_rep1"
        tag_b = f"IL_{conc}pct_rep2"
        if len(data_A[conc_i]) > 0:
            npz_data[tag_a] = data_A[conc_i]
        if len(data_B[conc_i]) > 0:
            npz_data[tag_b] = data_B[conc_i]
    npz_path = os.path.join(out_base, "boxplot_pixel_data.npz")
    np.savez(npz_path, **npz_data)
    print(f"Saved: {npz_path}")

    # ── Summary table ──
    save_summary_table(phantoms, out_base, means_A, means_B, concentrations)


def save_summary_table(phantoms, out_base, means_A, means_B, concentrations):
    """Print and save a comparison table."""
    lines = []
    lines.append("=" * 90)
    lines.append(f"{MODEL_TYPE.upper()} Adjusted (zFP={ADJUSTED_FOCAL_PLANE_PX}) — All {TOTAL_FRAMES} frames, "
                 "IntMask pixel-level stats")
    lines.append("=" * 90)
    lines.append("")

    header = (f"{'Conc':>6s} {'Rep':>4s} | {'IntMask mu':>10s} "
              f"{'CF ref':>8s} | {'err%':>8s}")
    lines.append(header)
    lines.append("-" * len(header))

    for conc_i, conc in enumerate(concentrations):
        cf_ref = CF_REFERENCE.get(conc, float("nan"))
        for rep_label, m_list in [("A", means_A), ("B", means_B)]:
            mu = m_list[conc_i]
            err = abs(mu - cf_ref) / cf_ref * 100 if not np.isnan(cf_ref) and not np.isnan(mu) else float("nan")
            line = f"{conc:>6.1f} {rep_label:>4s} | {mu:>10.3f} {cf_ref:>8.3f} | {err:>7.1f}%"
            lines.append(line)

    lines.append("")
    text = "\n".join(lines)
    print(text)

    out_path = os.path.join(out_base, f"{MODEL_TYPE}_adjusted_summary.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"Saved: {out_path}")


def _apply_paper_style():
    """Set matplotlib rcParams for publication figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth': 1.5,
    })


def render_3d_volumes(out_base):
    """Render 3D volume visualizations for one or all phantoms."""
    from src.visualization import load_volumes, render_3d_volume

    input_dir = os.environ.get(
        "PHANTOM_INPUT_DIR",
        os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206"),
    )
    phantoms = discover_despeckled_phantoms(input_dir)

    phantom_index = os.environ.get("PHANTOM_INDEX")
    downsample = int(os.environ.get("DOWNSAMPLE", "1"))
    render_gif = int(os.environ.get("RENDER_GIF", "0"))

    if phantom_index is not None:
        indices = [int(phantom_index)]
    else:
        indices = list(range(len(phantoms)))

    for pidx in indices:
        if pidx >= len(phantoms):
            print(f"Skipping phantom index {pidx} (only {len(phantoms)} found)")
            continue
        phantom = phantoms[pidx]
        label = f"IL_{phantom['concentration']}pct_rep{phantom['replicate']}"
        print(f"\n{'='*70}")
        print(f"3D Volume Rendering: {label} ({MODEL_TYPE.upper()})")
        print(f"{'='*70}")

        try:
            phantom_db, mu_mm, tissue_mean = load_volumes(
                pidx, MODEL_TAG, out_base, phantom["path"], TOTAL_FRAMES,
            )
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        viz_dir = os.path.join(out_base, label, "3d_visualization")
        os.makedirs(viz_dir, exist_ok=True)
        png_path = os.path.join(viz_dir, f"3d_cross_section_{MODEL_TAG}.png")

        # Save the underlying volumes + camera params so the Figure 3 notebook
        # (panels j and k) can re-render without re-running the optimization.
        npz_path = os.path.join(viz_dir, f"render_data_{MODEL_TAG}.npz")
        np.savez(npz_path, db_vol=phantom_db.astype(np.float32),
                            mu_vol=mu_mm.astype(np.float32))
        params_path = os.path.join(viz_dir, f"render_params_{MODEL_TAG}.json")
        with open(params_path, "w") as f:
            import json as _json
            _json.dump(dict(
                pixel_size_mm=dict(
                    dx=float(getattr(load_volumes, "_dx", 0.0093)),
                    dy=float(getattr(load_volumes, "_dy", 0.0093)),
                    dz=float(getattr(load_volumes, "_dz", 0.004)),
                ),
                window_size=[3200, 1600],
                camera={},
                scalar_bar_db={},
                scalar_bar_mu={},
                tissue_mean=float(tissue_mean),
            ), f, indent=2)
        print(f"  [Figure 3] {npz_path}\n  [Figure 3] {params_path}")

        render_3d_volume(
            phantom_db, mu_mm, png_path, MODEL_TAG, pidx,
            tissue_mean=tissue_mean, downsample=downsample,
            render_gif=render_gif,
        )
        del phantom_db, mu_mm

    print(f"\n3D visualizations saved under: {out_base}")


def main():
    # ---- Check for 3D visualization mode ----
    if int(os.environ.get("VISUALIZE_3D", "0")):
        out_base = os.path.join(PHANTOM_ROOT, "outputs", OUT_SUBDIR)
        print(f"=== VISUALIZE_3D mode: rendering volumes from {out_base} ===")
        render_3d_volumes(out_base)
        _stage_to_figure_paper_3(out_base)
        return

    # ---- Check for summary mode ----
    if int(os.environ.get("SUMMARIZE", "0")):
        out_base = os.path.join(PHANTOM_ROOT, "outputs", OUT_SUBDIR)
        print(f"=== SUMMARIZE mode: building boxplot from {out_base} ===")
        save_boxplot(out_base)
        _stage_to_figure_paper_3(out_base)
        return

    print(f"=== {MODEL_TYPE.upper()} Adjusted Optimization (zFP={ADJUSTED_FOCAL_PLANE_PX}, IntMask only) ===")

    # ---- Load configs ----
    # CONFIG_PATH / OPT_CONFIG_PATH env vars let supp_parameter_ablation.py
    # inject per-condition overrides; otherwise fall back to repo defaults.
    cfg_path = os.environ.get(
        "CONFIG_PATH", os.path.join(PHANTOM_ROOT, "configs", "config.yaml"))
    opt_path = os.environ.get(
        "OPT_CONFIG_PATH",
        os.path.join(PHANTOM_ROOT, "configs", "optimization_config_fast.yaml"))
    print(f"Loading config from: {cfg_path}")
    print(f"Loading opt_config from: {opt_path}")
    config = OmegaConf.load(cfg_path)
    opt_config = OmegaConf.load(opt_path)
    sc_config = OmegaConf.load(os.path.join(PHANTOM_ROOT, "configs",
                                             "scatterer_config.yaml"))

    # ---- Override focal plane in-memory ----
    # Skip the override when a custom CONFIG_PATH is provided (e.g., from
    # supp_parameter_ablation.py sweeping focal_plane_position_pixels), so the
    # caller-supplied value is preserved.
    original_zfp = float(config.forward_model_system.focal_plane_position_pixels)
    if "CONFIG_PATH" in os.environ:
        print(f"Custom CONFIG_PATH set; preserving zFP={original_zfp} px (no override)")
    else:
        config.forward_model_system.focal_plane_position_pixels = ADJUSTED_FOCAL_PLANE_PX
        print(f"Focal plane override: {original_zfp} -> {ADJUSTED_FOCAL_PLANE_PX} px")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Create forward model (SS or MS) ----
    config_fwd = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config_fwd.forward_model_system.forward_model_type = MODEL_TYPE
    forward_model = create_forward_model(config_fwd, device=device)
    forward_model.to(device)

    input_dir = os.environ.get(
        "PHANTOM_INPUT_DIR",
        os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206"),
    )
    phantoms = discover_despeckled_phantoms(input_dir)
    print(f"\nFound {len(phantoms)} despeckled phantom files:")
    for i, p in enumerate(phantoms):
        print(f"  [{i}] {p['concentration']}% (rep {p['replicate']}): {p['basename']}")

    # ---- Frame range ----
    start_frame = int(os.environ.get("START_FRAME", 0))
    end_frame = int(os.environ.get("END_FRAME", str(TOTAL_FRAMES)))
    out_base = os.path.join(PHANTOM_ROOT, "outputs", OUT_SUBDIR)
    os.makedirs(out_base, exist_ok=True)
    mat_dir = os.path.join(PHANTOM_ROOT, "matlab_CF_scripts_results")

    noise_floor_dB = float(config.forward_model_system.noise_floor_dB)
    mu_min = float(sc_config.scatterers.random_values_min)
    mu_max = float(sc_config.scatterers.random_values_max)
    alpha_init_min = float(getattr(sc_config.scatterers, "alpha_init_min", 1e-8))
    alpha_init_max = float(getattr(sc_config.scatterers, "alpha_init_max", 1e-2))
    log_alpha_min = math.log10(alpha_init_min)
    log_alpha_max = math.log10(alpha_init_max)
    prune_threshold = float(config.patient.prune_threshold_m_inv)

    D = int(config.forward_model_system.depth_samples)
    A = int(config.forward_model_system.a_lines)

    print(f"\nFrame range: [{start_frame}, {end_frame})")
    print(f"Output: {out_base}")
    print(f"noise_floor_dB: {noise_floor_dB}")
    print(f"N(z) subtraction: ENABLED")
    print(f"mu init range: [{mu_min}, {mu_max}] mm^-1")
    print(f"Model: {MODEL_TYPE.upper()} ({'Beer-Lambert' if MODEL_TYPE == 'ss' else 'EHF, g=' + str(config_fwd.forward_model_system.g)})")

    batch_size = int(os.environ.get("BATCH_SIZE", str(end_frame - start_frame)))
    print(f"Batch size: {batch_size} frames per GPU chunk")

    # Extract forward model buffers once for batched operation
    fwd_params = extract_fwd_params(forward_model, MODEL_TYPE)

    # Optional: process only a single phantom
    phantom_index_filter = os.environ.get("PHANTOM_INDEX")
    if phantom_index_filter is not None:
        phantom_index_filter = int(phantom_index_filter)
        print(f"PHANTOM_INDEX={phantom_index_filter} — processing only that phantom")

    total_start = time.time()

    for phantom_index, phantom in enumerate(phantoms):
        if phantom_index_filter is not None and phantom_index != phantom_index_filter:
            continue
        conc = phantom["concentration"]
        rep = phantom["replicate"]
        label = f"IL_{conc}pct_rep{rep}"

        print(f"\n{'='*70}")
        print(f"[{phantom_index + 1}/{len(phantoms)}] {label} | "
              f"frames {start_frame}-{end_frame - 1}")
        print(f"  File: {phantom['basename']}")

        # ── Identify frames that still need processing ──
        needed_frames = []
        for fi in range(start_frame, end_frame):
            frame_dir = os.path.join(out_base, label, f"bscan_{fi:03d}")
            intmask_path = os.path.join(frame_dir, "intmask_mu.npy")
            mu_path = os.path.join(frame_dir, MODEL_TAG, "mu_final_mm.npy")
            if not (os.path.exists(intmask_path) and os.path.exists(mu_path)):
                needed_frames.append(fi)

        if not needed_frames:
            print(f"  All frames already complete, skipping")
            continue

        n_range = end_frame - start_frame
        print(f"  {len(needed_frames)}/{n_range} frames need processing")

        # ── Bulk-load B-scans and IntMasks ──
        t_load = time.time()
        all_dB = load_all_bscans(phantom["path"], start_frame, end_frame)
        all_intmask = load_all_intmasks(phantom_index, mat_dir, start_frame, end_frame)

        # Select only frames that need processing
        sel = [fi - start_frame for fi in needed_frames]
        bscans_dB = all_dB[sel]
        intmasks = all_intmask[sel]
        del all_dB, all_intmask

        # Transfer to GPU, apply N(z) subtraction (batched element-wise)
        I_target_batch = torch.tensor(bscans_dB, dtype=torch.float32, device=device)
        del bscans_dB
        N_lin = 10.0 ** (noise_floor_dB / 10.0)
        I_lin = torch.pow(10.0, I_target_batch / 10.0)
        I_target_batch = 10.0 * torch.log10(torch.clamp(I_lin - N_lin, min=1e-12))
        del I_lin

        # Build noise-floor mask (batched)
        noise_mask_batch = (I_target_batch >= noise_floor_dB).float()

        # Combined IntMask AND noise-floor (numpy, for post-opt stats)
        nf_np = noise_mask_batch.cpu().numpy() > 0.5
        combined_masks = intmasks & nf_np
        del nf_np, intmasks

        tissue_frac = noise_mask_batch.mean().item()
        print(f"  Data loaded in {time.time() - t_load:.1f}s, "
              f"tissue fraction: {tissue_frac:.1%}")

        # ── Process in chunks of BATCH_SIZE ──
        B_total = len(needed_frames)
        for c0 in range(0, B_total, batch_size):
            c1 = min(c0 + batch_size, B_total)
            chunk_frames = needed_frames[c0:c1]
            Bc = len(chunk_frames)

            if B_total > batch_size:
                print(f"\n  Chunk [{c0}:{c1}] ({Bc} frames)")

            t_chunk = time.time()
            (mu_init_np, mu_best, alpha_best, best_mse_t, gain_final,
             best_iter_t, pred_init_np, loss_history_np) = batched_optimize(
                MODEL_TYPE, fwd_params, chunk_frames, phantom_index,
                I_target_batch[c0:c1], noise_mask_batch[c0:c1],
                opt_config, sc_config, config, device,
            )
            chunk_elapsed = time.time() - t_chunk
            per_frame_time = chunk_elapsed / max(Bc, 1)

            # ── Compute final predicted B-scans for panels ──
            with torch.no_grad():
                gain_3d = gain_final.unsqueeze(1).unsqueeze(2)  # (Bc, 1, 1)
                I_pred_final = _batched_forward(
                    MODEL_TYPE, mu_best, alpha_best,
                    noise_mask_batch[c0:c1], gain_3d, fwd_params)
            I_pred_np = I_pred_final.cpu().numpy()
            target_np = I_target_batch[c0:c1].cpu().numpy()
            nf_mask_np = noise_mask_batch[c0:c1].cpu().numpy()
            best_mse_np = best_mse_t.cpu().numpy()
            best_iter_np = best_iter_t.cpu().numpy()
            gain_final_np = gain_final.cpu().numpy()

            # ── Save per-frame results + panels ──
            mu_best_np = mu_best.cpu().numpy()
            cf_ref = CF_REFERENCE.get(conc, float("nan"))
            all_tissue_px = []

            # Showcase frame for Figure 3 panels d-i + m: by default 5.0% rep2 / B-scan 256.
            SHOWCASE_PHANTOM_INDEX = int(os.environ.get("SHOWCASE_PHANTOM_INDEX", 9))
            SHOWCASE_BSCAN_INDEX   = int(os.environ.get("SHOWCASE_BSCAN_INDEX", 256))

            for i, fi in enumerate(chunk_frames):
                frame_dir = os.path.join(out_base, label, f"bscan_{fi:03d}")
                model_dir = os.path.join(frame_dir, MODEL_TAG)
                os.makedirs(model_dir, exist_ok=True)

                np.save(os.path.join(model_dir, "mu_final_mm.npy"), mu_best_np[i])

                # Showcase-only extras (panels d-i + m of Figure 3) — written
                # only for the matching (phantom, B-scan) so we do not bloat
                # the output tree across all 5,120 frames.
                if (phantom_index == SHOWCASE_PHANTOM_INDEX
                        and fi == SHOWCASE_BSCAN_INDEX):
                    diff_dB = target_np[i] - I_pred_np[i]
                    np.save(os.path.join(model_dir, "mu_init_mm.npy"),         mu_init_np[i])
                    np.save(os.path.join(model_dir, "target_dB.npy"),          target_np[i])
                    np.save(os.path.join(model_dir, "predicted_init_dB.npy"),  pred_init_np[i])
                    np.save(os.path.join(model_dir, "predicted_final_dB.npy"), I_pred_np[i])
                    np.save(os.path.join(model_dir, "difference_dB.npy"),      diff_dB)
                    np.save(os.path.join(model_dir, "noise_floor_mask.npy"),   nf_mask_np[i].astype(np.uint8))
                    np.save(os.path.join(model_dir, "loss_curve.npy"),         loss_history_np[:, i])
                    print(f"  [showcase] Wrote panel d-i + m extras to {model_dir}")

                cmask = combined_masks[c0 + i]
                mu_tissue = mu_best_np[i][cmask]
                np.save(os.path.join(frame_dir, "intmask_mu.npy"),
                        mu_tissue.astype(np.float32))
                # Persist per-frame optimization stats so summarize_phase
                # (and any downstream notebook) can populate Final-MSE rows.
                np.savez(
                    os.path.join(frame_dir, "intmask_stats.npz"),
                    best_mse=np.float32(best_mse_np[i]),
                    best_iter=np.int64(best_iter_np[i]),
                    gain_dB=np.float32(gain_final_np[i]),
                    frame_time=np.float32(per_frame_time),
                )
                all_tissue_px.append(mu_tissue)

                # Summary panel: init mu | target | final mu | predicted
                _save_frame_panel(
                    mu_init_np[i], mu_best_np[i],
                    target_np[i], I_pred_np[i], nf_mask_np[i],
                    fi, float(best_mse_np[i]), int(best_iter_np[i]),
                    os.path.join(model_dir, "summary_panel.png"),
                )

            pooled = np.concatenate(all_tissue_px) if all_tissue_px else np.array([])
            mu_mean = float(pooled.mean()) if len(pooled) > 0 else float("nan")
            err = abs(mu_mean - cf_ref) / cf_ref * 100 if not np.isnan(cf_ref) and not np.isnan(mu_mean) else float("nan")
            print(f"  Result: mu={mu_mean:.3f} ({len(pooled)} tissue px), "
                  f"CF ref={cf_ref:.3f}, err={err:.1f}%")

            del mu_best, alpha_best, mu_best_np
            torch.cuda.empty_cache()

        del I_target_batch, noise_mask_batch, combined_masks
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All done in {total_time / 60:.1f} min")
    print(f"Outputs under: {out_base}")

    _stage_to_figure_paper_3(out_base)



def _stage_to_figure_paper_3(out_base):
    """Mirror the bare-minimum subset for the Figure 3 notebook.

    Copies into ``figures/data/figureS2_agar_phantom/``:
      * phantom_bscan_256/ — the eight per-frame npy files (panels d-i + m)
        produced for the showcase frame (SHOWCASE_PHANTOM_INDEX=9,
        SHOWCASE_BSCAN_INDEX=256 by default), plus mu_final_mm.npy.
      * phantom_volume_3d/ — render_data_{model}.npz + render_params_{model}.json
        (panels j and k).
      * boxplot_pixel_data.npz — per-phantom IntMask μ pixels (panel l).

    The full per-frame outputs remain under outputs/{out_subdir}/.
    """
    # S2 is the single-scattering agar comparison. S5 reads MS results from
    # the raw experiment folder through supp_ehf.py.
    if MODEL_TYPE != "ss":
        print("  [stage] MS results retained in their run folder; S2 uses SS results")
        return
    # Skip when invoked from supp_parameter_ablation.py (CONFIG_PATH is set
    # only for ablation subprocess runs). Otherwise each ablation condition's
    # showcase frame would overwrite the canonical phantom_validation output
    # in figures/data/figureS2_agar_phantom/, corrupting Figure 3.
    if "CONFIG_PATH" in os.environ:
        print("  [stage] CONFIG_PATH set (ablation subprocess) — skipping figureS2_agar_phantom staging")
        return
    import shutil
    fig_paper_3 = os.path.join(PHANTOM_ROOT, "figures", "data", "figureS2_agar_phantom")
    bscan_dst   = os.path.join(fig_paper_3, "phantom_bscan_256")
    vol_dst     = os.path.join(fig_paper_3, "phantom_volume_3d")
    os.makedirs(bscan_dst, exist_ok=True)
    os.makedirs(vol_dst,   exist_ok=True)

    SHOWCASE_PIDX  = int(os.environ.get("SHOWCASE_PHANTOM_INDEX", 9))
    SHOWCASE_BSCAN = int(os.environ.get("SHOWCASE_BSCAN_INDEX", 256))

    # Phantom label used by the per-phantom output dirs ("IL_<conc>pct_rep<r>")
    input_dir = os.environ.get(
        "PHANTOM_INPUT_DIR",
        os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206"),
    )
    phantoms = discover_despeckled_phantoms(input_dir)
    if SHOWCASE_PIDX >= len(phantoms):
        print(f"  [stage] showcase phantom index {SHOWCASE_PIDX} out of range; skipping")
        return
    sp = phantoms[SHOWCASE_PIDX]
    label = f"IL_{sp['concentration']}pct_rep{sp['replicate']}"
    src_bscan = os.path.join(out_base, label, f"bscan_{SHOWCASE_BSCAN:03d}", MODEL_TAG)

    showcase_files = ["mu_init_mm.npy", "mu_final_mm.npy",
                       "target_dB.npy", "predicted_init_dB.npy",
                       "predicted_final_dB.npy", "difference_dB.npy",
                       "noise_floor_mask.npy", "loss_curve.npy"]
    n_copied = 0
    for f in showcase_files:
        src = os.path.join(src_bscan, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(bscan_dst, f))
            n_copied += 1
    print(f"  [stage \u2192 figureS2_agar_phantom] {n_copied}/{len(showcase_files)} showcase frame files into {bscan_dst}")

    # 3D volume cross-section (panels j, k) — stage only the rendered PNG so the
    # data subset stays under git size limits. Notebook splits it into j+k halves.
    src_viz = os.path.join(out_base, label, "3d_visualization")
    rendered_png = os.path.join(src_viz, f"3d_cross_section_{MODEL_TAG}.png")
    if os.path.isfile(rendered_png):
        shutil.copy2(rendered_png,
                     os.path.join(vol_dst, f"3d_cross_section_{MODEL_TAG}.png"))
        print(f"  [stage \u2192 figureS2_agar_phantom] 3d_cross_section_{MODEL_TAG}.png "
              f"into {vol_dst}")
    # Also keep the params json (small, useful for re-rendering)
    params_src = os.path.join(src_viz, f"render_params_{MODEL_TAG}.json")
    if os.path.isfile(params_src):
        shutil.copy2(params_src, os.path.join(vol_dst, f"render_params_{MODEL_TAG}.json"))

    # Boxplot data (panel l)
    src_bp = os.path.join(out_base, "boxplot_pixel_data.npz")
    if os.path.isfile(src_bp):
        shutil.copy2(src_bp, os.path.join(fig_paper_3, "boxplot_pixel_data.npz"))
        print(f"  [stage \u2192 figureS2_agar_phantom] boxplot_pixel_data.npz")
    else:
        print(f"  [stage] {src_bp} missing; run with SUMMARIZE=1 first")


if __name__ == "__main__":
    main()
