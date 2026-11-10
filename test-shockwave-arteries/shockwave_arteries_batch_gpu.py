#!/usr/bin/env python3
"""
Shockwave Arteries Batched GPU Processing
==========================================
Processes multiple B-scan frames SIMULTANEOUSLY on one GPU by stacking
mu/alpha into (B, D, A) tensors.  This dramatically improves GPU utilization:
the benchmark shows only ~30 MB per frame, so a 32 GB Volta GPU can handle
hundreds of frames at once instead of processing them one-by-one.

The batched forward model and optimization loop are implemented locally
(no changes to src/) to support the (B, D, A) tensor layout.

Usage:
    # Worker mode (called by SLURM array tasks):
    PATIENT_ID=02008Post START_FRAME=0 END_FRAME=168 python shockwave_arteries_batch_gpu.py

    # Assembly mode (called after all workers finish):
    PATIENT_ID=02008Post ASSEMBLE=1 python shockwave_arteries_batch_gpu.py

Environment variables:
    PATIENT_ID        Patient prefix (required)
    START_FRAME       First frame index to process (worker mode)
    END_FRAME         Last frame index (exclusive) to process (worker mode)
    ASSEMBLE          Set to "1" to run assembly instead of optimization
    BATCH_SIZE        Override auto-computed batch size (default: auto)
    NUM_ITERS         Optimization iterations per stage (default: 25000)
    DB_SCALE          Pixel-to-dB scale factor (default: 130.0)
    NOISE_FLOOR_DB    Noise floor threshold in dB (default: 6.0)
    AXIAL_PIXEL_SIZE  Axial pixel size in mm (default: 0.020)
"""

import os
import sys
import math
import time
import shutil
import importlib.util
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import from sibling file (directory has hyphens, can't use normal import)
_sap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'shockwave_arteries_process.py')
_spec = importlib.util.spec_from_file_location('shockwave_arteries_process', _sap_path)
sap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sap)


# =========================================================================
# Batched forward model (local implementation, no changes to src/)
# =========================================================================

import torch
import torch.nn as nn


class BatchedSSForward(nn.Module):
    """Batched single-scatter Beer-Lambert forward model.

    Operates on (B, D, A) tensors where B = batch (frames), D = depth, A = A-lines.
    Each frame has its own gain_dB scalar, stored as a (B,) buffer.

    Equation per frame:
        I_dB(z) = gain_dB + 10*log10(alpha) + 10*log10(mu)
                  + confocal_dB(z) - K * cumsum_shifted(mu * dz)
    """

    def __init__(self, config, batch_size, device='cuda'):
        super().__init__()
        self.D = int(config.forward_model_system.depth_samples)
        self.A = int(config.forward_model_system.a_lines)
        self.axial_pixel_size_mm = float(config.forward_model_system.axial_pixel_size_mm)

        z_px = torch.arange(self.D, dtype=torch.float32, device=device)
        focal = float(config.forward_model_system.focal_plane_position_pixels)
        zR = float(config.forward_model_system.rayleigh_range_pixels)

        if bool(config.forward_model_system.confocal_correction):
            confocal = 1.0 / (((z_px - focal) / zR) ** 2 + 1.0)
        else:
            confocal = torch.ones(self.D, dtype=torch.float32, device=device)

        # confocal_dB: (D,) -> broadcasts as (1, D, 1)
        self.register_buffer('confocal_dB', 10.0 * torch.log10(confocal))
        # Per-frame gain: (B,)
        self.register_buffer('gain_dB',
                             torch.zeros(batch_size, dtype=torch.float32, device=device))
        self.register_buffer('_K',
                             torch.tensor(20.0 / math.log(10.0),
                                          dtype=torch.float32, device=device))

    def forward(self, mu, alpha, noise_mask=None):
        """Forward pass on batched tensors.

        Args:
            mu: (B, D, A) attenuation [mm^-1]
            alpha: (B, D, A) backscattering fraction
            noise_mask: (B, D, A) binary mask or None

        Returns:
            I_dB: (B, D, A) predicted B-scan in dB
        """
        eps = 1e-12
        mu_active = mu * noise_mask if noise_mask is not None else mu
        alpha_active = alpha * noise_mask if noise_mask is not None else alpha

        # Cumulative optical depth along depth (dim=1)
        cumulative = torch.cumsum(mu_active * self.axial_pixel_size_mm, dim=1)
        # Shift: prepend zeros row per batch
        zeros = torch.zeros(mu.shape[0], 1, mu.shape[2],
                            device=mu.device, dtype=mu.dtype)
        shifted = torch.cat([zeros, cumulative[:, :-1, :]], dim=1)

        I_dB = (self.gain_dB.view(-1, 1, 1)           # (B, 1, 1)
                + 10.0 * torch.log10(alpha_active + eps)
                + 10.0 * torch.log10(mu_active + eps)
                + self.confocal_dB.view(1, -1, 1)      # (1, D, 1)
                - self._K * shifted)
        return I_dB


# =========================================================================
# Batched optimization loop
# =========================================================================

def batched_optimize(forward_model, mu, alpha, targets, noise_masks,
                     opt_config, prune_threshold, device,
                     recompute_gain=True, log_interval_frac=10):
    """Run batched gradient-based optimization of mu and alpha.

    Args:
        forward_model: BatchedSSForward instance
        mu: nn.Parameter (B, D, A)
        alpha: nn.Parameter (B, D, A)
        targets: (B, D, A) target dB images
        noise_masks: (B, D, A) binary masks (or None)
        opt_config: OmegaConf config
        prune_threshold: mu values below this -> mu_min
        device: torch device
        recompute_gain: whether to recompute per-frame gain each iter
        log_interval_frac: print status every N/log_interval_frac iters

    Returns:
        mu, alpha, per_frame_best_mse (B,)
    """
    lr = float(opt_config.optimization.learning_rate)
    num_iterations = int(opt_config.optimization.num_iterations)
    loss_scale = float(opt_config.optimization.loss_scale)
    mu_min = float(getattr(opt_config.optimization, 'mu_min_mm_inv', 0.001))
    mu_max = float(opt_config.optimization.mu_max_mm_inv)
    alpha_min = float(getattr(opt_config.optimization, 'alpha_min', 1e-8))
    alpha_max = float(getattr(opt_config.optimization, 'alpha_max', 1e-2))
    max_delta_mu = float(opt_config.optimization.max_delta_mu_mm)
    mu_tv_weight = float(getattr(opt_config.optimization, 'mu_tv_weight', 0.0))
    lr_scheduler_type = str(getattr(opt_config.optimization, 'lr_scheduler', 'none'))
    lr_min_factor = float(getattr(opt_config.optimization, 'lr_min_factor', 0.01))

    eps = 1e-12
    B = mu.shape[0]

    # Zero mu/alpha at noise-floor pixels
    if noise_masks is not None:
        with torch.no_grad():
            mu.data *= noise_masks
            alpha.data *= noise_masks

    # Auto-init gain_dB per frame
    with torch.no_grad():
        forward_model.gain_dB.fill_(0.0)
        pred_init = forward_model(mu, alpha, noise_mask=noise_masks)
        if noise_masks is not None:
            # Per-frame median over tissue pixels
            for b in range(B):
                tissue = noise_masks[b] > 0.5
                if tissue.any():
                    pred_med = pred_init[b][tissue].median()
                    tgt_med = targets[b][tissue].median()
                    forward_model.gain_dB[b] = (tgt_med - pred_med).item()
        else:
            for b in range(B):
                forward_model.gain_dB[b] = (targets[b].median() - pred_init[b].median()).item()

    # Optimizer and scheduler
    optimizer = torch.optim.Adam([mu, alpha], lr=lr)
    scheduler = None
    if lr_scheduler_type == 'cosine':
        eta_min = lr * lr_min_factor
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iterations, eta_min=eta_min)

    # Best-model tracking (per frame)
    best_mse = torch.full((B,), float('inf'), device=device)
    best_mu = mu.detach().clone()
    best_alpha = alpha.detach().clone()
    best_gain = forward_model.gain_dB.clone()

    log_interval = max(1, num_iterations // log_interval_frac)

    print(f"  Batched optimization: B={B}, iters={num_iterations}, lr={lr}, "
          f"mu=[{mu_min},{mu_max}], alpha=[{alpha_min:.1e},{alpha_max:.1e}]")
    if mu_tv_weight > 0:
        print(f"  TV weight: {mu_tv_weight:.1e}")

    t_start = time.time()

    for it in range(num_iterations + 1):
        # --- Single forward pass ---
        optimizer.zero_grad(set_to_none=False)
        I_pred = forward_model(mu, alpha, noise_mask=noise_masks)

        # Analytical per-frame gain correction (no extra forward pass needed).
        # Since I_pred = gain_dB + f(mu,alpha), optimal gain adjustment is
        # just the mean residual over tissue pixels.  gain_adj is detached so
        # gradients still flow through I_pred -> mu.
        if recompute_gain:
            with torch.no_grad():
                if noise_masks is not None:
                    gain_adj = ((targets - I_pred) * noise_masks).sum(dim=(1, 2)) / (noise_masks.sum(dim=(1, 2)) + eps)
                else:
                    gain_adj = (targets - I_pred).mean(dim=(1, 2))
                forward_model.gain_dB.add_(gain_adj)
            I_pred = I_pred + gain_adj.view(-1, 1, 1)

        # Per-frame masked MSE -- use .sum() NOT .mean() so each frame's
        # gradient is identical to the single-frame case (no 1/B scaling)
        diff = (I_pred - targets) ** 2
        if noise_masks is not None:
            per_frame = (diff * noise_masks).sum(dim=(1, 2)) / (noise_masks.sum(dim=(1, 2)) + eps)
        else:
            per_frame = diff.mean(dim=(1, 2))

        # --- Track best per frame (before step, consistent with current mu) ---
        with torch.no_grad():
            frame_mse = per_frame.detach()
            improved = frame_mse < best_mse
            if improved.any():
                for b in range(B):
                    if improved[b]:
                        best_mse[b] = frame_mse[b]
                        best_mu[b] = mu.data[b]
                        best_alpha[b] = alpha.data[b]
                        best_gain[b] = forward_model.gain_dB[b]

        loss = loss_scale * per_frame.sum()

        # Per-frame TV regularization -- computed independently per frame, then summed
        if mu_tv_weight > 0:
            mu_active = mu * noise_masks if noise_masks is not None else mu
            dz = (mu_active[:, 1:, :] - mu_active[:, :-1, :]).abs()
            dx = (mu_active[:, :, 1:] - mu_active[:, :, :-1]).abs()
            if noise_masks is not None:
                mask_dz = noise_masks[:, 1:, :] * noise_masks[:, :-1, :]
                mask_dx = noise_masks[:, :, 1:] * noise_masks[:, :, :-1]
                tv_per_frame = (dz * mask_dz).sum(dim=(1, 2)) / (mask_dz.sum(dim=(1, 2)) + eps) \
                             + (dx * mask_dx).sum(dim=(1, 2)) / (mask_dx.sum(dim=(1, 2)) + eps)
            else:
                tv_per_frame = dz.mean(dim=(1, 2)) + dx.mean(dim=(1, 2))
            loss = loss + mu_tv_weight * tv_per_frame.sum()

        loss.backward()

        # --- Gradient post-processing ---
        for p in [mu, alpha]:
            if p.grad is not None:
                with torch.no_grad():
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    if lr > 0 and max_delta_mu < 1e6:
                        p.grad.clamp_(min=-max_delta_mu / lr, max=max_delta_mu / lr)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # --- Projection ---
        with torch.no_grad():
            mu.data.clamp_(min=mu_min, max=mu_max)
            alpha.data.clamp_(min=alpha_min, max=alpha_max)

            prune_mask = mu.data < prune_threshold
            if prune_mask.any():
                mu.data[prune_mask] = mu_min

            if noise_masks is not None:
                mu.data.mul_(noise_masks)
                alpha.data.mul_(noise_masks)

        if it % log_interval == 0 or it == num_iterations:
            elapsed = time.time() - t_start
            cur_lr = optimizer.param_groups[0]['lr']
            avg_mse = frame_mse.mean().item()
            mu_mean = mu.data.mean().item()
            print(f"    Iter {it:5d}/{num_iterations} | "
                  f"MSE={avg_mse:.4e} (best avg={best_mse.mean().item():.4e}) | "
                  f"mu_mean={mu_mean:.4f} | lr={cur_lr:.2e} | "
                  f"{elapsed:.1f}s")

    # Restore best per frame
    with torch.no_grad():
        mu.data.copy_(best_mu)
        alpha.data.copy_(best_alpha)
        forward_model.gain_dB.copy_(best_gain)

    total_time = time.time() - t_start
    print(f"  Done: {total_time:.1f}s total, {total_time / (num_iterations + 1):.4f}s/iter, "
          f"{total_time / B:.1f}s/frame")
    print(f"  Best MSE: mean={best_mse.mean().item():.4e}, "
          f"min={best_mse.min().item():.4e}, max={best_mse.max().item():.4e}")

    return mu, alpha, best_mse


# =========================================================================
# Batch size auto-computation
# =========================================================================

def compute_batch_size(depth, alines, device):
    """Estimate max batch size from available GPU memory.

    Conservative estimate: ~60 bytes per pixel per frame
    (params + grads + Adam states + forward intermediates + target + mask).
    """
    if device.type != 'cuda':
        return 4  # CPU: small batches

    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory
    # Reserve 2 GB for CUDA overhead, fragmentation, etc.
    available = total_mem - 2 * (1024 ** 3)

    bytes_per_pixel = 80  # conservative (measured ~30MB for 500*500=250K pixels -> ~120 B/px)
    bytes_per_frame = depth * alines * bytes_per_pixel

    batch = max(1, int(available / bytes_per_frame))
    print(f"  GPU memory: {total_mem / 1e9:.1f} GB, "
          f"available: {available / 1e9:.1f} GB, "
          f"est. {bytes_per_frame / 1e6:.1f} MB/frame -> "
          f"max batch size: {batch}")
    return batch


# =========================================================================
# Diagnostic panel saving (per frame, after batched optimization)
# =========================================================================

def save_frame_panel(mu_init_cart, mu_cart, pred_dB_cart, target_dB_cart,
                     frame_idx, frame_dir, patient_id):
    """Save a 4-panel diagnostic plot in Cartesian coordinates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    mu_vmin, mu_vmax = 0, 20.0

    im0 = axes[0, 0].imshow(mu_init_cart, cmap='hot', aspect='equal',
                              vmin=mu_vmin, vmax=mu_vmax)
    axes[0, 0].set_title('Initial mu (mm$^{-1}$)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(mu_cart, cmap='hot', aspect='equal',
                              vmin=mu_vmin, vmax=mu_vmax)
    axes[0, 1].set_title('Final mu (mm$^{-1}$)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    dB_vmin = target_dB_cart[target_dB_cart > 0].min() if (target_dB_cart > 0).any() else 0
    dB_vmax = target_dB_cart.max()

    im2 = axes[1, 0].imshow(pred_dB_cart, cmap='gray', aspect='equal',
                              vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 0].set_title('Predicted dB')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(target_dB_cart, cmap='gray', aspect='equal',
                              vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 1].set_title('Target B-scan dB')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'{patient_id} -- Frame {frame_idx}', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(frame_dir, f'frame_{frame_idx:03d}_panel.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def save_alpha_panel(alpha_init_cart, alpha_cart, pred_dB_cart, target_dB_cart,
                     frame_idx, frame_dir, patient_id):
    """Save a 4-panel diagnostic plot for alpha in Cartesian coordinates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    alpha_norm = LogNorm(vmin=1e-10, vmax=1e-2)

    alpha_init_display = np.clip(alpha_init_cart, 1e-12, None)
    alpha_display = np.clip(alpha_cart, 1e-12, None)

    im0 = axes[0, 0].imshow(alpha_init_display, cmap='viridis', aspect='equal',
                              norm=alpha_norm)
    axes[0, 0].set_title('Initial alpha (log scale)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(alpha_display, cmap='viridis', aspect='equal',
                              norm=alpha_norm)
    axes[0, 1].set_title('Final alpha (log scale)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    dB_vmin = target_dB_cart[target_dB_cart > 0].min() if (target_dB_cart > 0).any() else 0
    dB_vmax = target_dB_cart.max()

    im2 = axes[1, 0].imshow(pred_dB_cart, cmap='gray', aspect='equal',
                              vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 0].set_title('Predicted dB')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(target_dB_cart, cmap='gray', aspect='equal',
                              vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 1].set_title('Target B-scan dB')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'{patient_id} -- Frame {frame_idx} -- Alpha', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(frame_dir, f'frame_{frame_idx:03d}_alpha_panel.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
# Batch processing
# =========================================================================

def process_batch(frame_indices, patient_data, seg_data, pixel_max,
                  db_scale, noise_floor_dB, config, opt_config,
                  device, out_base, patient_id):
    """Process a batch of frames simultaneously on GPU.

    Two-stage optimization:
      Stage 1: optimize mu (alpha frozen at 1.0)
      Stage 2: optimize alpha (mu frozen from stage 1)
    """
    from src.inverse import subtract_noise_floor_linear, build_noise_floor_mask
    from omegaconf import OmegaConf

    B = len(frame_indices)
    D = sap.DEPTH_PIXELS
    A = sap.NUM_ALINES
    prune_threshold = 0.05

    print(f"\n{'='*70}")
    print(f"Processing batch of {B} frames: {frame_indices[0]}-{frame_indices[-1]}")
    print(f"{'='*70}")

    # --- Check which frames need processing ---
    frames_to_process = []
    frame_to_batch_idx = {}
    for i, fi in enumerate(frame_indices):
        frame_dir = os.path.join(out_base, f'frame_{fi:03d}')
        mu_out = os.path.join(frame_dir, 'mu_final_cartesian_mm.npy')
        alpha_out = os.path.join(frame_dir, 'alpha_final_cartesian.npy')
        if os.path.exists(mu_out) and os.path.exists(alpha_out):
            print(f"  Frame {fi}: already processed, skipping")
        else:
            frame_to_batch_idx[fi] = len(frames_to_process)
            frames_to_process.append(fi)

    if not frames_to_process:
        print("  All frames in batch already processed.")
        return

    B_actual = len(frames_to_process)
    print(f"  {B_actual} frames to process (skipped {B - B_actual} already done)")

    # --- Prepare all frames: Cartesian -> polar -> dB -> N(z) subtraction ---
    polar_imgs_dB = np.zeros((B_actual, D, A), dtype=np.float32)
    polar_imgs_raw = []  # keep for diagnostics

    t_prep = time.time()
    for i, fi in enumerate(frames_to_process):
        frame = patient_data[:, :, fi]
        polar_img = sap.cartesian_to_polar(frame)
        if pixel_max > 0:
            polar_dB = polar_img * (db_scale / pixel_max)
        else:
            polar_dB = polar_img * db_scale
        polar_imgs_dB[i] = polar_dB.astype(np.float32)
    print(f"  Polar conversion: {time.time() - t_prep:.1f}s")

    # Move to GPU and apply N(z) subtraction per frame
    targets_gpu = torch.tensor(polar_imgs_dB, dtype=torch.float32, device=device)
    noise_masks_gpu = torch.zeros_like(targets_gpu)

    for i in range(B_actual):
        targets_gpu[i] = subtract_noise_floor_linear(targets_gpu[i], noise_floor_dB)
        noise_masks_gpu[i] = build_noise_floor_mask(targets_gpu[i], noise_floor_dB, device)

    # --- Stage 1: Optimize mu (alpha frozen at 1.0) ---
    print(f"\n  ===== Stage 1: mu optimization (B={B_actual}) =====")
    forward_s1 = BatchedSSForward(config, batch_size=B_actual, device=str(device)).to(device)

    mu = torch.nn.Parameter(
        torch.empty(B_actual, D, A, device=device).uniform_(0.0, 10.0))
    alpha_frozen = torch.nn.Parameter(
        torch.ones(B_actual, D, A, device=device))

    # Capture initial mu for diagnostics
    mu_init_polar = mu.detach().cpu().numpy().copy()

    t_s1 = time.time()
    mu_opt, _, best_mse_s1 = batched_optimize(
        forward_s1, mu, alpha_frozen, targets_gpu, noise_masks_gpu,
        opt_config, prune_threshold, device, recompute_gain=True)
    t_s1_elapsed = time.time() - t_s1
    print(f"  Stage 1 total: {t_s1_elapsed:.1f}s ({t_s1_elapsed/B_actual:.1f}s/frame)")

    # Get stage 1 predictions for mu diagnostic panels
    with torch.no_grad():
        pred_s1 = forward_s1(mu_opt, alpha_frozen, noise_mask=noise_masks_gpu)
    pred_s1_np = pred_s1.cpu().numpy()

    # Freeze mu
    mu_frozen = mu_opt.detach().clone()
    mu_frozen_np = mu_frozen.cpu().numpy()

    # Free stage 1 model
    del forward_s1, mu, alpha_frozen, pred_s1
    torch.cuda.empty_cache()

    # --- Stage 2: Optimize alpha (mu frozen) ---
    print(f"\n  ===== Stage 2: alpha optimization (B={B_actual}) =====")
    forward_s2 = BatchedSSForward(config, batch_size=B_actual, device=str(device)).to(device)

    mu_param_s2 = torch.nn.Parameter(mu_frozen.clone())
    mu_param_s2.requires_grad_(False)
    alpha_s2 = torch.nn.Parameter(
        torch.full((B_actual, D, A), 1e-3, device=device))

    # Capture initial alpha for diagnostics
    alpha_init_polar = alpha_s2.detach().cpu().numpy().copy()

    opt_config_s2 = OmegaConf.create(OmegaConf.to_container(opt_config, resolve=True))
    opt_config_s2.optimization.alpha_min = 1e-10
    opt_config_s2.optimization.alpha_max = 1e-2

    t_s2 = time.time()
    _, alpha_opt, best_mse_s2 = batched_optimize(
        forward_s2, mu_param_s2, alpha_s2, targets_gpu, noise_masks_gpu,
        opt_config_s2, prune_threshold, device, recompute_gain=True)
    t_s2_elapsed = time.time() - t_s2
    print(f"  Stage 2 total: {t_s2_elapsed:.1f}s ({t_s2_elapsed/B_actual:.1f}s/frame)")

    # Final predictions for alpha diagnostic panels
    with torch.no_grad():
        pred_s2 = forward_s2(mu_param_s2, alpha_opt, noise_mask=noise_masks_gpu)
    pred_s2_np = pred_s2.cpu().numpy()

    alpha_opt_np = alpha_opt.detach().cpu().numpy()
    targets_np = targets_gpu.cpu().numpy()

    # Free GPU memory
    del forward_s2, mu_param_s2, alpha_s2, alpha_opt, mu_opt
    del targets_gpu, noise_masks_gpu, pred_s2, mu_frozen
    torch.cuda.empty_cache()

    # --- Save results: convert polar -> Cartesian and save per frame ---
    print(f"\n  Saving {B_actual} frames to disk...")
    t_save = time.time()
    for i, fi in enumerate(frames_to_process):
        frame_dir = os.path.join(out_base, f'frame_{fi:03d}')
        os.makedirs(frame_dir, exist_ok=True)

        # Convert to Cartesian
        mu_cart = sap.polar_to_cartesian(mu_frozen_np[i], fill_value=0.0).astype(np.float32)
        alpha_cart = sap.polar_to_cartesian(alpha_opt_np[i], fill_value=0.0).astype(np.float32)

        # Save npy
        mu_out = os.path.join(frame_dir, 'mu_final_cartesian_mm.npy')
        alpha_out = os.path.join(frame_dir, 'alpha_final_cartesian.npy')
        np.save(mu_out, mu_cart)
        np.save(alpha_out, alpha_cart)

        # Save diagnostic panels (only for first and last frame in batch to save time)
        if i == 0 or i == B_actual - 1 or B_actual <= 10:
            mu_init_cart = sap.polar_to_cartesian(mu_init_polar[i], fill_value=0.0).astype(np.float32)
            pred_s1_cart = sap.polar_to_cartesian(pred_s1_np[i], fill_value=0.0).astype(np.float32)
            target_cart = sap.polar_to_cartesian(targets_np[i], fill_value=0.0).astype(np.float32)
            alpha_init_cart = sap.polar_to_cartesian(alpha_init_polar[i], fill_value=0.0).astype(np.float32)
            pred_s2_cart = sap.polar_to_cartesian(pred_s2_np[i], fill_value=0.0).astype(np.float32)

            save_frame_panel(mu_init_cart, mu_cart, pred_s1_cart, target_cart,
                             fi, frame_dir, patient_id)
            save_alpha_panel(alpha_init_cart, alpha_cart, pred_s2_cart, target_cart,
                             fi, frame_dir, patient_id)

    print(f"  Save + Cartesian conversion: {time.time() - t_save:.1f}s")

    # Clean up stage subdirectories
    for i, fi in enumerate(frames_to_process):
        frame_dir = os.path.join(out_base, f'frame_{fi:03d}')
        for item in os.listdir(frame_dir):
            path = os.path.join(frame_dir, item)
            if os.path.isdir(path):
                shutil.rmtree(path)


# =========================================================================
# Assembly (reuse from shockwave_arteries_batch.py)
# =========================================================================

def assemble_nifti(patient_id, out_base, patient_path):
    """Assemble per-frame .npy files into OAC and Alpha NIfTI volumes."""
    import nibabel as nib

    ref_img = nib.load(patient_path)
    ref_affine = ref_img.affine
    num_frames = ref_img.shape[2]
    h, w = ref_img.shape[0], ref_img.shape[1]

    print(f"\nAssembling NIfTI volumes for {patient_id}")
    print(f"  Reference shape: ({h}, {w}, {num_frames})")

    mu_volume = np.zeros((h, w, num_frames), dtype=np.float32)
    alpha_volume = np.zeros((h, w, num_frames), dtype=np.float32)

    found = 0
    missing = []
    for fi in range(num_frames):
        frame_dir = os.path.join(out_base, f'frame_{fi:03d}')
        mu_path = os.path.join(frame_dir, 'mu_final_cartesian_mm.npy')
        alpha_path = os.path.join(frame_dir, 'alpha_final_cartesian.npy')

        if os.path.exists(mu_path) and os.path.exists(alpha_path):
            mu_volume[:, :, fi] = np.load(mu_path)
            alpha_volume[:, :, fi] = np.load(alpha_path)
            found += 1
        else:
            missing.append(fi)

    print(f"  Found {found}/{num_frames} frames")
    if missing:
        print(f"  WARNING: Missing frames: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    oac_path = os.path.join(out_base, f'{patient_id}_OAC.nii.gz')
    alpha_nifti_path = os.path.join(out_base, f'{patient_id}_Alpha.nii.gz')

    oac_img = nib.Nifti1Image(mu_volume, ref_affine)
    nib.save(oac_img, oac_path)
    print(f"  Saved OAC volume: {oac_path}")
    print(f"    Shape: {mu_volume.shape}, mu range: [{mu_volume.min():.4f}, {mu_volume.max():.4f}] mm^-1")

    alpha_img = nib.Nifti1Image(alpha_volume, ref_affine)
    nib.save(alpha_img, alpha_nifti_path)
    print(f"  Saved Alpha volume: {alpha_nifti_path}")
    print(f"    Shape: {alpha_volume.shape}, alpha range: [{alpha_volume.min():.6e}, {alpha_volume.max():.6e}]")

    # Keep frame 0 panels, clean up rest
    if found == num_frames and not missing:
        frame0_dir = os.path.join(out_base, 'frame_000')
        if os.path.isdir(frame0_dir):
            for fname in os.listdir(frame0_dir):
                if fname.endswith('.png'):
                    shutil.copy2(os.path.join(frame0_dir, fname),
                                 os.path.join(out_base, fname))
                    print(f"  Kept diagnostic: {fname}")

        print(f"\n  Cleaning up per-frame directories...")
        cleaned = 0
        for fi in range(num_frames):
            frame_dir = os.path.join(out_base, f'frame_{fi:03d}')
            if os.path.isdir(frame_dir):
                shutil.rmtree(frame_dir)
                cleaned += 1
        print(f"  Removed {cleaned} frame directories")
    else:
        print(f"\n  Skipping cleanup: {len(missing)} frames missing")

    return oac_path, alpha_nifti_path


# =========================================================================
# Main
# =========================================================================

def main():
    patient_id = os.environ.get('PATIENT_ID', '02008Pre')
    assemble_mode = os.environ.get('ASSEMBLE', '0') == '1'
    num_iters = int(os.environ.get('NUM_ITERS', '25000'))
    db_scale = float(os.environ.get('DB_SCALE', '130.0'))
    noise_floor_dB = float(os.environ.get('NOISE_FLOOR_DB', '6.0'))
    axial_pixel_size_mm = float(os.environ.get('AXIAL_PIXEL_SIZE', '0.020'))
    batch_size_override = os.environ.get('BATCH_SIZE', '')

    # Find patient files
    patient_path, seg_path = sap.find_patient_files(patient_id)

    # Output directory
    default_output_root = os.path.join(PROJECT_ROOT, 'outputs', 'patient_arteries')
    output_root = os.environ.get('OUTPUT_ROOT', default_output_root)
    out_base = os.path.join(output_root, patient_id)
    os.makedirs(out_base, exist_ok=True)

    if assemble_mode:
        assemble_nifti(patient_id, out_base, patient_path)
        return

    # Worker mode
    start_frame = int(os.environ.get('START_FRAME', '0'))
    end_frame_env = os.environ.get('END_FRAME', '')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*70}")
    print(f"Shockwave Arteries BATCHED GPU Worker")
    print(f"{'='*70}")
    print(f"Patient ID:       {patient_id}")
    print(f"Num iterations:   {num_iters} per stage")
    print(f"dB scale:         {db_scale}")
    print(f"Noise floor:      {noise_floor_dB} dB")
    print(f"Device:           {device}")
    if device.type == 'cuda':
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory:       {props.total_memory / 1e9:.1f} GB")

    # Load data
    print(f"\nLoading patient data...")
    patient_data = sap.load_nifti(patient_path)
    print(f"  Shape: {patient_data.shape}")
    seg_data = sap.load_nifti(seg_path)
    print(f"  Seg shape: {seg_data.shape}")

    num_frames = patient_data.shape[2]
    if not end_frame_env:
        end_frame = num_frames
    else:
        end_frame = min(int(end_frame_env), num_frames)

    start_frame = max(0, min(start_frame, num_frames))
    end_frame = max(start_frame, end_frame)

    total_to_process = end_frame - start_frame
    print(f"Frame range:      [{start_frame}, {end_frame}) of {num_frames}")
    print(f"Frames to process: {total_to_process}")
    print(f"Output directory: {out_base}")

    # Compute batch size
    max_batch = compute_batch_size(sap.DEPTH_PIXELS, sap.NUM_ALINES, device)
    if batch_size_override:
        batch_size = min(int(batch_size_override), total_to_process)
        print(f"  Batch size override: {batch_size}")
    else:
        batch_size = min(max_batch, total_to_process)
        print(f"  Auto batch size: {batch_size} (of {total_to_process} frames)")

    # Build configs
    config = sap.build_ivoct_config(
        depth_pixels=sap.DEPTH_PIXELS, num_alines=sap.NUM_ALINES,
        noise_floor_dB=noise_floor_dB, axial_pixel_size_mm=axial_pixel_size_mm)
    opt_config = sap.build_opt_config(num_iterations=num_iters)

    pixel_max = patient_data.max()
    t_total_start = time.time()

    # Process in batches
    all_frames = list(range(start_frame, end_frame))
    num_batches = (len(all_frames) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(all_frames))
        batch_frames = all_frames[batch_start:batch_end]

        print(f"\n{'#'*70}")
        print(f"# Batch {batch_idx + 1}/{num_batches}: "
              f"frames {batch_frames[0]}-{batch_frames[-1]} "
              f"({len(batch_frames)} frames)")
        print(f"{'#'*70}")

        process_batch(batch_frames, patient_data, seg_data, pixel_max,
                      db_scale, noise_floor_dB, config, opt_config,
                      device, out_base, patient_id)

        # Report GPU memory usage after first batch
        if batch_idx == 0 and device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e6
            print(f"\n  Peak GPU memory (batch 1): {peak_mem:.0f} MB")

    elapsed_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"Worker done: {total_to_process} frames in {elapsed_total:.1f}s "
          f"({elapsed_total / max(total_to_process, 1):.1f}s/frame)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
