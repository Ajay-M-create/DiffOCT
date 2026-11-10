import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def subtract_noise_floor_linear(
    I_target_dB: torch.Tensor,
    noise_floor_dB: float,
) -> torch.Tensor:
    """Subtract the noise floor from a dB-scale target in linear space.

    Per Cannon thesis Eq. 2.13, the measured OCT signal is:
        I_measured(z) = I_signal(z) + N(z)
    where N(z) is additive in LINEAR space.  The corrected signal is:
        I_corrected(z) = I_measured(z) - N(z)

    Since our data is in dB, the operation is:
        I_corrected_dB = 10·log10(10^(I_dB/10) - 10^(noise_floor_dB/10))

    Pixels where the signal is at or below the noise floor are clamped to a
    small epsilon to avoid log(0).

    Args:
        I_target_dB: Target B-scan in dB scale (e.g. 0–130 dB).
        noise_floor_dB: Noise floor level in dB (e.g. 79.0).

    Returns:
        Corrected target in dB scale with noise floor subtracted.
    """
    eps = 1e-12
    I_linear = torch.pow(10.0, I_target_dB / 10.0)
    N_linear = 10.0 ** (noise_floor_dB / 10.0)
    I_corrected_linear = torch.clamp(I_linear - N_linear, min=eps)
    I_corrected_dB = 10.0 * torch.log10(I_corrected_linear)

    # Diagnostics
    tissue_mask = I_target_dB >= noise_floor_dB
    if tissue_mask.any():
        orig_tissue = I_target_dB[tissue_mask]
        corr_tissue = I_corrected_dB[tissue_mask]
        shift = (orig_tissue - corr_tissue)
        print(f"  N(z) subtraction: noise_floor={noise_floor_dB:.1f} dB, "
              f"tissue shift: mean={shift.mean():.3f} dB, max={shift.max():.3f} dB")
    return I_corrected_dB


def compute_loss(I_pred_dB, I_target_dB, noise_mask, loss_scale):
    """Pixel-level masked MSE loss in dB space.

    Args:
        I_pred_dB: Forward model output in dB scale.
        I_target_dB: Target B-scan in dB scale.
        noise_mask: Binary mask (1 = tissue, 0 = below noise floor).  May be None.
        loss_scale: Scalar multiplier applied to the raw MSE.
    """
    eps = 1e-12
    diff = (I_pred_dB - I_target_dB) ** 2

    if noise_mask is not None:
        loss_raw = (diff * noise_mask).sum() / (noise_mask.sum() + eps)
    else:
        loss_raw = diff.mean()

    return loss_scale * loss_raw


def build_noise_floor_mask(I_target_dB, noise_floor_dB, device):
    """Build a binary mask from a dB-scale target image.

    Args:
        I_target_dB: Target B-scan already in dB scale (e.g. 0-130 dB).
        noise_floor_dB: Threshold in dB; pixels below this are noise.
        device: torch device (unused, kept for API compat).
    """
    dB_min = float(I_target_dB.min().item())
    dB_max = float(I_target_dB.max().item())
    eps = 1e-12
    print(f"  Target dB range: [{dB_min:.1f}, {dB_max:.1f}]")
    mask = (I_target_dB >= noise_floor_dB).float()
    tissue_frac = mask.sum() / mask.numel()
    print(f"  Noise floor mask: {tissue_frac:.1%} pixels above {noise_floor_dB:.1f} dB "
          f"(normalized ≈ {(noise_floor_dB - dB_min) / (dB_max - dB_min + eps):.2f})")
    return mask


def _compute_mu_tv(mu, noise_mask=None):
    """Anisotropic Total Variation of mu over tissue pixels.

    TV(mu) = mean(|mu[i+1,j] - mu[i,j]|)  +  mean(|mu[i,j+1] - mu[i,j]|)

    Only adjacent pixel-pairs where *both* pixels are tissue contribute.
    """
    eps = 1e-12
    dz = (mu[1:, :] - mu[:-1, :]).abs()   # vertical (depth) diffs
    dx = (mu[:, 1:] - mu[:, :-1]).abs()   # horizontal (lateral) diffs

    if noise_mask is not None:
        mask_dz = noise_mask[1:, :] * noise_mask[:-1, :]
        mask_dx = noise_mask[:, 1:] * noise_mask[:, :-1]
        tv = (dz * mask_dz).sum() / (mask_dz.sum() + eps) \
           + (dx * mask_dx).sum() / (mask_dx.sum() + eps)
    else:
        tv = dz.mean() + dx.mean()
    return tv


def _save_iter_panel(mu_np, alpha_np, pred_dB_np, tgt_dB_np,
                     delta_mu_np, delta_alpha_np,
                     iteration, mse_val, panel_dir,
                     dB_vmin, dB_vmax, noise_mask_np=None):
    """Save a 2x3 diagnostic figure for a single iteration.

    Layout:
        (0,0) mu map       (0,1) alpha map       (0,2) predicted B-scan
        (1,0) target B-scan (1,1) delta mu        (1,2) delta alpha

    Predicted and target B-scans share the same absolute dB colour scale.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # If mask provided, set noise pixels to dB_vmin so they appear black.
    if noise_mask_np is not None:
        pred_display = np.where(noise_mask_np > 0.5, pred_dB_np, dB_vmin)
        tgt_display = np.where(noise_mask_np > 0.5, tgt_dB_np, dB_vmin)
    else:
        pred_display = pred_dB_np
        tgt_display = tgt_dB_np

    # (0,0) current mu
    im0 = axes[0, 0].imshow(mu_np, cmap='viridis', aspect='auto', origin='upper')
    axes[0, 0].set_title(f'mu (mm$^{{-1}}$)  —  iter {iteration}')
    axes[0, 0].set_ylabel('Depth (px)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # (0,1) current alpha — use log scale since it spans orders of magnitude
    from matplotlib.colors import LogNorm
    alpha_display = np.clip(alpha_np, 1e-12, None)  # avoid log(0)
    im1 = axes[0, 1].imshow(alpha_display, cmap='plasma', aspect='auto', origin='upper',
                             norm=LogNorm(vmin=max(alpha_display.min(), 1e-10),
                                         vmax=alpha_display.max()))
    axes[0, 1].set_title(f'alpha (backscatter)  —  iter {iteration}')
    axes[0, 1].set_ylabel('Depth (px)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (0,2) predicted B-scan — same dB scale as target
    im2 = axes[0, 2].imshow(pred_display, cmap='gray', aspect='auto', origin='upper',
                             vmin=dB_vmin, vmax=dB_vmax)
    axes[0, 2].set_title(f'Predicted B-scan (dB)  —  MSE={mse_val:.4e}')
    axes[0, 2].set_ylabel('Depth (px)')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04, label='dB')

    # (1,0) target B-scan — same dB scale as prediction
    im3 = axes[1, 0].imshow(tgt_display, cmap='gray', aspect='auto', origin='upper',
                             vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 0].set_title('Target B-scan (dB)')
    axes[1, 0].set_xlabel('A-scan')
    axes[1, 0].set_ylabel('Depth (px)')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04, label='dB')

    # (1,1) delta mu from init
    vmax_mu = max(np.max(np.abs(delta_mu_np)), 1e-6)
    im4 = axes[1, 1].imshow(delta_mu_np, cmap='coolwarm', aspect='auto', origin='upper',
                             vmin=-vmax_mu, vmax=vmax_mu)
    axes[1, 1].set_title('Δmu from init (mm$^{-1}$)')
    axes[1, 1].set_xlabel('A-scan')
    axes[1, 1].set_ylabel('Depth (px)')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # (1,2) delta alpha from init
    vmax_a = max(np.max(np.abs(delta_alpha_np)), 1e-6)
    im5 = axes[1, 2].imshow(delta_alpha_np, cmap='coolwarm', aspect='auto', origin='upper',
                             vmin=-vmax_a, vmax=vmax_a)
    axes[1, 2].set_title('Δalpha from init')
    axes[1, 2].set_xlabel('A-scan')
    axes[1, 2].set_ylabel('Depth (px)')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(panel_dir, f'iter_{iteration:05d}.png'), dpi=150)
    plt.close(fig)


def optimize(forward_model, mu, alpha, I_target_dB, opt_config,
             prune_threshold, device, out_dir, noise_floor_mask=None,
             recompute_gain_mse_optimal_each_iter=False,
             mu_snapshot_callback=None):
    """Run gradient-based optimisation of mu and alpha with dB-space MSE loss.

    Both ``mu`` (attenuation) and ``alpha`` (backscattering fraction) are
    per-pixel learnable parameters optimised jointly.  A frozen scalar
    ``gain_dB`` on the forward model absorbs the absolute system gain.

    Args:
        forward_model: ``RasterScanOCTForward`` instance (returns dB).
        mu: ``nn.Parameter`` tensor (depth x a_lines) — attenuation.
        alpha: ``nn.Parameter`` tensor (depth x a_lines) — backscattering.
        I_target_dB: Target B-scan in absolute dB scale (e.g. 0-130).
        opt_config: OmegaConf config with ``optimization`` section.
        prune_threshold: mu values below this are set to mu_min each iter.
        device: torch device.
        out_dir: directory to save initial arrays.
        noise_floor_mask: Optional binary mask (1 = tissue).
        recompute_gain_mse_optimal_each_iter: If True, after each step set
            gain_dB to the closed-form MSE-optimal additive gain for current mu/alpha.
    """
    lr = float(opt_config.optimization.learning_rate)
    num_iterations = int(opt_config.optimization.num_iterations)
    loss_scale = float(opt_config.optimization.loss_scale)
    mu_min_mm_inv = float(getattr(opt_config.optimization, 'mu_min_mm_inv', 0.001))
    mu_max_mm_inv = float(opt_config.optimization.mu_max_mm_inv)
    alpha_min = float(getattr(opt_config.optimization, 'alpha_min', 1e-8))
    alpha_max = float(getattr(opt_config.optimization, 'alpha_max', 1e-2))
    max_delta_mu_mm = float(opt_config.optimization.max_delta_mu_mm)
    panel_save_interval = int(getattr(opt_config.optimization, 'panel_save_interval', 5))
    lr_scheduler_type = str(getattr(opt_config.optimization, 'lr_scheduler', 'none'))
    lr_min_factor = float(getattr(opt_config.optimization, 'lr_min_factor', 0.01))
    mu_tv_weight = float(getattr(opt_config.optimization, 'mu_tv_weight', 0.0))

    eps = 1e-12

    # ------------------------------------------------------------------
    # Zero mu and alpha at noise-floor pixels
    # ------------------------------------------------------------------
    if noise_floor_mask is not None:
        with torch.no_grad():
            mu.data *= noise_floor_mask
            alpha.data *= noise_floor_mask
        print(f"  Applied noise floor mask: {int(noise_floor_mask.sum().item())} active scatterers")

    # Initialise gain_dB: use fixed value from config, or auto-init from data.
    gain_dB_cfg = getattr(opt_config.optimization, 'gain_dB', 'auto')
    if isinstance(gain_dB_cfg, (int, float)):
        with torch.no_grad():
            forward_model.gain_dB.fill_(float(gain_dB_cfg))
        print(f"  gain_dB = {forward_model.gain_dB.item():.1f} dB (fixed from config)")
    else:
        with torch.no_grad():
            forward_model.gain_dB.fill_(0.0)  # reset before auto-init
            I_pred_init = forward_model(mu, alpha, noise_mask=noise_floor_mask)
            if noise_floor_mask is not None:
                tissue = noise_floor_mask > 0.5
                pred_median = I_pred_init[tissue].median()
                tgt_median = I_target_dB[tissue].median()
            else:
                pred_median = I_pred_init.median()
                tgt_median = I_target_dB.median()
            offset = tgt_median - pred_median
            forward_model.gain_dB.fill_(offset.item())
        if recompute_gain_mse_optimal_each_iter:
            print(f"  gain_dB = {forward_model.gain_dB.item():.1f} dB (auto-init, will recompute each iter)")
        else:
            print(f"  gain_dB = {forward_model.gain_dB.item():.1f} dB (auto-init, frozen)")

    # ------------------------------------------------------------------
    # Set up optimiser — mu and alpha share the same learning rate
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam([mu, alpha], lr=lr)
    print(f"  Optimizer: Adam  lr={lr}  params=[mu, alpha]")
    print(f"  Bounds: mu=[{mu_min_mm_inv}, {mu_max_mm_inv}] mm^-1  "
          f"alpha=[{alpha_min:.1e}, {alpha_max:.1e}]")
    if mu_tv_weight > 0:
        print(f"  TV regularization: mu_tv_weight={mu_tv_weight:.1e}")

    # ------------------------------------------------------------------
    # Learning rate scheduler
    # ------------------------------------------------------------------
    if lr_scheduler_type == 'cosine':
        eta_min = lr * lr_min_factor
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iterations, eta_min=eta_min)
        print(f"  LR schedule: cosine  T_max={num_iterations}  lr_min={eta_min:.2e}")
    else:
        scheduler = None
        print(f"  LR schedule: none (constant)")

    mu_init = mu.detach().clone()
    alpha_init = alpha.detach().clone()
    np.save(os.path.join(out_dir, 'mu_init_mm.npy'), mu_init.cpu().numpy())
    np.save(os.path.join(out_dir, 'alpha_init.npy'), alpha_init.cpu().numpy())

    metrics = {'iters': [], 'mse': [],
               'mu_mean_mm': [], 'mu_std_mm': [],
               'alpha_mean': [], 'alpha_std': [],
               'gain_dB': [], 'lr': [], 'time': []}
    seq_steps = []
    seq_imgs = []

    # Early stopping: track best MSE and snapshot
    best_mse = float('inf')
    best_iter = 0
    best_mu = mu.detach().clone()
    best_alpha = alpha.detach().clone()

    # ------------------------------------------------------------------
    # Pre-compute display constants for panels
    # ------------------------------------------------------------------
    panel_dir = os.path.join(out_dir, 'iter_panels')
    os.makedirs(panel_dir, exist_ok=True)

    # Shared dB display range for both predicted & target B-scan panels.
    if noise_floor_mask is not None:
        tgt_tissue_vals = I_target_dB[noise_floor_mask > 0.5]
        dB_vmin = float(tgt_tissue_vals.min().item())
        dB_vmax = float(tgt_tissue_vals.max().item())
    else:
        dB_vmin = float(I_target_dB.min().item())
        dB_vmax = float(I_target_dB.max().item())
    print(f"  Display dB range: [{dB_vmin:.1f}, {dB_vmax:.1f}] dB")

    tgt_dB_np = I_target_dB.cpu().numpy()
    noise_mask_np = noise_floor_mask.cpu().numpy() if noise_floor_mask is not None else None
    mu_init_np = mu_init.cpu().numpy()
    alpha_init_np = alpha_init.cpu().numpy()

    # ------------------------------------------------------------------
    # Warmup + optional CUDA-graph capture (disabled when gain is updated each iter)
    # ------------------------------------------------------------------
    use_cuda_graph = (device.type == 'cuda') and not recompute_gain_mse_optimal_each_iter

    def _warmup_step():
        """One warmup iteration with gradient sanitisation and projection."""
        optimizer.zero_grad(set_to_none=False)
        I_pred_w = forward_model(mu, alpha, noise_mask=noise_floor_mask)
        loss_w = compute_loss(I_pred_w, I_target_dB,
                              noise_floor_mask, loss_scale)
        if mu_tv_weight > 0:
            loss_w = loss_w + mu_tv_weight * _compute_mu_tv(mu, noise_floor_mask)
        loss_w.backward()
        with torch.no_grad():
            for p in [mu, alpha]:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    if lr > 0 and max_delta_mu_mm < 1e6:
                        p.grad.clamp_(min=-max_delta_mu_mm / lr,
                                      max=max_delta_mu_mm / lr)
        optimizer.step()
        with torch.no_grad():
            mu.data.clamp_(min=mu_min_mm_inv, max=mu_max_mm_inv)
            alpha.data.clamp_(min=alpha_min, max=alpha_max)
            if noise_floor_mask is not None:
                mu.data.mul_(noise_floor_mask)
                alpha.data.mul_(noise_floor_mask)

    if use_cuda_graph:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                _warmup_step()
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            optimizer.zero_grad(set_to_none=False)
            I_pred_static = forward_model(mu, alpha, noise_mask=noise_floor_mask)
            loss_static = compute_loss(I_pred_static, I_target_dB,
                                       noise_floor_mask, loss_scale)
            if mu_tv_weight > 0:
                loss_static = loss_static + mu_tv_weight * _compute_mu_tv(mu, noise_floor_mask)
            loss_static.backward()
    else:
        for _ in range(3):
            _warmup_step()

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------
    print(f"  Starting {num_iterations} iterations (lr={lr}, "
          f"mu_min={mu_min_mm_inv}, loss=pixel-MSE-dB)")
    total_start = time.time()

    for it in range(num_iterations + 1):
        iter_start = time.time()

        if use_cuda_graph:
            graph.replay()
        else:
            optimizer.zero_grad(set_to_none=False)
            I_pred_eager = forward_model(mu, alpha, noise_mask=noise_floor_mask)
            loss_eager = compute_loss(I_pred_eager, I_target_dB,
                                      noise_floor_mask, loss_scale)
            if mu_tv_weight > 0:
                loss_eager = loss_eager + mu_tv_weight * _compute_mu_tv(mu, noise_floor_mask)
            loss_eager.backward()

        # --- gradient post-processing on mu and alpha (sanitise, clamp) ---
        for p in [mu, alpha]:
            if p.grad is not None:
                with torch.no_grad():
                    grad = p.grad
                    torch.nan_to_num_(grad, nan=0.0, posinf=0.0, neginf=0.0)

                    if lr > 0 and max_delta_mu_mm < 1e6:
                        grad_cap = max_delta_mu_mm / lr
                        grad.clamp_(min=-grad_cap, max=grad_cap)

        # --- Adam step (updates mu and alpha) ---
        optimizer.step()

        # --- learning rate schedule ---
        if scheduler is not None:
            scheduler.step()

        # --- project mu and alpha back into feasible set ---
        with torch.no_grad():
            mu.data.clamp_(min=mu_min_mm_inv, max=mu_max_mm_inv)
            alpha.data.clamp_(min=alpha_min, max=alpha_max)

            prune_mask = mu.data < prune_threshold
            if torch.any(prune_mask):
                mu.data[prune_mask] = mu_min_mm_inv

            # Re-zero masked pixels so clamp doesn't lift them to mu_min
            if noise_floor_mask is not None:
                mu.data.mul_(noise_floor_mask)
                alpha.data.mul_(noise_floor_mask)

            # Per-iter MSE-optimal gain (closed-form)
            if recompute_gain_mse_optimal_each_iter:
                forward_model.gain_dB.fill_(0.0)
                pred_no_gain = forward_model(mu, alpha, noise_mask=noise_floor_mask)
                if noise_floor_mask is not None:
                    g_star = ((I_target_dB - pred_no_gain) * noise_floor_mask).sum() / (noise_floor_mask.sum() + eps)
                else:
                    g_star = (I_target_dB - pred_no_gain).mean()
                forward_model.gain_dB.fill_(g_star.item())

        iter_time = time.time() - iter_start

        # --- diagnostics ---
        with torch.no_grad():
            mu_mean = float(mu.mean().item())
            mu_std = float(mu.std().item())
            alpha_mean = float(alpha.mean().item())
            alpha_std = float(alpha.std().item())
            gain_dB_val = float(forward_model.gain_dB.item())

            pred_dB_diag = forward_model(mu, alpha, noise_mask=noise_floor_mask)

            diff_diag = (pred_dB_diag - I_target_dB) ** 2
            if noise_floor_mask is not None:
                mse_val = float((diff_diag * noise_floor_mask).sum().item()
                                / (noise_floor_mask.sum().item() + eps))
            else:
                mse_val = float(diff_diag.mean().item())

        # Track best model
        if mse_val < best_mse:
            best_mse = mse_val
            best_iter = it
            best_mu.copy_(mu.detach())
            best_alpha.copy_(alpha.detach())

        metrics['iters'].append(it)
        metrics['mse'].append(mse_val)
        metrics['mu_mean_mm'].append(mu_mean)
        metrics['mu_std_mm'].append(mu_std)
        metrics['alpha_mean'].append(alpha_mean)
        metrics['alpha_std'].append(alpha_std)
        metrics['gain_dB'].append(gain_dB_val)
        metrics['lr'].append(float(optimizer.param_groups[0]['lr']))
        metrics['time'].append(iter_time)

        if mu_snapshot_callback is not None:
            mu_snapshot_callback(it, mu)

        if it % max(1, num_iterations // 10) == 0 or it == num_iterations:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"    Iter {it:5d} | MSE={mse_val:.6e} | "
                  f"mu={mu_mean:.4f}±{mu_std:.4f} | "
                  f"alpha={alpha_mean:.2e}±{alpha_std:.2e} | "
                  f"lr={cur_lr:.2e} | time={iter_time:.3f}s")

        # --- 2x3 diagnostic panel ---
        if it % panel_save_interval == 0 or it == num_iterations:
            with torch.no_grad():
                mu_np = mu.detach().cpu().numpy()
                alpha_np_snap = alpha.detach().cpu().numpy()
                pred_dB_np = pred_dB_diag.cpu().numpy()
                delta_mu_np = mu_np - mu_init_np
                delta_alpha_np = alpha_np_snap - alpha_init_np
            _save_iter_panel(mu_np, alpha_np_snap, pred_dB_np, tgt_dB_np,
                             delta_mu_np, delta_alpha_np,
                             it, mse_val, panel_dir,
                             dB_vmin=dB_vmin, dB_vmax=dB_vmax,
                             noise_mask_np=noise_mask_np)

        snapshot_interval = max(1, num_iterations // 10)
        if it % snapshot_interval == 0:
            with torch.no_grad():
                vis_min = pred_dB_diag.min()
                vis_range = pred_dB_diag.max() - vis_min + eps
                pred_vis = ((pred_dB_diag - vis_min) / vis_range).cpu().numpy()
                seq_steps.append(it)
                seq_imgs.append(pred_vis)

    total_time = time.time() - total_start

    # Restore best model
    with torch.no_grad():
        mu.data.copy_(best_mu)
        alpha.data.copy_(best_alpha)
    print(f"  Done: {total_time:.1f}s total, {total_time / (num_iterations + 1):.3f}s/iter")
    print(f"  Best MSE={best_mse:.6e} at iter {best_iter} (restored)")
    print(f"  Frozen gain_dB = {float(forward_model.gain_dB.item()):.2f} dB")

    return mu, alpha, metrics, seq_steps, seq_imgs, mu_init, alpha_init
