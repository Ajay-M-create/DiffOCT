#!/usr/bin/env python3
"""
20_render_IVOCT.py — IV-OCT Property Extraction and Rendering
==============================================================
Selects 12 frames from an IV-OCT pullback, runs heavy overfitting optimization
(no regularization, many iterations) to recover per-pixel attenuation maps (mu),
and renders synthetic OCT images that should be virtually identical to the
noise-floor-corrected ground truth.

Output:
  - Per-frame panels: GT IV-OCT | OAC Map | Rendered IV-OCT | Diff
  - Composite 3x4 figure with all 12 frames (48 sub-images total)
  - outputs/20_render_ivoct/{patient_id}/

Environment variables:
    PATIENT_ID       Patient prefix (default: "02008Pre")
    FRAME_INDICES    Comma-separated frame indices (default: 12 evenly spaced)
    ARRAY_TASK_ID    Process only the Nth frame from the list (for SLURM array)
    NUM_ITERS        Optimization iterations (default: 50000)
    DB_SCALE         dB conversion scale (default: 130.0)
    NOISE_FLOOR_DB   Noise floor threshold in dB (default: 15.0)
    SUMMARY          "1" to skip optimization and build paper figures (clinical_src.pdf, supp_fig_alpha_refinement.pdf)
"""

import os
import sys
import time
import glob
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import map_coordinates

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from src.forward import RasterScanOCTForward
from src.inverse import optimize, build_noise_floor_mask, subtract_noise_floor_linear

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
PATIENT_DIR = os.environ.get(
    "PATIENT_DATA_DIR",
    "/home/gridsan/amanicka/erelab_shared/Ajay/DiffOCTDatasets/IVLPatientsAndSegmentations/PrePatients")

# IV-OCT geometry
NUM_ALINES = 500
DEPTH_PIXELS = 250
IMAGE_SIZE = 500
CENTER = IMAGE_SIZE // 2


# ===================================================================
# Data loading
# ===================================================================

def find_patient_file(patient_id):
    """Find patient NIfTI file."""
    pattern = os.path.join(PATIENT_DIR, f"{patient_id}*.nii.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No patient file matching {pattern}")
    print(f"Patient file: {files[0]}")
    return files[0]


def load_nifti(path):
    import nibabel as nib
    return nib.load(path).get_fdata()


# ===================================================================
# Coordinate conversion
# ===================================================================

def cartesian_to_polar(image_2d, center=(CENTER, CENTER),
                       num_alines=NUM_ALINES, depth_pixels=DEPTH_PIXELS):
    """Convert 500x500 Cartesian IV-OCT image to polar (depth x A-lines)."""
    cy, cx = center
    angles = np.linspace(0, 2 * np.pi, num_alines, endpoint=False)
    radii = np.arange(depth_pixels, dtype=np.float64)
    r_grid, a_grid = np.meshgrid(radii, angles, indexing='ij')
    y_coords = cy + r_grid * np.sin(a_grid)
    x_coords = cx + r_grid * np.cos(a_grid)
    coords = np.array([y_coords, x_coords])
    return map_coordinates(image_2d, coords, order=1, mode='constant', cval=0.0)


def polar_to_cartesian(polar_img, center=(CENTER, CENTER),
                       image_size=IMAGE_SIZE, num_alines=NUM_ALINES,
                       depth_pixels=DEPTH_PIXELS, fill_value=0.0):
    """Convert polar (depth x A-lines) back to 500x500 Cartesian."""
    cy, cx = center
    y_coords, x_coords = np.mgrid[0:image_size, 0:image_size]
    dy = y_coords - cy
    dx = x_coords - cx
    radii = np.sqrt(dy**2 + dx**2)
    angles = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    aline_idx = angles / (2 * np.pi) * num_alines
    coords = np.array([radii, aline_idx])
    cart = map_coordinates(polar_img, coords, order=1, mode='constant', cval=fill_value)
    cart[radii >= depth_pixels] = fill_value
    return cart


# ===================================================================
# Config builders
# ===================================================================

def build_ivoct_config(noise_floor_dB=15.0):
    """IV-OCT system config (ILUMIEN OPTIS / C7-XR Dragonfly)."""
    return OmegaConf.create({
        'forward_model_system': {
            'central_wavelength_nm': 1305,
            'bandwidth_nm': 110,
            'numerical_aperture': 0.017,
            'depth_samples': DEPTH_PIXELS,
            'a_lines': NUM_ALINES,
            'b_lines': 1,
            'sensitivity_falloff': False,
            'confocal_correction': True,
            'focal_plane_position_pixels': int(os.environ.get('CONFOCAL_ZFP_PX', '62')),
            'rayleigh_range_pixels': int(os.environ.get('CONFOCAL_ZR_PX', '62')),
            'noise_floor_dB': noise_floor_dB,
            'lateral_pixel_size_mm': 0.01932,
            'axial_pixel_size_mm': 0.01932,
            'forward_model_type': 'ss',
        },
        'loss': 'mse',
    })


def build_opt_config(num_iterations=50000, alpha_min=1.0, alpha_max=1.0):
    """Optimization config for heavy overfitting (no regularization)."""
    return OmegaConf.create({
        'optimization': {
            'learning_rate': 1e-3,
            'num_iterations': num_iterations,
            'panel_save_interval': max(1, num_iterations // 5),
            'loss_function_type': 'mse',
            'loss_scale': 1.0e+3,
            'lr_scheduler': 'cosine',
            'lr_min_factor': 0.01,
            'mu_min_mm_inv': 0.01,
            'mu_max_mm_inv': 20.0,
            'alpha_min': alpha_min,
            'alpha_max': alpha_max,
            'max_delta_mu_mm': 1.0,
            'mu_tv_weight': 0.0,        # NO TV — allow full overfitting
            'mu_l1_weight': 0.0,         # NO L1 — allow full overfitting
            'gain_dB': 'auto',
            'gradient_scaling_enabled': False,
            'gradient_scaling_factor': 2.0,
            'gradient_scaling_cap': 50.0,
            'mu_stats_threshold_mm_inv': 0,
        }
    })


# ===================================================================
# Single frame processing
# ===================================================================

def process_frame(frame_cart, frame_idx, config, opt_config_s1, opt_config_s2,
                  noise_floor_dB, db_scale, pixel_max, device, out_dir):
    """Run two-stage optimization on one frame and return results dict.

    Stage 1: Optimize mu with alpha frozen at 1.0 — captures attenuation structure.
    Stage 2: Freeze mu, optimize alpha in [1e-10, 1e-2] — captures per-pixel
             backscattering to close the brightness gap.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert Cartesian to polar
    polar_img = cartesian_to_polar(frame_cart)

    # Scale to dB: pixel values [0, max] -> [0, db_scale]
    if pixel_max > 0:
        polar_img_dB = polar_img * (db_scale / pixel_max)
    else:
        polar_img_dB = polar_img * db_scale

    print(f"\n{'='*60}")
    print(f"Frame {frame_idx}")
    print(f"{'='*60}")
    print(f"  Polar shape: {polar_img_dB.shape}")
    print(f"  dB range: [{polar_img_dB.min():.1f}, {polar_img_dB.max():.1f}]")

    # Subtract noise floor in linear space (Cannon Eq. 2.13)
    I_target_dB_orig = torch.tensor(polar_img_dB, dtype=torch.float32, device=device)
    I_target_dB = subtract_noise_floor_linear(I_target_dB_orig, noise_floor_dB)
    print(f"  Post-subtraction dB range: [{I_target_dB.min().item():.1f}, "
          f"{I_target_dB.max().item():.1f}]")

    # Build noise floor mask on subtracted image
    noise_floor_mask = build_noise_floor_mask(I_target_dB, noise_floor_dB, device)

    # ---- Stage 1: Optimize mu (alpha frozen at 1.0) ----
    stage1_dir = os.path.join(out_dir, 'stage1_mu')
    os.makedirs(stage1_dir, exist_ok=True)

    mu = torch.nn.Parameter(
        torch.empty(DEPTH_PIXELS, NUM_ALINES, device=device).uniform_(0.0, 10.0))
    alpha = torch.nn.Parameter(
        torch.ones(DEPTH_PIXELS, NUM_ALINES, device=device))

    forward_model = RasterScanOCTForward(config, device=str(device))

    n_iters_s1 = int(opt_config_s1.optimization.num_iterations)
    print(f"\n  === Stage 1: Optimize mu (alpha=1.0) — {n_iters_s1} iters ===")
    t0 = time.time()
    mu_opt, _, metrics_s1, _, _, _, _ = optimize(
        forward_model, mu, alpha, I_target_dB, opt_config_s1,
        prune_threshold=0.05, device=device, out_dir=stage1_dir,
        noise_floor_mask=noise_floor_mask,
        recompute_gain_mse_optimal_each_iter=True)
    elapsed_s1 = time.time() - t0
    print(f"  Stage 1 done in {elapsed_s1:.1f}s  |  Best MSE: {min(metrics_s1['mse']):.6e}")

    # Get stage 1 rendered OCT (before alpha optimization)
    with torch.no_grad():
        pred_s1_polar = forward_model(mu_opt, alpha, noise_mask=noise_floor_mask)
    pred_s1_polar_np = pred_s1_polar.cpu().numpy()

    # ---- Stage 2: Freeze mu, optimize alpha ----
    stage2_dir = os.path.join(out_dir, 'stage2_alpha')
    os.makedirs(stage2_dir, exist_ok=True)

    mu_frozen = torch.nn.Parameter(mu_opt.detach().clone())
    mu_frozen.requires_grad_(False)

    alpha_s2 = torch.nn.Parameter(
        torch.full((DEPTH_PIXELS, NUM_ALINES), 1e-3, device=device))

    forward_model_s2 = RasterScanOCTForward(config, device=str(device))

    n_iters_s2 = int(opt_config_s2.optimization.num_iterations)
    print(f"\n  === Stage 2: Optimize alpha (mu frozen) — {n_iters_s2} iters ===")
    t0 = time.time()
    _, alpha_opt, metrics_s2, _, _, _, _ = optimize(
        forward_model_s2, mu_frozen, alpha_s2, I_target_dB, opt_config_s2,
        prune_threshold=0.05, device=device, out_dir=stage2_dir,
        noise_floor_mask=noise_floor_mask,
        recompute_gain_mse_optimal_each_iter=True)
    elapsed_s2 = time.time() - t0
    print(f"  Stage 2 done in {elapsed_s2:.1f}s  |  Best MSE: {min(metrics_s2['mse']):.6e}")

    # Get stage 2 rendered OCT
    with torch.no_grad():
        pred_s2_polar = forward_model_s2(mu_frozen, alpha_opt, noise_mask=noise_floor_mask)

    # Convert to numpy
    target_polar_np = I_target_dB.cpu().numpy()
    pred_s2_polar_np = pred_s2_polar.cpu().numpy()
    mu_np = mu_opt.detach().cpu().numpy()
    alpha_np = alpha_opt.detach().cpu().numpy()
    nfm_np = noise_floor_mask.cpu().numpy()

    # Compute PSNR on tissue pixels for both stages
    tissue = nfm_np > 0.5
    signal_range = float(target_polar_np[tissue].max() - target_polar_np[tissue].min()) if tissue.any() else 1.0

    def _psnr_stats(pred, target, tissue_mask):
        if not tissue_mask.any():
            return 0.0, float('inf'), 0.0
        d = pred[tissue_mask] - target[tissue_mask]
        mse = float(np.mean(d**2))
        psnr = 10 * np.log10(signal_range**2 / (mse + 1e-12))
        mae = float(np.mean(np.abs(d)))
        return mse, psnr, mae

    mse_s1, psnr_s1, mae_s1 = _psnr_stats(pred_s1_polar_np, target_polar_np, tissue)
    mse_s2, psnr_s2, mae_s2 = _psnr_stats(pred_s2_polar_np, target_polar_np, tissue)
    print(f"  Stage 1 — PSNR: {psnr_s1:.1f} dB  |  MSE: {mse_s1:.4f}  |  MAE: {mae_s1:.4f} dB")
    print(f"  Stage 2 — PSNR: {psnr_s2:.1f} dB  |  MSE: {mse_s2:.4f}  |  MAE: {mae_s2:.4f} dB")
    print(f"  Total time: {elapsed_s1 + elapsed_s2:.1f}s")

    # Convert to Cartesian for display
    gt_cart = polar_to_cartesian(target_polar_np, fill_value=0.0)
    rendered_s1_cart = polar_to_cartesian(pred_s1_polar_np, fill_value=0.0)
    rendered_s2_cart = polar_to_cartesian(pred_s2_polar_np, fill_value=0.0)
    mu_cart = polar_to_cartesian(mu_np, fill_value=0.0)
    alpha_cart = polar_to_cartesian(alpha_np, fill_value=0.0)

    # Save arrays
    np.save(os.path.join(out_dir, 'mu_final_mm.npy'), mu_np)
    np.save(os.path.join(out_dir, 'alpha_final.npy'), alpha_np)
    np.save(os.path.join(out_dir, 'gt_cart.npy'), gt_cart)
    np.save(os.path.join(out_dir, 'rendered_s1_cart.npy'), rendered_s1_cart)
    np.save(os.path.join(out_dir, 'rendered_s2_cart.npy'), rendered_s2_cart)
    np.save(os.path.join(out_dir, 'mu_cart.npy'), mu_cart)
    np.save(os.path.join(out_dir, 'alpha_cart.npy'), alpha_cart)
    np.savez(os.path.join(out_dir, 'metrics.npz'),
             **{k: np.array(v) for k, v in metrics_s1.items()})
    with open(os.path.join(out_dir, 'psnr.txt'), 'w') as f:
        f.write(f"Stage1_PSNR = {psnr_s1:.2f} dB\nStage1_MSE  = {mse_s1:.6e}\n"
                f"Stage2_PSNR = {psnr_s2:.2f} dB\nStage2_MSE  = {mse_s2:.6e}\n")

    return {
        'gt_cart': gt_cart,
        'rendered_s1_cart': rendered_s1_cart,
        'rendered_s2_cart': rendered_s2_cart,
        'mu_cart': mu_cart,
        'alpha_cart': alpha_cart,
        'psnr_s1': psnr_s1,
        'psnr_s2': psnr_s2,
        'frame_idx': frame_idx,
    }


# ===================================================================
# Visualization
# ===================================================================

def _get_display_params(gt_cart):
    """Compute display parameters from a GT Cartesian image."""
    positive = gt_cart > 1.0  # above noise floor
    if positive.any():
        dB_vmin = float(np.percentile(gt_cart[positive], 1))
        dB_vmax = float(np.percentile(gt_cart[positive], 99))
    else:
        dB_vmin, dB_vmax = 0.0, 1.0
    return dB_vmin, dB_vmax, positive


def save_frame_panel(result, out_path):
    """Save a single frame's 2x4 panel:
    Row 1 (mu stage):    GT | OAC (mu) | Rendered S1 | Diff S1
    Row 2 (alpha stage): GT | Alpha    | Rendered S2 | Diff S2
    """
    from matplotlib.colors import LogNorm

    gt = result['gt_cart']
    rendered_s1 = result['rendered_s1_cart']
    rendered_s2 = result['rendered_s2_cart']
    mu = result['mu_cart']
    alpha = result['alpha_cart']
    fi = result['frame_idx']

    dB_vmin, dB_vmax, tissue = _get_display_params(gt)

    mu_display = np.where(tissue, mu, 0.0)
    mu_vmax = float(np.percentile(mu_display[tissue], 99)) if tissue.any() else 1.0

    alpha_tissue = alpha[tissue & (alpha > 1e-10)]
    alpha_vmin = float(np.percentile(alpha_tissue, 1)) if len(alpha_tissue) > 0 else 1e-10
    alpha_vmax = float(np.percentile(alpha_tissue, 99)) if len(alpha_tissue) > 0 else 1e-2

    diff_s1 = np.where(tissue, rendered_s1 - gt, 0.0)
    diff_s2 = np.where(tissue, rendered_s2 - gt, 0.0)
    vabs = max(float(np.abs(diff_s1[tissue]).max()),
               float(np.abs(diff_s2[tissue]).max()), 0.1) if tissue.any() else 1.0

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # ---- Row 0: mu stage ----
    im00 = axes[0, 0].imshow(np.clip(gt, 0, None), cmap='gray', vmin=dB_vmin, vmax=dB_vmax)
    axes[0, 0].set_title(f'GT IV-OCT (Frame {fi})', fontsize=11)
    axes[0, 0].axis('off')
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.02, label='dB')

    im01 = axes[0, 1].imshow(mu_display, cmap='viridis', vmin=0, vmax=mu_vmax)
    axes[0, 1].set_title('OAC Map (mm$^{-1}$)', fontsize=11)
    axes[0, 1].axis('off')
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.02, label='mm$^{-1}$')

    im02 = axes[0, 2].imshow(np.clip(rendered_s1, 0, None), cmap='gray', vmin=dB_vmin, vmax=dB_vmax)
    axes[0, 2].set_title(f'Rendered (mu only, {result["psnr_s1"]:.1f} dB)', fontsize=11)
    axes[0, 2].axis('off')
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.02, label='dB')

    im03 = axes[0, 3].imshow(diff_s1, cmap='coolwarm', vmin=-vabs, vmax=vabs)
    axes[0, 3].set_title('Diff (mu only $-$ GT)', fontsize=11)
    axes[0, 3].axis('off')
    plt.colorbar(im03, ax=axes[0, 3], fraction=0.046, pad=0.02, label='dB')

    # ---- Row 1: alpha stage ----
    im10 = axes[1, 0].imshow(np.clip(gt, 0, None), cmap='gray', vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 0].set_title(f'GT IV-OCT (Frame {fi})', fontsize=11)
    axes[1, 0].axis('off')
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.02, label='dB')

    alpha_log_display = np.where(tissue, alpha, np.nan)
    im11 = axes[1, 1].imshow(alpha_log_display, cmap='plasma',
                             norm=LogNorm(vmin=max(alpha_vmin, 1e-10), vmax=alpha_vmax))
    axes[1, 1].set_title('Alpha (backscatter)', fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.02)

    im12 = axes[1, 2].imshow(np.clip(rendered_s2, 0, None), cmap='gray', vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 2].set_title(f'Rendered (mu+alpha, {result["psnr_s2"]:.1f} dB)', fontsize=11)
    axes[1, 2].axis('off')
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.02, label='dB')

    im13 = axes[1, 3].imshow(diff_s2, cmap='coolwarm', vmin=-vabs, vmax=vabs)
    axes[1, 3].set_title('Diff (mu+alpha $-$ GT)', fontsize=11)
    axes[1, 3].axis('off')
    plt.colorbar(im13, ax=axes[1, 3], fraction=0.046, pad=0.02, label='dB')

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved panel: {out_path}")


def _load_all_results(out_dir, frame_indices):
    """Load saved per-frame results and compute global display scales."""
    from matplotlib.colors import LogNorm

    results = []
    for fi in frame_indices:
        frame_dir = os.path.join(out_dir, f'frame_{fi:03d}')
        gt = np.load(os.path.join(frame_dir, 'gt_cart.npy'))
        rs1 = np.load(os.path.join(frame_dir, 'rendered_s1_cart.npy'))
        rs2 = np.load(os.path.join(frame_dir, 'rendered_s2_cart.npy'))
        mu = np.load(os.path.join(frame_dir, 'mu_cart.npy'))
        alpha = np.load(os.path.join(frame_dir, 'alpha_cart.npy'))
        with open(os.path.join(frame_dir, 'psnr.txt')) as f:
            lines = f.readlines()
            psnr_s1 = float(lines[0].split('=')[1].strip().split()[0])
            psnr_s2 = float(lines[2].split('=')[1].strip().split()[0])
        results.append({'gt': gt, 'rs1': rs1, 'rs2': rs2, 'mu': mu,
                        'alpha': alpha, 'psnr_s1': psnr_s1,
                        'psnr_s2': psnr_s2, 'fi': int(fi)})

    all_mu_tissue, all_alpha_tissue = [], []
    for r in results:
        tissue = r['gt'] > 1.0
        if tissue.any():
            all_mu_tissue.append(r['mu'][tissue])
            a = r['alpha'][tissue]
            all_alpha_tissue.append(a[a > 1e-10])

    global_mu_vmax = float(np.percentile(np.concatenate(all_mu_tissue), 99)) if all_mu_tissue else 1.0
    if all_alpha_tissue:
        cat = np.concatenate(all_alpha_tissue)
        global_alpha_vmin = float(np.percentile(cat, 1))
        global_alpha_vmax = float(np.percentile(cat, 99))
    else:
        global_alpha_vmin, global_alpha_vmax = 1e-10, 1e-2

    return results, global_mu_vmax, global_alpha_vmin, global_alpha_vmax


def _dice_noise_floor(rendered, target, noise_floor_dB=15.0):
    """Dice overlap of two log-scaled (dB) images binarized at the noise floor.

    Both `rendered` and `target` are Cartesian dB images; they are binarized as
    ``img > noise_floor_dB`` and the Dice coefficient is computed over the full
    frame:

        Dice = 2 |A & B| / (|A| + |B|)

    where A = (target > noise_floor_dB) and B = (rendered > noise_floor_dB).
    Returns 0.0 if both masks are empty.
    """
    A = target > noise_floor_dB
    B = rendered > noise_floor_dB
    denom = int(A.sum()) + int(B.sum())
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(A, B).sum() / denom)


def save_paper_figures(out_dir, frame_indices, patient_id):
    """Generate paper-quality PDFs matching the style of clinical_src panels (a)/(b).

    Layout: 6 rows x 2 columns. Each cell = [IV-OCT | OAC | Rendered | Residual].
    Column headers on first row only, row labels ("Frame N") on left.

    clinical_src.pdf:             mu-only (GT | OAC | Rendered | Residual)
    supp_fig_alpha_refinement.pdf: two sub-rows per frame (mu stage + alpha stage)
    """
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import gc

    n_frames = len(frame_indices)
    if n_frames < 1:
        raise ValueError("Need at least one frame")

    paper_dir = os.path.join(PROJECT_ROOT, 'paper', 'Figs')
    os.makedirs(paper_dir, exist_ok=True)

    # Center 400x400 crop of the 500x500 Cartesian frames — removes the
    # mostly-empty corners so tissue detail is larger in the figure.
    _crop_lo = (IMAGE_SIZE - 400) // 2
    _crop_hi = _crop_lo + 400
    CROP = np.s_[_crop_lo:_crop_hi, _crop_lo:_crop_hi]

    def _load_cart(fd, name):
        return np.load(os.path.join(fd, name))[CROP]

    # Pre-scan for global scales (load one frame at a time to save memory).
    # Dice is computed on the FULL uncropped frame, as in the paper's original
    # metric; everything else is evaluated over the same cropped window used for display.
    all_mu_tissue, all_alpha_tissue = [], []
    psnr_s1_list, psnr_s2_list = [], []
    dice_s1_list, dice_s2_list = [], []
    for fi in frame_indices:
        fd = os.path.join(out_dir, f'frame_{fi:03d}')
        mu = _load_cart(fd, 'mu_cart.npy')
        gt = _load_cart(fd, 'gt_cart.npy')
        alpha = _load_cart(fd, 'alpha_cart.npy')
        rendered_s1_full = np.load(os.path.join(fd, 'rendered_s1_cart.npy'))
        rendered_s2_full = np.load(os.path.join(fd, 'rendered_s2_cart.npy'))
        gt_full = np.load(os.path.join(fd, 'gt_cart.npy'))
        tissue = gt > 1.0
        if tissue.any():
            all_mu_tissue.append(mu[tissue])
            a = alpha[tissue]
            all_alpha_tissue.append(a[a > 1e-10])
        with open(os.path.join(fd, 'psnr.txt')) as f:
            lines = f.readlines()
            psnr_s1_list.append(float(lines[0].split('=')[1].strip().split()[0]))
            psnr_s2_list.append(float(lines[2].split('=')[1].strip().split()[0]))
        dice_s1_list.append(_dice_noise_floor(rendered_s1_full, gt_full, noise_floor_dB=15.0))
        dice_s2_list.append(_dice_noise_floor(rendered_s2_full, gt_full, noise_floor_dB=15.0))
        del mu, gt, alpha, rendered_s1_full, rendered_s2_full, gt_full

    global_mu_vmax = float(np.percentile(np.concatenate(all_mu_tissue), 99))
    cat_alpha = np.concatenate(all_alpha_tissue)
    global_alpha_vmin = float(np.percentile(cat_alpha, 1))
    global_alpha_vmax = float(np.percentile(cat_alpha, 99))
    del all_mu_tissue, all_alpha_tissue, cat_alpha
    gc.collect()

    # Style constants matching clinical_src panels (a)/(b)
    COL_HEADER_SIZE = 18
    ROW_LABEL_SIZE = 16
    CBAR_TICK_SIZE = 14
    CBAR_LABEL_SIZE = 16

    n_panel_cols = 2                     # two halves side-by-side
    n_panel_rows = (n_frames + n_panel_cols - 1) // n_panel_cols  # ceil
    n_sub_cols = 4                       # IV-OCT | OAC | Rendered | Residual

    def _add_cbar(fig, ax, im, label=''):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.02)
        cb = fig.colorbar(im, cax=cax)
        if label:
            cb.set_label(label, size=CBAR_LABEL_SIZE)
        cax.tick_params(labelsize=CBAR_TICK_SIZE)

    # =====================================================================
    # clinical_src.pdf — mu only, 6 rows x 2 panel-cols, tightly packed
    # =====================================================================
    print("\nGenerating clinical_src.pdf ...")
    total_img_cols = n_panel_cols * n_sub_cols + 1  # 9 (with gap column)
    width_ratios = [1, 1, 1, 1, 0.15, 1, 1, 1, 1]  # gap between panel cols
    # Height scales with n_panel_rows so each row has roughly the same pixel height
    clinical_figsize = (21, max(5, int(2.5 * n_panel_rows)))
    fig, axes = plt.subplots(n_panel_rows, total_img_cols,
                             figsize=clinical_figsize,
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.005,
                                          'width_ratios': width_ratios})
    # Hide the gap column
    for r in range(n_panel_rows):
        axes[r, 4].set_visible(False)

    # Column headers (top row only) — quantity names only, units go on the colorbars
    col_headers = ['IV-OCT', 'OAC', 'Rendered', 'Residual']
    panel_col_starts = [0, 5]  # columns where each panel starts (gap at col 4)
    for pc_start in panel_col_starts:
        for sub in range(n_sub_cols):
            axes[0, pc_start + sub].set_title(col_headers[sub],
                                              fontsize=COL_HEADER_SIZE,
                                              fontweight='bold', pad=6)

    # Compute global dB display range across all frames (cropped window)
    all_dB_vmin, all_dB_vmax = [], []
    for fi in frame_indices:
        fd = os.path.join(out_dir, f'frame_{fi:03d}')
        gt = _load_cart(fd, 'gt_cart.npy')
        vmin, vmax, _ = _get_display_params(gt)
        all_dB_vmin.append(vmin)
        all_dB_vmax.append(vmax)
        del gt
    global_dB_vmin = float(np.median(all_dB_vmin))
    global_dB_vmax = float(np.median(all_dB_vmax))

    # Global residual scale
    all_vabs = []
    for fi in frame_indices:
        fd = os.path.join(out_dir, f'frame_{fi:03d}')
        gt = _load_cart(fd, 'gt_cart.npy')
        rs1 = _load_cart(fd, 'rendered_s1_cart.npy')
        tissue = gt > 1.0
        diff = np.where(tissue, rs1 - gt, 0.0)
        if tissue.any():
            all_vabs.append(float(np.abs(diff[tissue]).max()))
        del gt, rs1, diff
    global_vabs = min(float(np.median(all_vabs)), 20.0)

    for idx, fi in enumerate(frame_indices):
        row, panel_col = divmod(idx, n_panel_cols)
        fd = os.path.join(out_dir, f'frame_{fi:03d}')
        gt = _load_cart(fd, 'gt_cart.npy')
        mu = _load_cart(fd, 'mu_cart.npy')
        rs1 = _load_cart(fd, 'rendered_s1_cart.npy')

        _, _, tissue = _get_display_params(gt)
        mu_display = np.where(tissue, mu, 0.0)
        diff = np.where(tissue, rs1 - gt, 0.0)

        c0 = panel_col_starts[panel_col]

        # Row label — left column on the IV-OCT axis, right column in the gap
        if panel_col == 0:
            axes[row, c0].set_ylabel(f'Frame {fi}', fontsize=ROW_LABEL_SIZE,
                                     fontweight='bold', rotation=90, labelpad=6)
        else:
            # Place right-column label in the gap column (col 4)
            gap_ax = axes[row, 4]
            gap_ax.set_visible(True)
            gap_ax.set_xlim(0, 1); gap_ax.set_ylim(0, 1)
            gap_ax.text(0.5, 0.5, f'Frame {fi}', fontsize=ROW_LABEL_SIZE,
                        fontweight='bold', ha='center', va='center', rotation=90)
            gap_ax.axis('off')

        # IV-OCT
        axes[row, c0].imshow(np.clip(gt, 0, None), cmap='gray',
                             vmin=global_dB_vmin, vmax=global_dB_vmax)
        axes[row, c0].set_xticks([]); axes[row, c0].set_yticks([])
        for sp in axes[row, c0].spines.values(): sp.set_visible(False)

        # OAC
        axes[row, c0+1].imshow(mu_display, cmap='viridis',
                               vmin=0, vmax=global_mu_vmax)
        axes[row, c0+1].axis('off')

        # Rendered
        axes[row, c0+2].imshow(np.clip(rs1, 0, None), cmap='gray',
                               vmin=global_dB_vmin, vmax=global_dB_vmax)
        axes[row, c0+2].axis('off')

        # Residual
        axes[row, c0+3].imshow(diff, cmap='coolwarm',
                               vmin=-global_vabs, vmax=global_vabs)
        axes[row, c0+3].axis('off')

        del gt, mu, rs1, diff

    # Hide any unused cells (for odd n_frames, the bottom-right panel slot is empty)
    total_slots = n_panel_rows * n_panel_cols
    for extra_idx in range(n_frames, total_slots):
        row, panel_col = divmod(extra_idx, n_panel_cols)
        c0 = panel_col_starts[panel_col]
        for sub in range(n_sub_cols):
            axes[row, c0 + sub].set_visible(False)
        # Also hide the row-label gap axis for that slot if it was being used
        if panel_col != 0:
            axes[row, 4].set_visible(False)

    # 4 colorbars directly under the left panel's bottom row, aligned to each column
    # Get the bounding boxes of the bottom-left panel axes after layout
    fig.canvas.draw()
    def _ax_xbounds(ax):
        bbox = ax.get_position()
        return bbox.x0, bbox.x1
    bot_row = n_panel_rows - 1
    y_bottom = axes[bot_row, 0].get_position().y0  # bottom edge of last row

    cbar_y = y_bottom - 0.04   # just below the last row
    cbar_h = 0.012

    cbar_info = [
        (0, 'gray',     plt.Normalize(global_dB_vmin, global_dB_vmax), 'dB'),
        (1, 'viridis',  plt.Normalize(0, global_mu_vmax),              'mm$^{-1}$'),
        (2, 'gray',     plt.Normalize(global_dB_vmin, global_dB_vmax), 'dB'),
        (3, 'coolwarm', plt.Normalize(-global_vabs, global_vabs),      'dB'),
    ]
    for pc_start in panel_col_starts:
        for sub_col, cmap_name, norm, label in cbar_info:
            col_idx = pc_start + sub_col
            x0, x1 = _ax_xbounds(axes[bot_row, col_idx])
            cax = fig.add_axes([x0, cbar_y, x1 - x0, cbar_h])
            sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
            cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
            cb.set_label(label, fontsize=CBAR_LABEL_SIZE)
            cb.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    gc.collect()
    path1 = os.path.join(paper_dir, 'clinical_src.pdf')
    plt.savefig(path1, dpi=300, bbox_inches='tight', pad_inches=0.02)
    path1_mirror = os.path.join(out_dir, 'clinical_src.pdf')
    plt.savefig(path1_mirror, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    del fig
    gc.collect()
    print(f"  Saved: {path1}")
    print(f"  Saved: {path1_mirror}")

    # =====================================================================
    # supp_fig_alpha_refinement.pdf
    # Transposed layout: 6 rows (data type) x 12 cols (frames).
    # Rows   : GT | OAC (mu) | Rendered (mu) | Alpha | Rendered (mu+alpha) | Residual
    # Columns: Frame {fi} for each of the 12 frames
    # Colorbars on the right edge, one per row.
    # =====================================================================
    print("Generating supp_fig_alpha_refinement.pdf ...")

    # Plasma colormap where NaN (non-tissue) pixels render with the darkest
    # plasma color (blue-purple at the bottom of the range) instead of the
    # matplotlib default (transparent / white figure background).
    _plasma_bg = plt.get_cmap('plasma').copy()
    _plasma_bg.set_bad(_plasma_bg(0.0))

    # Row labels on the left: quantity name only (units go on the colorbar)
    # Colorbar labels on the right: units only
    supp_rows = [
        ('GT',                        'gray',      plt.Normalize(global_dB_vmin, global_dB_vmax), 'dB'),
        ('OAC',                       'viridis',   plt.Normalize(0, global_mu_vmax),              'mm$^{-1}$'),
        ('Rendered ($\\mu$)',         'gray',      plt.Normalize(global_dB_vmin, global_dB_vmax), 'dB'),
        ('Alpha',                     _plasma_bg,  LogNorm(vmin=max(global_alpha_vmin, 1e-10), vmax=global_alpha_vmax), 'a.u.'),
        ('Rendered ($\\mu+\\alpha$)', 'gray',      plt.Normalize(global_dB_vmin, global_dB_vmax), 'dB'),
        ('Residual',                  'coolwarm',  plt.Normalize(-global_vabs, global_vabs),      'dB'),
    ]
    n_supp_rows = len(supp_rows)          # 6
    n_supp_cols = len(frame_indices)      # 8 or 12

    # Size the figure so each image cell is ~2.5 units square (same per-image
    # footprint as clinical_src's 21-wide / 8-image layout) and rows stack
    # tightly like clinical_src.
    cell = 2.6
    supp_figsize = (n_supp_cols * cell, n_supp_rows * cell)
    fig, axes = plt.subplots(n_supp_rows, n_supp_cols,
                             figsize=supp_figsize,
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.005})

    # Column headers (top row only): Frame {fi}
    for col_idx, fi in enumerate(frame_indices):
        axes[0, col_idx].set_title(f'Frame {fi}',
                                   fontsize=COL_HEADER_SIZE,
                                   fontweight='bold', pad=6)

    # Row labels (left column only)
    for row_idx, (label, _cmap, _norm, _lab) in enumerate(supp_rows):
        axes[row_idx, 0].set_ylabel(label, fontsize=ROW_LABEL_SIZE,
                                     fontweight='bold', rotation=90, labelpad=10)

    # Fill the grid, column by column (= frame by frame)
    for col_idx, fi in enumerate(frame_indices):
        fd = os.path.join(out_dir, f'frame_{fi:03d}')
        gt = _load_cart(fd, 'gt_cart.npy')
        mu = _load_cart(fd, 'mu_cart.npy')
        alpha = _load_cart(fd, 'alpha_cart.npy')
        rs1 = _load_cart(fd, 'rendered_s1_cart.npy')
        rs2 = _load_cart(fd, 'rendered_s2_cart.npy')

        _, _, tissue = _get_display_params(gt)
        mu_display = np.where(tissue, mu, 0.0)
        alpha_display = np.where(tissue, alpha, np.nan)
        diff_s2 = np.where(tissue, rs2 - gt, 0.0)

        row_images = [
            (np.clip(gt, 0, None),  'gray',      dict(vmin=global_dB_vmin, vmax=global_dB_vmax)),
            (mu_display,            'viridis',   dict(vmin=0, vmax=global_mu_vmax)),
            (np.clip(rs1, 0, None), 'gray',      dict(vmin=global_dB_vmin, vmax=global_dB_vmax)),
            (alpha_display,         _plasma_bg,  dict(norm=LogNorm(vmin=max(global_alpha_vmin, 1e-10), vmax=global_alpha_vmax))),
            (np.clip(rs2, 0, None), 'gray',      dict(vmin=global_dB_vmin, vmax=global_dB_vmax)),
            (diff_s2,               'coolwarm',  dict(vmin=-global_vabs, vmax=global_vabs)),
        ]

        for row_idx, (img, cmap, imshow_kwargs) in enumerate(row_images):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap=cmap, **imshow_kwargs)
            ax.set_xticks([]); ax.set_yticks([])
            # keep the leftmost axis's ylabel visible (already set above); hide spines
            for sp in ax.spines.values(): sp.set_visible(False)

        del gt, mu, alpha, rs1, rs2, diff_s2

    # One colorbar per row, pinned to the right edge of that row
    fig.canvas.draw()
    cbar_pad = 0.012
    cbar_w = 0.016
    for row_idx, (_label, cmap_name, norm, unit) in enumerate(supp_rows):
        bbox = axes[row_idx, -1].get_position()
        cax = fig.add_axes([bbox.x1 + cbar_pad, bbox.y0, cbar_w, bbox.height])
        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(unit, fontsize=CBAR_LABEL_SIZE)
        cb.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    gc.collect()
    path2 = os.path.join(paper_dir, 'supp_fig_alpha_refinement.pdf')
    plt.savefig(path2, dpi=300, bbox_inches='tight', pad_inches=0.02)
    path2_mirror = os.path.join(out_dir, 'supp_fig_alpha_refinement.pdf')
    plt.savefig(path2_mirror, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    del fig
    gc.collect()
    print(f"  Saved: {path2}")
    print(f"  Saved: {path2_mirror}")

    # =====================================================================
    # supp_fig_alpha_refinement_rows.pdf
    # Variant layout: n_frames rows (one per frame) x 6 cols (data types).
    # Columns: GT | OAC | Rendered (mu) | Alpha | Rendered (mu+alpha) | Residual
    # Rows   : Frame {fi} for each of the frames
    # Colorbars at the bottom, one per column.
    # =====================================================================
    print("Generating supp_fig_alpha_refinement_rows.pdf ...")

    # Reuse supp_rows (defined above): (label, cmap, norm, unit)
    n_row_rows = len(frame_indices)       # frames
    n_row_cols = len(supp_rows)           # 6 data types

    cell_r = 2.6
    rows_figsize = (n_row_cols * cell_r, n_row_rows * cell_r)
    fig, axes = plt.subplots(n_row_rows, n_row_cols,
                             figsize=rows_figsize,
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.005})

    # Column headers (top row only) — same labels as the transposed version
    for col_idx, (label, _cmap, _norm, _unit) in enumerate(supp_rows):
        axes[0, col_idx].set_title(label, fontsize=COL_HEADER_SIZE,
                                   fontweight='bold', pad=6)

    # Row labels (left column only) — frame indices
    for row_idx, fi in enumerate(frame_indices):
        axes[row_idx, 0].set_ylabel(f'Frame {fi}',
                                     fontsize=ROW_LABEL_SIZE,
                                     fontweight='bold', rotation=90, labelpad=10)

    # Per-row crops (operate on uncropped 500x500 Cartesian frames).
    def _crop_for_row(row_idx):
        if row_idx == 0:    # Frame 0  — image moves down 15 (window up), zoomed out 10 each dir
            return np.s_[80:340, 125:385]   # 260x260, matches frames 42/85/127
        if row_idx == 1:    # Frame 42 — image moves right+down 20 (window left+up), zoom in 20 each dir
            return np.s_[100:360, 100:360]
        if row_idx == 2:    # Frame 85 — zoom in 20 each dir
            return np.s_[120:380, 120:380]
        if row_idx == 3:    # Frame 127 — image moves left 20 (window right), zoom in 20 each dir
            return np.s_[120:380, 140:400]
        return np.s_[100:400, 100:400]       # Frame 170 — middle 300x300

    # Fill the grid, row by row (= frame by frame)
    for row_idx, fi in enumerate(frame_indices):
        fd = os.path.join(out_dir, f'frame_{fi:03d}')
        crop = _crop_for_row(row_idx)
        gt = np.load(os.path.join(fd, 'gt_cart.npy'))[crop]
        mu = np.load(os.path.join(fd, 'mu_cart.npy'))[crop]
        alpha = np.load(os.path.join(fd, 'alpha_cart.npy'))[crop]
        rs1 = np.load(os.path.join(fd, 'rendered_s1_cart.npy'))[crop]
        rs2 = np.load(os.path.join(fd, 'rendered_s2_cart.npy'))[crop]

        _, _, tissue = _get_display_params(gt)
        mu_display = np.where(tissue, mu, 0.0)
        alpha_display = np.where(tissue, alpha, np.nan)
        diff_s2 = np.where(tissue, rs2 - gt, 0.0)

        col_images = [
            (np.clip(gt, 0, None),  'gray',      dict(vmin=global_dB_vmin, vmax=global_dB_vmax)),
            (mu_display,            'viridis',   dict(vmin=0, vmax=global_mu_vmax)),
            (np.clip(rs1, 0, None), 'gray',      dict(vmin=global_dB_vmin, vmax=global_dB_vmax)),
            (alpha_display,         _plasma_bg,  dict(norm=LogNorm(vmin=max(global_alpha_vmin, 1e-10), vmax=global_alpha_vmax))),
            (np.clip(rs2, 0, None), 'gray',      dict(vmin=global_dB_vmin, vmax=global_dB_vmax)),
            (diff_s2,               'coolwarm',  dict(vmin=-global_vabs, vmax=global_vabs)),
        ]

        for col_idx, (img, cmap, imshow_kwargs) in enumerate(col_images):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap=cmap, **imshow_kwargs)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)

        del gt, mu, alpha, rs1, rs2, diff_s2

    # One colorbar per column, pinned to the bottom of that column
    fig.canvas.draw()
    cbar_pad_bot = 0.012
    cbar_h_bot = 0.014
    for col_idx, (_label, cmap_name, norm, unit) in enumerate(supp_rows):
        bbox = axes[-1, col_idx].get_position()
        cax = fig.add_axes([bbox.x0, bbox.y0 - cbar_pad_bot - cbar_h_bot,
                             bbox.width, cbar_h_bot])
        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
        if unit:
            cb.set_label(unit, fontsize=CBAR_LABEL_SIZE)
        cb.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    gc.collect()
    path3 = os.path.join(paper_dir, 'supp_fig_alpha_refinement_rows.pdf')
    plt.savefig(path3, dpi=300, bbox_inches='tight', pad_inches=0.02)
    path3_mirror = os.path.join(out_dir, 'supp_fig_alpha_refinement_rows.pdf')
    plt.savefig(path3_mirror, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    del fig
    gc.collect()
    print(f"  Saved: {path3}")
    print(f"  Saved: {path3_mirror}")

    # Print summary table (PSNR + Dice at 15 dB noise floor, both stages)
    print(f"\n{'='*74}")
    print(f"  Rendering Quality Summary ({patient_id})")
    print(f"{'='*74}")
    print(f"  {'Frame':>6s}  {'S1 PSNR':>10s}  {'S2 PSNR':>10s}  {'S1 Dice':>10s}  {'S2 Dice':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for fi, p1, p2, d1, d2 in zip(frame_indices, psnr_s1_list, psnr_s2_list,
                                  dice_s1_list, dice_s2_list):
        print(f"  {fi:>6d}  {p1:>10.2f}  {p2:>10.2f}  {d1:>10.3f}  {d2:>10.3f}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    s1_psnr = np.array(psnr_s1_list); s2_psnr = np.array(psnr_s2_list)
    s1_dice = np.array(dice_s1_list); s2_dice = np.array(dice_s2_list)
    print(f"  {'Mean':>6s}  {s1_psnr.mean():>10.2f}  {s2_psnr.mean():>10.2f}  "
          f"{s1_dice.mean():>10.3f}  {s2_dice.mean():>10.3f}")
    print(f"  {'Std':>6s}  {s1_psnr.std():>10.2f}  {s2_psnr.std():>10.2f}  "
          f"{s1_dice.std():>10.3f}  {s2_dice.std():>10.3f}")
    print(f"  {'Min':>6s}  {s1_psnr.min():>10.2f}  {s2_psnr.min():>10.2f}  "
          f"{s1_dice.min():>10.3f}  {s2_dice.min():>10.3f}")
    print(f"  {'Max':>6s}  {s1_psnr.max():>10.2f}  {s2_psnr.max():>10.2f}  "
          f"{s1_dice.max():>10.3f}  {s2_dice.max():>10.3f}")


# ===================================================================
# Main
# ===================================================================

def main():
    patient_id = os.environ.get('PATIENT_ID', '02008Pre')
    num_iters = int(os.environ.get('NUM_ITERS', '50000'))
    db_scale = float(os.environ.get('DB_SCALE', '130.0'))
    noise_floor_dB = float(os.environ.get('NOISE_FLOOR_DB', '15.0'))
    composite_only = os.environ.get('COMPOSITE_ONLY', '') == '1'
    summary_only = os.environ.get('SUMMARY', '') == '1'
    array_task_id = os.environ.get('ARRAY_TASK_ID', '')

    print(f"{'='*70}")
    print(f"20_render_IVOCT.py — IV-OCT Property Extraction and Rendering")
    print(f"{'='*70}")
    print(f"Patient:       {patient_id}")

    # Output directory
    out_dir = os.path.join(PROJECT_ROOT, 'outputs', '20_render_ivoct', patient_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir:    {out_dir}")

    # --- Summary-only or composite-only mode: build paper figures (no NIfTI needed) ---
    if summary_only or composite_only:
        fi_path = os.path.join(out_dir, 'frame_indices.npy')
        if not os.path.exists(fi_path):
            print(f"ERROR: {fi_path} not found. Run optimization first.")
            return
        all_frame_indices = np.load(fi_path).tolist()
        all_done = all(
            os.path.exists(os.path.join(out_dir, f'frame_{fi:03d}', 'mu_final_mm.npy'))
            for fi in all_frame_indices)
        if all_done and len(all_frame_indices) == 12:
            save_paper_figures(out_dir, all_frame_indices, patient_id)
        else:
            missing = [fi for fi in all_frame_indices
                       if not os.path.exists(
                           os.path.join(out_dir, f'frame_{fi:03d}', 'mu_final_mm.npy'))]
            print(f"Cannot build figures — missing frames: {missing}")
        return

    # --- Full optimization mode: load patient data ---
    print(f"Iterations:    {num_iters}")
    print(f"dB scale:      {db_scale}")
    print(f"Noise floor:   {noise_floor_dB} dB")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:        {device}")

    patient_path = find_patient_file(patient_id)
    print(f"Loading patient data...")
    patient_data = load_nifti(patient_path)
    print(f"  Shape: {patient_data.shape}")
    print(f"  Range: [{patient_data.min():.4f}, {patient_data.max():.4f}]")

    num_frames = patient_data.shape[2]
    pixel_max = float(patient_data.max())

    frame_indices_str = os.environ.get('FRAME_INDICES', '')
    if frame_indices_str:
        all_frame_indices = [int(x.strip()) for x in frame_indices_str.split(',')]
    else:
        all_frame_indices = np.linspace(0, num_frames - 1, 12, dtype=int).tolist()
    print(f"Frame indices ({len(all_frame_indices)}): {all_frame_indices}")

    np.save(os.path.join(out_dir, 'frame_indices.npy'), np.array(all_frame_indices))

    # --- Determine which frames to process ---
    if array_task_id:
        task_idx = int(array_task_id)
        if task_idx >= len(all_frame_indices):
            print(f"ARRAY_TASK_ID={task_idx} >= num frames ({len(all_frame_indices)}). "
                  f"Nothing to do.")
            return
        frames_to_process = [all_frame_indices[task_idx]]
    else:
        frames_to_process = list(all_frame_indices)

    # Build configs — Stage 1: mu only (alpha=1.0), Stage 2: alpha only
    num_iters_s2 = int(os.environ.get('NUM_ITERS_S2', str(num_iters)))
    config = build_ivoct_config(noise_floor_dB=noise_floor_dB)
    opt_config_s1 = build_opt_config(num_iterations=num_iters,
                                     alpha_min=1.0, alpha_max=1.0)
    opt_config_s2 = build_opt_config(num_iterations=num_iters_s2,
                                     alpha_min=1e-10, alpha_max=1e-2)

    # --- Process frames ---
    for fi in frames_to_process:
        frame_dir = os.path.join(out_dir, f'frame_{fi:03d}')

        # Skip if already done
        if os.path.exists(os.path.join(frame_dir, 'mu_final_mm.npy')):
            print(f"\nFrame {fi}: already done — skipping.")
            continue

        frame_cart = patient_data[:, :, fi]
        result = process_frame(
            frame_cart, fi, config, opt_config_s1, opt_config_s2,
            noise_floor_dB, db_scale, pixel_max, device, frame_dir)

        # Save individual panel
        save_frame_panel(result, os.path.join(frame_dir, 'panel.jpg'))

    # --- Build paper figures if all 12 frames are done ---
    all_done = all(
        os.path.exists(os.path.join(out_dir, f'frame_{fi:03d}', 'mu_final_mm.npy'))
        for fi in all_frame_indices)
    if all_done and len(all_frame_indices) == 12:
        save_paper_figures(out_dir, all_frame_indices, patient_id)
    elif not all_done:
        done = sum(1 for fi in all_frame_indices
                   if os.path.exists(
                       os.path.join(out_dir, f'frame_{fi:03d}', 'mu_final_mm.npy')))
        print(f"\n{done}/{len(all_frame_indices)} frames done — "
              f"run with SUMMARY=1 when all complete.")


if __name__ == '__main__':
    main()
