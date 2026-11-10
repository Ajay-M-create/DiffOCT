#!/usr/bin/env python3
"""
Shockwave Arteries IV-OCT Processing
=====================================
Loads IV-OCT patient data and one-hot segmentations from NIfTI files,
converts Cartesian cross-sections to polar (depth x A-lines) representation,
runs single-scatter OAC optimization using the existing forward/inverse models,
and evaluates recovered attenuation coefficients per tissue class (wall, calcium).

Usage:
    python test-shockwave-arteries/shockwave_arteries_process.py

Environment variables:
    PATIENT_ID      Patient prefix, e.g. "02008Pre" or "03005Post" (default: "02008Pre")
    FRAME_INDICES   Comma-separated frame indices to process, e.g. "50,85,120" (default: 3 evenly spaced)
    NUM_ITERS       Number of optimization iterations (default: 5000)
    DB_SCALE        Scale factor to convert pixel values to dB (default: 130.0)
    NOISE_FLOOR_DB  Noise floor threshold in dB (default: 79.0, matching phantom pipeline)
"""

import os
import sys
import re
import time
import glob
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import map_coordinates

# ---------------------------------------------------------------------------
# Path setup -- import src modules and configs
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from src.forward import RasterScanOCTForward
from src.inverse import optimize, build_noise_floor_mask, subtract_noise_floor_linear

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
SEGMENTATION_DIR = os.environ.get("SEGMENTATION_DATA_DIR", "/home/gridsan/amanicka/erelab_shared/Ajay/DiffOCT-IVLDatasetsAndSegsPrePost/onehot_nifti_segmentations")
PATIENT_DIR = os.environ.get("PATIENT_DATA_DIR", "/home/gridsan/amanicka/erelab_shared/Ajay/DiffOCT-IVLDatasetsAndSegsPrePost/Pre_Post")

# Tissue class labels (channels in one-hot segmentation)
TISSUE_CLASSES = {0: 'background', 1: 'lumen', 2: 'wall', 3: 'calcium'}
TISSUE_COLORS = {0: (0, 0, 0, 0), 1: (1, 0, 0, 0.4), 2: (0, 1, 0, 0.4), 3: (0, 0, 1, 0.5)}

# IV-OCT geometry
NUM_ALINES = 500          # angular samples (0.72 deg apart)
DEPTH_PIXELS = 250        # center to edge
IMAGE_SIZE = 500          # Cartesian image size
CENTER = IMAGE_SIZE // 2  # catheter center pixel


# ===================================================================
# Data loading
# ===================================================================

def find_patient_files(patient_id):
    """Find patient image and segmentation NIfTI files for a given patient ID.

    Patient files have naming like: 02008Pre_target_Post.nii.gz or 02008Post_reg.nii.gz
    Segmentation files have naming like: 002008Pre_seg_onehot.nii.gz (note extra leading 0)
    """
    # Find patient image file
    patient_pattern = os.path.join(PATIENT_DIR, f"{patient_id}*.nii.gz")
    patient_files = sorted(glob.glob(patient_pattern))
    if not patient_files:
        raise FileNotFoundError(f"No patient file found matching {patient_pattern}")

    # Find segmentation file -- numeric part is always 6 digits (zero-padded)
    match = re.match(r'(\d+)(Pre|Post)', patient_id)
    if not match:
        raise ValueError(f"Cannot parse patient_id '{patient_id}'. Expected format: '02008Pre' or '03005Post'")
    num_part, state = match.groups()
    seg_prefix = num_part.zfill(6) + state
    seg_pattern = os.path.join(SEGMENTATION_DIR, f"{seg_prefix}*seg_onehot.nii.gz")
    seg_files = sorted(glob.glob(seg_pattern))
    if not seg_files:
        raise FileNotFoundError(f"No segmentation file found matching {seg_pattern}")

    print(f"Patient file:      {patient_files[0]}")
    print(f"Segmentation file: {seg_files[0]}")
    return patient_files[0], seg_files[0]


def load_nifti(path):
    """Load a NIfTI file and return the data array."""
    import nibabel as nib
    img = nib.load(path)
    return img.get_fdata()


# ===================================================================
# Cartesian to polar conversion
# ===================================================================

def cartesian_to_polar(image_2d, center=(CENTER, CENTER),
                       num_alines=NUM_ALINES, depth_pixels=DEPTH_PIXELS):
    """Convert a Cartesian 500x500 IV-OCT image to polar representation.

    Extracts radial A-lines from center outward.

    Args:
        image_2d: (500, 500) Cartesian image
        center: (cy, cx) catheter center in pixels
        num_alines: number of angular samples
        depth_pixels: number of radial samples (center to edge)

    Returns:
        polar: (depth_pixels, num_alines) array -- depth goes top-to-bottom,
               A-lines go left-to-right (angular direction)
    """
    cy, cx = center
    angles = np.linspace(0, 2 * np.pi, num_alines, endpoint=False)
    radii = np.arange(depth_pixels, dtype=np.float64)

    # Build coordinate grid for interpolation
    r_grid, a_grid = np.meshgrid(radii, angles, indexing='ij')
    y_coords = cy + r_grid * np.sin(a_grid)
    x_coords = cx + r_grid * np.cos(a_grid)

    coords = np.array([y_coords, x_coords])
    polar = map_coordinates(image_2d, coords, order=1, mode='constant', cval=0.0)
    return polar


def cartesian_seg_to_polar(seg_2d_onehot, center=(CENTER, CENTER),
                           num_alines=NUM_ALINES, depth_pixels=DEPTH_PIXELS):
    """Convert one-hot segmentation from Cartesian to polar using nearest-neighbor.

    Args:
        seg_2d_onehot: (500, 500, 4) one-hot segmentation

    Returns:
        seg_polar: (depth_pixels, num_alines) integer class labels (0-3)
    """
    class_labels = np.argmax(seg_2d_onehot, axis=-1).astype(np.float64)

    cy, cx = center
    angles = np.linspace(0, 2 * np.pi, num_alines, endpoint=False)
    radii = np.arange(depth_pixels, dtype=np.float64)

    r_grid, a_grid = np.meshgrid(radii, angles, indexing='ij')
    y_coords = cy + r_grid * np.sin(a_grid)
    x_coords = cx + r_grid * np.cos(a_grid)

    coords = np.array([y_coords, x_coords])
    seg_polar = map_coordinates(class_labels, coords, order=0, mode='constant', cval=0.0)
    return seg_polar.astype(np.int32)


def polar_to_cartesian(polar_img, center=(CENTER, CENTER),
                       image_size=IMAGE_SIZE, num_alines=NUM_ALINES,
                       depth_pixels=DEPTH_PIXELS, fill_value=0.0):
    """Convert a polar (depth x A-lines) image back to Cartesian (500x500).

    For each Cartesian pixel, computes its (radius, angle) and samples from polar.

    Args:
        polar_img: (depth_pixels, num_alines) polar image
        center: (cy, cx) center in Cartesian coordinates
        image_size: output Cartesian image size
        fill_value: value for pixels outside the polar range

    Returns:
        cart: (image_size, image_size) Cartesian image
    """
    cy, cx = center
    y_coords, x_coords = np.mgrid[0:image_size, 0:image_size]
    dy = y_coords - cy
    dx = x_coords - cx
    radii = np.sqrt(dy**2 + dx**2)
    angles = np.arctan2(dy, dx)  # [-pi, pi]
    angles = np.mod(angles, 2 * np.pi)  # [0, 2*pi)

    # Map angles to A-line index (continuous)
    aline_idx = angles / (2 * np.pi) * num_alines
    # radii already in pixel units (0 to ~250)

    coords = np.array([radii, aline_idx])
    # Use mode='constant' so pixels outside depth range get fill_value
    cart = map_coordinates(polar_img, coords, order=1, mode='constant', cval=fill_value)
    # Mask out pixels beyond the max radius
    cart[radii >= depth_pixels] = fill_value
    return cart


# ===================================================================
# Visualization
# ===================================================================

def plot_segmentation_overlay(patient_data, seg_data, frame_indices, out_dir, patient_id):
    """Plot Cartesian frames with segmentation overlays.

    Creates a figure with rows = frames, cols = [image, seg overlay, polar image, polar seg].
    """
    n_frames = len(frame_indices)
    fig, axes = plt.subplots(n_frames, 4, figsize=(20, 5 * n_frames))
    if n_frames == 1:
        axes = axes[np.newaxis, :]

    for row, fi in enumerate(frame_indices):
        frame = patient_data[:, :, fi]
        seg_frame = seg_data[:, :, fi, :]  # (500, 500, 4)
        class_map = np.argmax(seg_frame, axis=-1)

        # Column 0: raw Cartesian image
        axes[row, 0].imshow(frame, cmap='gray', aspect='equal')
        axes[row, 0].set_title(f'Frame {fi} -- Cartesian')
        axes[row, 0].set_ylabel(f'Frame {fi}')

        # Column 1: image + segmentation overlay
        axes[row, 1].imshow(frame, cmap='gray', aspect='equal')
        overlay = np.zeros((*frame.shape, 4))
        for cls_id in [1, 2, 3]:
            mask = class_map == cls_id
            overlay[mask] = TISSUE_COLORS[cls_id]
        axes[row, 1].imshow(overlay, aspect='equal')
        axes[row, 1].set_title(f'Seg overlay (R=lumen, G=wall, B=calcium)')

        # Column 2: polar image
        polar_img = cartesian_to_polar(frame)
        axes[row, 2].imshow(polar_img, cmap='gray', aspect='auto', origin='upper')
        axes[row, 2].set_title(f'Polar (depth x A-lines)')
        axes[row, 2].set_xlabel('A-line index')
        axes[row, 2].set_ylabel('Depth (px from center)')

        # Column 3: polar segmentation
        seg_polar = cartesian_seg_to_polar(seg_frame)
        seg_cmap = ListedColormap([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        axes[row, 3].imshow(seg_polar, cmap=seg_cmap, vmin=0, vmax=3,
                            aspect='auto', origin='upper')
        axes[row, 3].set_title(f'Polar seg (R=lumen, G=wall, B=calcium)')
        axes[row, 3].set_xlabel('A-line index')

    plt.suptitle(f'Patient {patient_id} -- Segmentation Overview', fontsize=16, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{patient_id}_segmentation_overview.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved segmentation overview: {out_path}")


def plot_noise_floor_comparison(patient_data, seg_data, frame_idx, db_scale,
                                out_dir, patient_id):
    """Show how a single Cartesian frame looks under different noise floor thresholds.

    Displays the original image alongside masked versions at several noise floor dB values.
    """
    frame = patient_data[:, :, frame_idx]
    seg_frame = seg_data[:, :, frame_idx, :]
    class_map = np.argmax(seg_frame, axis=-1)

    pixel_max = patient_data.max()
    if pixel_max > 0:
        frame_dB = frame * (db_scale / pixel_max)
    else:
        frame_dB = frame * db_scale

    noise_floors = [0, 2, 5, 8, 10, 13, 17, 20, 25, 30, 35, 40, 45, 50,
                     55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    n_cols = len(noise_floors) + 1  # original + each threshold
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

    # Row 0: masked dB images; Row 1: masked with seg overlay
    # First column: original
    axes[0, 0].imshow(frame_dB, cmap='gray', aspect='equal')
    axes[0, 0].set_title('Original', fontsize=9)
    axes[1, 0].imshow(frame_dB, cmap='gray', aspect='equal')
    overlay = np.zeros((*frame.shape, 4))
    for cls_id in [1, 2, 3]:
        overlay[class_map == cls_id] = TISSUE_COLORS[cls_id]
    axes[1, 0].imshow(overlay, aspect='equal')
    axes[1, 0].set_title('Original + seg', fontsize=9)

    for col_idx, nf in enumerate(noise_floors, start=1):
        mask = frame_dB >= nf
        masked = np.where(mask, frame_dB, 0.0)
        pct_kept = 100.0 * mask.sum() / mask.size

        axes[0, col_idx].imshow(masked, cmap='gray', aspect='equal',
                                 vmin=0, vmax=frame_dB.max())
        axes[0, col_idx].set_title(f'NF={nf} dB\n{pct_kept:.1f}% kept', fontsize=9)

        axes[1, col_idx].imshow(masked, cmap='gray', aspect='equal',
                                 vmin=0, vmax=frame_dB.max())
        axes[1, col_idx].imshow(overlay, aspect='equal')
        axes[1, col_idx].set_title(f'NF={nf} dB + seg', fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Patient {patient_id} -- Frame {frame_idx} -- Noise Floor Comparison',
                 fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{patient_id}_noise_floor_comparison_frame{frame_idx}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved noise floor comparison: {out_path}")


def plot_mu_results(mu_polar, alpha_polar, seg_polar, polar_target_dB, pred_dB, metrics,
                    frame_idx, out_dir, patient_id, noise_floor_dB,
                    cart_image, cart_seg_onehot):
    """Plot optimization results with mu, alpha, and gain analysis for all tissue types.

    Layout: 4x3 grid
        Row 0: Image+seg, mu map+contours, alpha map+contours
        Row 1: Target dB, Predicted dB, Residual
        Row 2: Loss+gain curves, mu histogram by class, alpha histogram by class
        Row 3: Per-tissue residual boxplot, mu summary text, alpha+gain summary text
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))

    noise_mask_np = (polar_target_dB >= noise_floor_dB).astype(float)
    tissue_pixels = noise_mask_np > 0.5
    class_map_cart = np.argmax(cart_seg_onehot, axis=-1)

    # Convert polar results back to Cartesian for display
    mu_cart = polar_to_cartesian(mu_polar, fill_value=0.0)
    alpha_cart = polar_to_cartesian(alpha_polar, fill_value=0.0)
    pred_dB_cart = polar_to_cartesian(pred_dB, fill_value=0.0)
    noise_mask_cart = polar_to_cartesian(noise_mask_np, fill_value=0.0)
    tissue_cart = noise_mask_cart > 0.5

    colors_hist = {'wall': 'green', 'calcium': 'blue', 'lumen': 'red'}

    # --- Row 0: Image + seg, mu map, alpha map ---
    axes[0, 0].imshow(cart_image, cmap='gray', aspect='equal')
    overlay = np.zeros((*cart_image.shape, 4))
    for cls_id in [1, 2, 3]:
        mask = class_map_cart == cls_id
        overlay[mask] = TISSUE_COLORS[cls_id]
    axes[0, 0].imshow(overlay, aspect='equal')
    axes[0, 0].set_title(f'Frame {frame_idx} + Segmentation')

    mu_cart_display = mu_cart.copy()
    mu_cart_display[~tissue_cart] = 0
    mu_vmax = np.percentile(mu_cart_display[mu_cart_display > 0], 99) if (mu_cart_display > 0).any() else 1.0
    im1 = axes[0, 1].imshow(mu_cart_display, cmap='hot', aspect='equal', vmin=0, vmax=mu_vmax)
    for cls_id, color in [(1, 'red'), (2, 'lime'), (3, 'cyan')]:
        mask_cls = (class_map_cart == cls_id).astype(float)
        if mask_cls.any():
            axes[0, 1].contour(mask_cls, levels=[0.5], colors=[color], linewidths=0.8)
    axes[0, 1].set_title('mu (mm^-1) + seg contours')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    alpha_cart_display = alpha_cart.copy()
    alpha_cart_display[~tissue_cart] = 0
    alpha_tissue_vals = alpha_cart_display[alpha_cart_display > 0]
    alpha_vmax = np.percentile(alpha_tissue_vals, 99) if len(alpha_tissue_vals) > 0 else 1e-2
    alpha_vmin = np.percentile(alpha_tissue_vals, 1) if len(alpha_tissue_vals) > 0 else 0
    im_a = axes[0, 2].imshow(alpha_cart_display, cmap='viridis', aspect='equal',
                              vmin=alpha_vmin, vmax=alpha_vmax)
    for cls_id, color in [(1, 'red'), (2, 'lime'), (3, 'cyan')]:
        mask_cls = (class_map_cart == cls_id).astype(float)
        if mask_cls.any():
            axes[0, 2].contour(mask_cls, levels=[0.5], colors=[color], linewidths=0.8)
    axes[0, 2].set_title('alpha + seg contours')
    plt.colorbar(im_a, ax=axes[0, 2], fraction=0.046)

    # --- Row 1: Target dB, Predicted dB, Residual ---
    cart_dB = polar_to_cartesian(polar_target_dB, fill_value=0.0)
    dB_vmin = cart_dB[tissue_cart].min() if tissue_cart.any() else 0
    dB_vmax = cart_dB[tissue_cart].max() if tissue_cart.any() else 1

    im3 = axes[1, 0].imshow(cart_dB, cmap='gray', aspect='equal', vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 0].set_title('Target (dB, Cartesian)')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(pred_dB_cart, cmap='gray', aspect='equal', vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 1].set_title('Predicted (dB, Cartesian)')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    residual_cart = pred_dB_cart - cart_dB
    residual_cart[~tissue_cart] = 0
    vabs = max(np.abs(residual_cart[tissue_cart]).max(), 1e-6) if tissue_cart.any() else 1.0
    im5 = axes[1, 2].imshow(residual_cart, cmap='coolwarm', aspect='equal', vmin=-vabs, vmax=vabs)
    axes[1, 2].set_title('Residual (pred - target)')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    # --- Row 2: Loss+gain curves, mu histogram, alpha histogram ---
    ax_loss = axes[2, 0]
    ax_loss.semilogy(metrics['iters'], metrics['mse'], 'b-', label='MSE')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('MSE', color='b')
    ax_loss.tick_params(axis='y', labelcolor='b')
    ax_loss.set_title('Loss & gain_dB convergence')
    ax_loss.grid(True, alpha=0.3)
    ax_gain_twin = ax_loss.twinx()
    ax_gain_twin.plot(metrics['iters'], metrics['gain_dB'], 'r-', alpha=0.7, label='gain_dB')
    ax_gain_twin.set_ylabel('gain_dB', color='r')
    ax_gain_twin.tick_params(axis='y', labelcolor='r')
    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_gain_twin.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    for cls_id, cls_name in TISSUE_CLASSES.items():
        if cls_id == 0:
            continue
        mask = (seg_polar == cls_id) & tissue_pixels & (mu_polar > 0.01)
        if mask.any():
            vals = mu_polar[mask]
            axes[2, 1].hist(vals, bins=50, alpha=0.6, label=f'{cls_name} (n={len(vals)})',
                            color=colors_hist.get(cls_name, 'gray'))
    axes[2, 1].set_xlabel('mu (mm^-1)')
    axes[2, 1].set_ylabel('Count')
    axes[2, 1].set_title('mu distribution by tissue class')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    for cls_id, cls_name in TISSUE_CLASSES.items():
        if cls_id == 0:
            continue
        mask = (seg_polar == cls_id) & tissue_pixels & (alpha_polar > 1e-10)
        if mask.any():
            vals = alpha_polar[mask]
            axes[2, 2].hist(vals, bins=50, alpha=0.6, label=f'{cls_name} (n={len(vals)})',
                            color=colors_hist.get(cls_name, 'gray'))
    axes[2, 2].set_xlabel('alpha')
    axes[2, 2].set_ylabel('Count')
    axes[2, 2].set_title('alpha distribution by tissue class')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    # --- Row 3: Per-tissue residual boxplot, mu summary, alpha+gain summary ---
    residual_polar = pred_dB - polar_target_dB
    residual_data = []
    residual_labels = []
    residual_colors = []
    for cls_id, cls_name in TISSUE_CLASSES.items():
        if cls_id == 0:
            continue
        mask = (seg_polar == cls_id) & tissue_pixels
        if mask.any():
            residual_data.append(residual_polar[mask])
            residual_labels.append(cls_name)
            residual_colors.append(colors_hist.get(cls_name, 'gray'))
    if residual_data:
        bp = axes[3, 0].boxplot(residual_data, labels=residual_labels, patch_artist=True,
                                 showfliers=False, whis=[5, 95])
        for patch, color in zip(bp['boxes'], residual_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    axes[3, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[3, 0].set_ylabel('Residual (dB)')
    axes[3, 0].set_title('Per-tissue residual (pred - target)')
    axes[3, 0].grid(True, alpha=0.3, axis='y')

    summary_mu = [f"Patient: {patient_id}  Frame: {frame_idx}",
                  f"Final MSE: {metrics['mse'][-1]:.4e}",
                  f"gain_dB: {metrics['gain_dB'][-1]:.1f}", "",
                  "mu (mm^-1) per tissue:"]
    for cls_id, cls_name in TISSUE_CLASSES.items():
        if cls_id == 0:
            continue
        mask = (seg_polar == cls_id) & tissue_pixels & (mu_polar > 0.01)
        if mask.any():
            vals = mu_polar[mask]
            summary_mu.append(
                f"  {cls_name:>8s}: {vals.mean():.3f} +/- {vals.std():.3f}"
                f" (med={np.median(vals):.3f}, n={len(vals)})")
        else:
            summary_mu.append(f"  {cls_name:>8s}: no tissue pixels")
    axes[3, 1].axis('off')
    axes[3, 1].text(0.05, 0.95, '\n'.join(summary_mu), transform=axes[3, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[3, 1].set_title('Per-class mu summary')

    summary_alpha = ["Alpha per tissue class:", ""]
    for cls_id, cls_name in TISSUE_CLASSES.items():
        if cls_id == 0:
            continue
        mask = (seg_polar == cls_id) & tissue_pixels & (alpha_polar > 1e-10)
        if mask.any():
            vals = alpha_polar[mask]
            summary_alpha.append(
                f"  {cls_name:>8s}: {vals.mean():.2e} +/- {vals.std():.2e}"
                f" (med={np.median(vals):.2e}, n={len(vals)})")
        else:
            summary_alpha.append(f"  {cls_name:>8s}: no tissue pixels")
    summary_alpha.append("")
    gain_arr = np.array(metrics['gain_dB'])
    summary_alpha.append(f"Final gain_dB: {gain_arr[-1]:.2f}")
    summary_alpha.append(f"Gain range: [{gain_arr.min():.1f}, {gain_arr.max():.1f}]")
    summary_alpha.append(f"Gain drift: {gain_arr[-1] - gain_arr[0]:+.2f} dB")
    axes[3, 2].axis('off')
    axes[3, 2].text(0.05, 0.95, '\n'.join(summary_alpha), transform=axes[3, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[3, 2].set_title('Alpha + gain summary')

    plt.suptitle(f'{patient_id} -- Frame {frame_idx} -- OAC Results (mu + alpha + gain)', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'frame_{frame_idx:03d}_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved results panel: {out_path}")


def save_cartesian_iter_panel(mu_polar, alpha_polar, pred_dB_polar, cart_image,
                              cart_seg_onehot, noise_mask_polar,
                              iteration, mse_val, gain_dB_val, panel_dir):
    """Save a Cartesian-frame iteration panel with mu, alpha, and gain info.

    Layout: 2x3 grid:
        (0,0) Original image + seg    (0,1) mu map + seg contours    (0,2) alpha map + seg contours
        (1,0) Predicted dB            (1,1) text: iter, MSE, gain    (1,2) (empty)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    class_map_cart = np.argmax(cart_seg_onehot, axis=-1)

    # Convert polar to Cartesian
    mu_cart = polar_to_cartesian(mu_polar, fill_value=0.0)
    alpha_cart = polar_to_cartesian(alpha_polar, fill_value=0.0)
    pred_dB_cart = polar_to_cartesian(pred_dB_polar, fill_value=0.0)
    noise_mask_cart = polar_to_cartesian(noise_mask_polar, fill_value=0.0)
    tissue_cart = noise_mask_cart > 0.5

    # (0,0) Original + seg
    axes[0, 0].imshow(cart_image, cmap='gray', aspect='equal')
    overlay = np.zeros((*cart_image.shape, 4))
    for cls_id in [1, 2, 3]:
        mask = class_map_cart == cls_id
        overlay[mask] = TISSUE_COLORS[cls_id]
    axes[0, 0].imshow(overlay, aspect='equal')
    axes[0, 0].set_title('Image + Segmentation')

    # (0,1) mu Cartesian + seg contours
    mu_cart_display = mu_cart.copy()
    mu_cart_display[~tissue_cart] = 0
    mu_vmax = np.percentile(mu_cart_display[mu_cart_display > 0], 99) if (mu_cart_display > 0).any() else 1.0
    im1 = axes[0, 1].imshow(mu_cart_display, cmap='hot', aspect='equal', vmin=0, vmax=mu_vmax)
    for cls_id, color in [(1, 'red'), (2, 'lime'), (3, 'cyan')]:
        mask_cls = (class_map_cart == cls_id).astype(float)
        if mask_cls.any():
            axes[0, 1].contour(mask_cls, levels=[0.5], colors=[color], linewidths=0.8)
    axes[0, 1].set_title(f'mu (mm^-1) -- iter {iteration}')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # (0,2) alpha Cartesian + seg contours
    alpha_cart_display = alpha_cart.copy()
    alpha_cart_display[~tissue_cart] = 0
    alpha_tissue_vals = alpha_cart_display[alpha_cart_display > 0]
    alpha_vmax = np.percentile(alpha_tissue_vals, 99) if len(alpha_tissue_vals) > 0 else 1e-2
    alpha_vmin = np.percentile(alpha_tissue_vals, 1) if len(alpha_tissue_vals) > 0 else 0
    im_a = axes[0, 2].imshow(alpha_cart_display, cmap='viridis', aspect='equal',
                              vmin=alpha_vmin, vmax=alpha_vmax)
    for cls_id, color in [(1, 'red'), (2, 'lime'), (3, 'cyan')]:
        mask_cls = (class_map_cart == cls_id).astype(float)
        if mask_cls.any():
            axes[0, 2].contour(mask_cls, levels=[0.5], colors=[color], linewidths=0.8)
    axes[0, 2].set_title(f'alpha -- iter {iteration}')
    plt.colorbar(im_a, ax=axes[0, 2], fraction=0.046)

    # (1,0) Predicted dB Cartesian
    pred_display = pred_dB_cart.copy()
    pred_display[~tissue_cart] = 0
    im2 = axes[1, 0].imshow(pred_display, cmap='gray', aspect='equal')
    axes[1, 0].set_title('Predicted dB (Cartesian)')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # (1,1) Text info with gain
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.5,
                    f'Iteration: {iteration}\nMSE: {mse_val:.4e}\ngain_dB: {gain_dB_val:.2f}',
                    transform=axes[1, 1].transAxes, fontsize=16,
                    ha='center', va='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (1,2) Empty
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(panel_dir, f'iter_{iteration:05d}.png'), dpi=100)
    plt.close(fig)


def plot_cross_frame_summary(all_frame_stats, frame_indices, out_dir, patient_id):
    """Plot cross-frame summary of per-class mu, alpha, and gain statistics.

    Layout: 2x3 grid
        Row 0: mu bar by frame, alpha bar by frame, gain_dB by frame
        Row 1: mu boxplot (all frames), alpha boxplot (all frames), summary text
    """
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    class_names = ['lumen', 'wall', 'calcium']
    class_colors = ['red', 'green', 'blue']
    x = np.arange(len(frame_indices))
    width = 0.25

    # --- (0,0) Per-class mu bar chart by frame ---
    for i, (cls_name, color) in enumerate(zip(class_names, class_colors)):
        means = [s.get(cls_name, {}).get('mu_mean', s.get(cls_name, {}).get('mean', 0))
                 for s in all_frame_stats]
        stds = [s.get(cls_name, {}).get('mu_std', s.get(cls_name, {}).get('std', 0))
                for s in all_frame_stats]
        axes[0, 0].bar(x + i * width, means, width, yerr=stds, label=cls_name,
                        color=color, alpha=0.7, capsize=3)
    axes[0, 0].set_xlabel('Frame index')
    axes[0, 0].set_ylabel('mu (mm^-1)')
    axes[0, 0].set_title('Per-class mu by frame')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels([str(fi) for fi in frame_indices])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # --- (0,1) Per-class alpha bar chart by frame ---
    for i, (cls_name, color) in enumerate(zip(class_names, class_colors)):
        means = [s.get(cls_name, {}).get('alpha_mean', 0) for s in all_frame_stats]
        stds = [s.get(cls_name, {}).get('alpha_std', 0) for s in all_frame_stats]
        axes[0, 1].bar(x + i * width, means, width, yerr=stds, label=cls_name,
                        color=color, alpha=0.7, capsize=3)
    axes[0, 1].set_xlabel('Frame index')
    axes[0, 1].set_ylabel('alpha')
    axes[0, 1].set_title('Per-class alpha by frame')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels([str(fi) for fi in frame_indices])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].ticklabel_format(axis='y', style='scientific', scilimits=(-2, -2))

    # --- (0,2) gain_dB by frame ---
    gain_values = [s.get('_gain', {}).get('final_dB', 0) for s in all_frame_stats]
    gain_mins = [s.get('_gain', {}).get('min_dB', 0) for s in all_frame_stats]
    gain_maxs = [s.get('_gain', {}).get('max_dB', 0) for s in all_frame_stats]
    axes[0, 2].bar(x, gain_values, 0.5, color='purple', alpha=0.7, label='Final gain_dB')
    # Error bars showing gain range during optimization
    gain_err_low = [g - lo for g, lo in zip(gain_values, gain_mins)]
    gain_err_high = [hi - g for g, hi in zip(gain_values, gain_maxs)]
    axes[0, 2].errorbar(x, gain_values, yerr=[gain_err_low, gain_err_high],
                         fmt='none', ecolor='black', capsize=5, label='Optim range')
    axes[0, 2].set_xlabel('Frame index')
    axes[0, 2].set_ylabel('gain_dB')
    axes[0, 2].set_title('gain_dB by frame')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels([str(fi) for fi in frame_indices])
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # --- (1,0) Aggregated mu boxplot ---
    mu_data = {cn: [] for cn in class_names}
    for stats in all_frame_stats:
        for cn in class_names:
            if cn in stats:
                vals = stats[cn].get('mu_values', stats[cn].get('values', []))
                mu_data[cn].extend(vals)
    bp_data = [mu_data[cn] for cn in class_names if mu_data[cn]]
    bp_labels = [cn for cn in class_names if mu_data[cn]]
    bp_colors_list = [c for cn, c in zip(class_names, class_colors) if mu_data[cn]]
    if bp_data:
        bp = axes[1, 0].boxplot(bp_data, labels=bp_labels, patch_artist=True,
                                 showfliers=False, whis=[5, 95])
        for patch, color in zip(bp['boxes'], bp_colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    axes[1, 0].set_ylabel('mu (mm^-1)')
    axes[1, 0].set_title('Aggregated mu (all frames)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # --- (1,1) Aggregated alpha boxplot ---
    alpha_data = {cn: [] for cn in class_names}
    for stats in all_frame_stats:
        for cn in class_names:
            if cn in stats and 'alpha_values' in stats[cn]:
                alpha_data[cn].extend(stats[cn]['alpha_values'])
    bp_data_a = [alpha_data[cn] for cn in class_names if alpha_data[cn]]
    bp_labels_a = [cn for cn in class_names if alpha_data[cn]]
    bp_colors_a = [c for cn, c in zip(class_names, class_colors) if alpha_data[cn]]
    if bp_data_a:
        bp_a = axes[1, 1].boxplot(bp_data_a, labels=bp_labels_a, patch_artist=True,
                                   showfliers=False, whis=[5, 95])
        for patch, color in zip(bp_a['boxes'], bp_colors_a):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    axes[1, 1].set_ylabel('alpha')
    axes[1, 1].set_title('Aggregated alpha (all frames)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].ticklabel_format(axis='y', style='scientific', scilimits=(-2, -2))

    # --- (1,2) Summary text ---
    summary = [f"Patient: {patient_id}", f"Frames: {frame_indices}", ""]
    for cn in class_names:
        if mu_data[cn]:
            arr = np.array(mu_data[cn])
            summary.append(f"{cn:>8s} mu: {arr.mean():.3f}+/-{arr.std():.3f} mm^-1 (n={len(arr)})")
        if alpha_data[cn]:
            arr = np.array(alpha_data[cn])
            summary.append(f"{'':>8s} alpha: {arr.mean():.2e}+/-{arr.std():.2e} (n={len(arr)})")
        summary.append("")
    if gain_values:
        summary.append(f"gain_dB: {np.mean(gain_values):.2f} +/- {np.std(gain_values):.2f}")
        summary.append(f"  range: [{min(gain_values):.1f}, {max(gain_values):.1f}]")
    axes[1, 2].axis('off')
    axes[1, 2].text(0.05, 0.95, '\n'.join(summary), transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    axes[1, 2].set_title('Aggregated summary')

    plt.suptitle(f'{patient_id} -- Cross-frame Summary (mu + alpha + gain)', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{patient_id}_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cross-frame summary: {out_path}")


# ===================================================================
# OAC optimization for a single polar frame
# ===================================================================

def build_ivoct_config(depth_pixels=DEPTH_PIXELS, num_alines=NUM_ALINES,
                       noise_floor_dB=79.0, axial_pixel_size_mm=0.01932):
    """Build an OmegaConf config for the ILUMIEN OPTIS / C7-XR Dragonfly IV-OCT system.

    Specs from FDA 510(k) K141453, Table 12-3:
      - Swept source laser: 1305 nm ± 55 nm -> center 1305 nm, full bandwidth 110 nm
      - A-scan range in air: 7.0 mm; in contrast: 4.83 mm
      - 250 depth pixels -> 4.83mm / 250 = 0.01932 mm/pixel
      - Axial resolution: 20 um in tissue
      - 500 A-lines/frame (C7-XR), 100 fps, 90 kHz A-scan rate
      - Optical sensitivity: 90 dB minimum

    Beam parameters:
      - NA estimated from spot size: lateral res ~25 um -> NA ~ 0.017
      - Rayleigh range: z_R = pi*w0^2/lambda ~ 1.2mm -> ~62 pixels at 0.01932 mm/px
      - Focal plane: catheter optics focus ~1-2 mm into tissue -> ~50-75 px from center

    Measured values from Gargesha 2015 / Ughi 2016 (C7-XR / DragonFly catheter):
      - z_R = 0.59 mm -> 31 pixels at 0.01932 mm/px
      - z_0 = 1.06 mm from catheter surface -> 55 pixels at 0.01932 mm/px

    Environment variable overrides (for batch scripts):
      - CONFOCAL_ZR_PX:  override rayleigh_range_pixels
      - CONFOCAL_ZFP_PX: override focal_plane_position_pixels
    """
    # Allow env var overrides for batch scripts with different confocal params
    zfp_px = int(os.environ.get('CONFOCAL_ZFP_PX', '62'))
    zr_px = int(os.environ.get('CONFOCAL_ZR_PX', '62'))

    config = OmegaConf.create({
        'forward_model_system': {
            'central_wavelength_nm': 1305,      # FDA K141453: 1305 nm center
            'bandwidth_nm': 110,                 # FDA K141453: ±55 nm -> 110 nm full bandwidth
            'numerical_aperture': 0.017,         # estimated from ~25 um lateral spot
            'depth_samples': depth_pixels,
            'a_lines': num_alines,
            'b_lines': 1,
            'sensitivity_falloff': False,
            'confocal_correction': True,
            'focal_plane_position_pixels': zfp_px,
            'rayleigh_range_pixels': zr_px,
            'noise_floor_dB': noise_floor_dB,
            'lateral_pixel_size_mm': 0.01932,    # angular: 2*pi*r/500 ~ varies, use axial as approx
            'axial_pixel_size_mm': axial_pixel_size_mm,  # 4.83mm / 250 px = 0.01932 mm
            'forward_model_type': 'ss',
        },
        'loss': 'mse',
    })
    return config


def build_opt_config(num_iterations=5000):
    """Build optimization config adapted for IV-OCT.

    mu optimizable range: [0.0, 20.0] mm^-1.  Alpha frozen at 1.0 (0 dB).
    """
    opt_config = OmegaConf.create({
        'optimization': {
            'learning_rate': 1e-3,
            'num_iterations': num_iterations,
            'panel_save_interval': int(os.environ.get('PANEL_INTERVAL', str(max(1, num_iterations // 3)))),
            'loss_function_type': 'mse',
            'loss_scale': 1.0e+3,
            'lr_scheduler': 'cosine',
            'lr_min_factor': 0.01,
            'mu_min_mm_inv': 0.01,
            'mu_max_mm_inv': 20.0,
            'alpha_min': 1.0,
            'alpha_max': 1.0,
            'max_delta_mu_mm': 1.0,
            'mu_tv_weight': 1.0e+3,
            'mu_l1_weight': 1.0e+1,
            'gain_dB': 'auto',
            'gradient_scaling_enabled': False,
            'gradient_scaling_factor': 2.0,
            'gradient_scaling_cap': 50.0,
            'mu_stats_threshold_mm_inv': 0,
        }
    })
    return opt_config


def run_frame_optimization(polar_img_dB, seg_polar, frame_idx,
                           out_dir, patient_id, config, opt_config,
                           noise_floor_dB, device,
                           cart_image=None, cart_seg_onehot=None):
    """Run OAC optimization on a single polar-converted frame.

    Args:
        cart_image: (500, 500) original Cartesian image for Cartesian-frame visualization
        cart_seg_onehot: (500, 500, 4) original Cartesian segmentation

    Returns:
        dict of per-class mu statistics
    """
    os.makedirs(out_dir, exist_ok=True)
    depth_pixels, num_alines = polar_img_dB.shape

    print(f"\n--- Frame {frame_idx} ---")
    print(f"  Polar shape: {polar_img_dB.shape}")
    print(f"  dB range: [{polar_img_dB.min():.1f}, {polar_img_dB.max():.1f}]")

    # Match 7b processing order: N(z) subtract first, then build noise floor mask.
    # For IV-OCT, noise_floor_dB serves as both the subtraction level and the mask threshold.
    I_target_dB_orig = torch.tensor(polar_img_dB, dtype=torch.float32, device=device)
    print(f"  Subtracting noise floor ({noise_floor_dB:.1f} dB) in linear space...")
    I_target_dB = subtract_noise_floor_linear(I_target_dB_orig, noise_floor_dB)
    print(f"  Post-subtraction dB range: [{I_target_dB.min():.1f}, {I_target_dB.max():.1f}]")

    # Build noise floor mask on the subtracted image (same as 7b)
    noise_floor_mask = build_noise_floor_mask(I_target_dB, noise_floor_dB, device)
    nfm_np = noise_floor_mask.cpu().numpy()
    n_masked = int((nfm_np < 0.5).sum())
    n_total = nfm_np.size
    print(f"  Noise floor mask: {n_masked}/{n_total} pixels masked "
          f"({100*n_masked/n_total:.1f}%)")

    # Initialize mu in [0, 10] mm^-1, alpha at 1.0 (frozen in stage 1)
    mu = torch.nn.Parameter(
        torch.empty(depth_pixels, num_alines, device=device).uniform_(0.0, 10.0))
    alpha = torch.nn.Parameter(
        torch.ones(depth_pixels, num_alines, device=device))

    # Create forward model
    forward_model = RasterScanOCTForward(config, device=str(device))

    prune_threshold = 0.05

    # ---- Stage 1: Optimize mu (alpha frozen at 1.0, per-iter gain reopt) ----
    # Captures the essential exponential decay relationship.
    stage1_dir = os.path.join(out_dir, 'stage1_mu')
    os.makedirs(stage1_dir, exist_ok=True)
    print(f"\n  === Stage 1: Optimize mu (alpha=1.0, per-iter gain reopt) ===")
    print(f"  Running optimization ({int(opt_config.optimization.num_iterations)} iters)...")
    t0 = time.time()
    mu_opt, alpha_opt, metrics, seq_steps, seq_imgs, mu_init, alpha_init = optimize(
        forward_model, mu, alpha, I_target_dB, opt_config,
        prune_threshold, device, stage1_dir,
        noise_floor_mask=noise_floor_mask,
        recompute_gain_mse_optimal_each_iter=True)
    elapsed = time.time() - t0
    print(f"  Stage 1 done in {elapsed:.1f}s")
    print(f"  Stage 1 gain_dB = {forward_model.gain_dB.item():.2f} dB")

    # ---- Stage 2: Freeze mu, optimize per-pixel alpha + per-iter scalar gain ----
    # alpha captures per-pixel backscattering variation [1e-10, 1e-2].
    # gain_dB is analytically recomputed each iter (MSE-optimal L_0 scalar).
    stage2_dir = os.path.join(out_dir, 'stage2_alpha')
    os.makedirs(stage2_dir, exist_ok=True)
    print(f"\n  === Stage 2: Optimize alpha + gain (mu frozen) ===")

    # Freeze mu at stage 1 result
    mu_frozen = torch.nn.Parameter(mu_opt.detach().clone())
    mu_frozen.requires_grad_(False)

    # Initialize alpha at 1e-3 (midpoint in log space of [1e-10, 1e-2])
    alpha_stage2 = torch.nn.Parameter(
        torch.full((depth_pixels, num_alines), 1e-3, device=device))

    # Stage 2 config: unfreeze alpha, keep mu bounds same (won't change since frozen)
    num_iters_s2 = int(opt_config.optimization.num_iterations)
    opt_config_s2 = OmegaConf.create(OmegaConf.to_container(opt_config, resolve=True))
    opt_config_s2.optimization.alpha_min = 1e-10
    opt_config_s2.optimization.alpha_max = 1e-2

    # Fresh forward model for stage 2
    forward_model_s2 = RasterScanOCTForward(config, device=str(device))

    print(f"  Running optimization ({num_iters_s2} iters)...")
    t0 = time.time()
    _, alpha_opt, metrics_s2, _, _, _, _ = optimize(
        forward_model_s2, mu_frozen, alpha_stage2, I_target_dB, opt_config_s2,
        prune_threshold, device, stage2_dir,
        noise_floor_mask=noise_floor_mask,
        recompute_gain_mse_optimal_each_iter=True)
    elapsed = time.time() - t0
    print(f"  Stage 2 done in {elapsed:.1f}s")
    print(f"  Stage 2 gain_dB = {forward_model_s2.gain_dB.item():.2f} dB")

    # Use stage 2 forward model for final prediction
    mu_opt = mu_frozen
    forward_model = forward_model_s2

    # Merge metrics (append stage 2 after stage 1)
    last_iter_s1 = metrics['iters'][-1] if metrics['iters'] else 0
    for key in metrics:
        if key == 'iters':
            metrics[key].extend([last_iter_s1 + 1 + v for v in metrics_s2[key]])
        else:
            metrics[key].extend(metrics_s2[key])

    # Extract results
    mu_np = mu_opt.detach().cpu().numpy()
    alpha_np = alpha_opt.detach().cpu().numpy()

    # Final prediction
    with torch.no_grad():
        pred_dB = forward_model(mu_opt, alpha_opt, noise_mask=noise_floor_mask)
    pred_dB_np = pred_dB.cpu().numpy()

    # Save numpy outputs (polar representation)
    np.save(os.path.join(out_dir, 'mu_final_mm.npy'), mu_np)
    np.save(os.path.join(out_dir, 'mu_init_mm.npy'), mu_init.cpu().numpy())
    np.save(os.path.join(out_dir, 'alpha_final.npy'), alpha_np)
    np.save(os.path.join(out_dir, 'seg_polar.npy'), seg_polar)
    np.savez(os.path.join(out_dir, 'metrics.npz'), **{k: np.array(v) for k, v in metrics.items()})

    # Save Cartesian mu map
    mu_cart = polar_to_cartesian(mu_np, fill_value=0.0)
    np.save(os.path.join(out_dir, 'mu_final_cartesian_mm.npy'), mu_cart)

    # Compute per-class statistics for mu, alpha, and gain
    frame_stats = {}
    final_gain_dB = float(metrics['gain_dB'][-1])
    gain_arr = np.array(metrics['gain_dB'])

    print(f"  Per-class mu & alpha:")
    for cls_id, cls_name in TISSUE_CLASSES.items():
        if cls_id == 0:
            continue
        cls_stats = {}

        # mu stats
        mu_mask = (seg_polar == cls_id) & (mu_np > 0.01)
        if mu_mask.any():
            mu_vals = mu_np[mu_mask]
            cls_stats['mu_mean'] = float(mu_vals.mean())
            cls_stats['mu_std'] = float(mu_vals.std())
            cls_stats['mu_median'] = float(np.median(mu_vals))
            cls_stats['mu_n'] = int(len(mu_vals))
            cls_stats['mu_values'] = mu_vals.tolist()
            print(f"    {cls_name:>8s}: mu = {cls_stats['mu_mean']:.3f} +/- {cls_stats['mu_std']:.3f} mm^-1  "
                  f"(median={cls_stats['mu_median']:.3f}, n={cls_stats['mu_n']})")
        else:
            print(f"    {cls_name:>8s}: no mu pixels in segmentation")

        # alpha stats
        alpha_mask = (seg_polar == cls_id) & (alpha_np > 1e-10)
        if alpha_mask.any():
            alpha_vals = alpha_np[alpha_mask]
            cls_stats['alpha_mean'] = float(alpha_vals.mean())
            cls_stats['alpha_std'] = float(alpha_vals.std())
            cls_stats['alpha_median'] = float(np.median(alpha_vals))
            cls_stats['alpha_n'] = int(len(alpha_vals))
            cls_stats['alpha_values'] = alpha_vals.tolist()
            print(f"    {'':>8s}  alpha = {cls_stats['alpha_mean']:.2e} +/- {cls_stats['alpha_std']:.2e}  "
                  f"(median={cls_stats['alpha_median']:.2e}, n={cls_stats['alpha_n']})")

        # Legacy keys for backward compatibility with cross-frame summary
        if 'mu_mean' in cls_stats:
            cls_stats['mean'] = cls_stats['mu_mean']
            cls_stats['std'] = cls_stats['mu_std']
            cls_stats['median'] = cls_stats['mu_median']
            cls_stats['n'] = cls_stats['mu_n']
            cls_stats['values'] = cls_stats['mu_values']

        if cls_stats:
            frame_stats[cls_name] = cls_stats

    # Store gain info in frame_stats
    frame_stats['_gain'] = {
        'final_dB': final_gain_dB,
        'min_dB': float(gain_arr.min()),
        'max_dB': float(gain_arr.max()),
        'drift_dB': float(gain_arr[-1] - gain_arr[0]),
    }
    print(f"  gain_dB: final={final_gain_dB:.2f}, range=[{gain_arr.min():.1f}, {gain_arr.max():.1f}], "
          f"drift={gain_arr[-1] - gain_arr[0]:+.2f}")

    # Plot results in Cartesian frame (mu + alpha + gain)
    plot_mu_results(mu_np, alpha_np, seg_polar, polar_img_dB, pred_dB_np, metrics,
                    frame_idx, out_dir, patient_id, noise_floor_dB,
                    cart_image, cart_seg_onehot)

    # Save Cartesian iter panels (with alpha and gain)
    if cart_image is not None and cart_seg_onehot is not None:
        cart_panel_dir = os.path.join(out_dir, 'iter_panels_cartesian')
        os.makedirs(cart_panel_dir, exist_ok=True)
        save_cartesian_iter_panel(
            mu_np, alpha_np, pred_dB_np, cart_image, cart_seg_onehot,
            nfm_np, int(metrics['iters'][-1]), metrics['mse'][-1],
            float(metrics['gain_dB'][-1]), cart_panel_dir)
        print(f"  Saved Cartesian iter panel: {cart_panel_dir}")

    # Write PSNR (over all pixels)
    with torch.no_grad():
        diff = (pred_dB - I_target_dB) ** 2
        mse = float(diff.mean())
        signal_range = float(I_target_dB.max() - I_target_dB.min())
        psnr = 10 * np.log10(signal_range ** 2 / (mse + 1e-12)) if mse > 0 else float('inf')
    with open(os.path.join(out_dir, 'psnr.txt'), 'w') as f:
        f.write(f"PSNR = {psnr:.2f} dB\nMSE  = {mse:.6e}\n")

    return frame_stats


# ===================================================================
# Utilities
# ===================================================================

def clear_outputs():
    """Remove all files in outputs/patient_arteries/."""
    import shutil
    out_root = os.path.join(PROJECT_ROOT, 'outputs', 'patient_arteries')
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
        print(f"Cleared: {out_root}")
    else:
        print(f"Nothing to clear: {out_root} does not exist")


# ===================================================================
# Main
# ===================================================================

def main():
    # Handle --clear flag
    if '--clear' in sys.argv or os.environ.get('CLEAR_OUTPUTS', '') == '1':
        clear_outputs()
        if '--clear' in sys.argv and len(sys.argv) == 2:
            return  # only clearing, don't run optimization
    patient_id = os.environ.get('PATIENT_ID', '02008Pre')
    num_iters = int(os.environ.get('NUM_ITERS', '15000'))
    db_scale = float(os.environ.get('DB_SCALE', '130.0'))
    noise_floor_dB = float(os.environ.get('NOISE_FLOOR_DB', '15.0'))
    axial_pixel_size_mm = float(os.environ.get('AXIAL_PIXEL_SIZE_MM', '0.01932'))

    print(f"=" * 70)
    print(f"Shockwave Arteries IV-OCT Processing")
    print(f"=" * 70)
    print(f"Patient ID:         {patient_id}")
    print(f"Num iterations:     {num_iters}")
    print(f"dB scale:           {db_scale}")
    print(f"Noise floor (dB):   {noise_floor_dB}")
    print(f"Axial pixel size:   {axial_pixel_size_mm} mm")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:             {device}")

    # Find and load files
    patient_path, seg_path = find_patient_files(patient_id)
    print(f"\nLoading patient data...")
    patient_data = load_nifti(patient_path)
    print(f"  Patient shape: {patient_data.shape}")
    print(f"  Value range: [{patient_data.min():.4f}, {patient_data.max():.4f}]")

    print(f"Loading segmentation...")
    seg_data = load_nifti(seg_path)
    print(f"  Seg shape: {seg_data.shape}")

    num_frames = patient_data.shape[2]
    assert seg_data.shape[2] == num_frames, \
        f"Frame count mismatch: patient={num_frames}, seg={seg_data.shape[2]}"
    print(f"  Num frames: {num_frames}")

    # Frame selection
    frame_indices_str = os.environ.get('FRAME_INDICES', '')
    if frame_indices_str:
        frame_indices = [int(x.strip()) for x in frame_indices_str.split(',')]
    else:
        # Pick 1 frame from the middle
        indices = [int(num_frames * 0.5)]
        frame_indices = indices
    print(f"Frame indices:      {frame_indices}")

    # Output directory
    out_base = os.path.join(PROJECT_ROOT, 'outputs', 'patient_arteries', patient_id)
    os.makedirs(out_base, exist_ok=True)
    print(f"Output directory:   {out_base}")

    # Step 1: Segmentation overview plot
    print(f"\n--- Plotting segmentation overview ---")
    plot_segmentation_overlay(patient_data, seg_data, frame_indices, out_base, patient_id)

    # Step 1b: Noise floor comparison (first frame)
    print(f"\n--- Plotting noise floor comparison ---")
    plot_noise_floor_comparison(patient_data, seg_data, frame_indices[0], db_scale,
                                out_base, patient_id)

    # Step 2: Convert to polar and run optimization per frame
    config = build_ivoct_config(
        depth_pixels=DEPTH_PIXELS, num_alines=NUM_ALINES,
        noise_floor_dB=noise_floor_dB, axial_pixel_size_mm=axial_pixel_size_mm)
    opt_config = build_opt_config(num_iterations=num_iters)

    all_frame_stats = []

    pixel_max = patient_data.max()
    for fi in frame_indices:
        frame = patient_data[:, :, fi]
        seg_frame = seg_data[:, :, fi, :]

        # Convert to polar for optimization
        polar_img = cartesian_to_polar(frame)
        seg_polar = cartesian_seg_to_polar(seg_frame)

        # Convert to dB scale: pixel values [0, max] -> [0, db_scale] dB
        if pixel_max > 0:
            polar_img_dB = polar_img * (db_scale / pixel_max)
        else:
            polar_img_dB = polar_img * db_scale

        print(f"  Polar dB range: [{polar_img_dB.min():.1f}, {polar_img_dB.max():.1f}]")

        frame_out_dir = os.path.join(out_base, f'frame_{fi:03d}')

        stats = run_frame_optimization(
            polar_img_dB, seg_polar, fi,
            frame_out_dir, patient_id, config, opt_config,
            noise_floor_dB, device,
            cart_image=frame, cart_seg_onehot=seg_frame)
        all_frame_stats.append(stats)

    # Step 3: Cross-frame summary
    print(f"\n--- Cross-frame summary ---")
    plot_cross_frame_summary(all_frame_stats, frame_indices, out_base, patient_id)

    # Save aggregated stats (mu + alpha + gain)
    summary_path = os.path.join(out_base, f'{patient_id}_stats.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Patient: {patient_id}\n")
        f.write(f"Frames: {frame_indices}\n")
        f.write(f"dB scale: {db_scale}\n")
        f.write(f"Noise floor: {noise_floor_dB} dB\n")
        f.write(f"Num iterations: {num_iters}\n\n")
        for fi, stats in zip(frame_indices, all_frame_stats):
            f.write(f"Frame {fi}:\n")
            # Gain info
            if '_gain' in stats:
                g = stats['_gain']
                f.write(f"  gain_dB: final={g['final_dB']:.2f}, "
                        f"range=[{g['min_dB']:.1f}, {g['max_dB']:.1f}], "
                        f"drift={g['drift_dB']:+.2f}\n")
            for cls_name in ['lumen', 'wall', 'calcium']:
                if cls_name in stats:
                    s = stats[cls_name]
                    # mu
                    mu_mean = s.get('mu_mean', s.get('mean', None))
                    mu_std = s.get('mu_std', s.get('std', None))
                    mu_med = s.get('mu_median', s.get('median', None))
                    mu_n = s.get('mu_n', s.get('n', None))
                    if mu_mean is not None:
                        f.write(f"  {cls_name:>8s}: mu = {mu_mean:.3f} +/- {mu_std:.3f} mm^-1  "
                                f"(median={mu_med:.3f}, n={mu_n})\n")
                    # alpha
                    if 'alpha_mean' in s:
                        f.write(f"  {'':>8s}  alpha = {s['alpha_mean']:.2e} +/- {s['alpha_std']:.2e}  "
                                f"(median={s['alpha_median']:.2e}, n={s['alpha_n']})\n")
                else:
                    f.write(f"  {cls_name:>8s}: no tissue pixels\n")
            f.write('\n')

        # Aggregate across all frames
        f.write("Aggregated (all frames):\n")
        # Gain summary
        all_gains = [s.get('_gain', {}).get('final_dB', None) for s in all_frame_stats]
        all_gains = [g for g in all_gains if g is not None]
        if all_gains:
            f.write(f"  gain_dB: mean={np.mean(all_gains):.2f} +/- {np.std(all_gains):.2f}, "
                    f"range=[{min(all_gains):.1f}, {max(all_gains):.1f}]\n")
        for cls_name in ['lumen', 'wall', 'calcium']:
            # mu
            all_mu = []
            all_alpha = []
            for stats in all_frame_stats:
                if cls_name in stats:
                    mu_vals = stats[cls_name].get('mu_values', stats[cls_name].get('values', []))
                    all_mu.extend(mu_vals)
                    if 'alpha_values' in stats[cls_name]:
                        all_alpha.extend(stats[cls_name]['alpha_values'])
            if all_mu:
                arr = np.array(all_mu)
                f.write(f"  {cls_name:>8s}: mu = {arr.mean():.3f} +/- {arr.std():.3f} mm^-1  "
                        f"(median={np.median(arr):.3f}, n={len(arr)})\n")
            else:
                f.write(f"  {cls_name:>8s}: no mu pixels\n")
            if all_alpha:
                arr_a = np.array(all_alpha)
                f.write(f"  {'':>8s}  alpha = {arr_a.mean():.2e} +/- {arr_a.std():.2e}  "
                        f"(median={np.median(arr_a):.2e}, n={len(arr_a)})\n")
    print(f"Saved stats: {summary_path}")
    print(f"\nDone!")


if __name__ == '__main__':
    main()
