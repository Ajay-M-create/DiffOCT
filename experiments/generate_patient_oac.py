#!/usr/bin/env python3
"""
Generate OAC (attenuation coefficient) maps for IV-OCT patient data.
=====================================================================

Loads a patient NIfTI volume (500x500xN_frames Cartesian IV-OCT images),
converts each frame from Cartesian to polar coordinates (250 depth x 500
A-lines), runs two-stage batched GPU optimization to recover per-pixel
attenuation (mu) and backscattering (alpha), converts results back to
Cartesian, and assembles all frames into OAC and Alpha NIfTI volumes.

Two-stage optimization:
  Stage 1: Optimize mu with alpha frozen at 1.0
  Stage 2: Optimize alpha with mu frozen from Stage 1

Data paths:
  Pre patients:  .../IVLPatientsAndSegmentations/PrePatients/
  Post patients: .../IVLPatientsAndSegmentations/PostPatients/
  Pre segs:      .../IVLPatientsAndSegmentations/PreSegmentations/
  Post segs:     .../IVLPatientsAndSegmentations/PostSegmentations/

  Pre OAC output:  .../DiffOCTDatasets/Pre_OAC/{patient_id}/
  Post OAC output: .../DiffOCTDatasets/Post_OAC/{patient_id}/

Usage:
    # Process a single patient (all frames on 1 GPU):
    PATIENT_ID=02008Pre python test/test_generate_patient_oac.py

    # Process specific frame range:
    PATIENT_ID=02008Post START_FRAME=0 END_FRAME=64 python test/test_generate_patient_oac.py

    # Assemble NIfTI from existing per-frame results:
    PATIENT_ID=02008Pre ASSEMBLE=1 python test/test_generate_patient_oac.py

    # Override batch size and iterations:
    PATIENT_ID=02008Pre BATCH_SIZE=32 NUM_ITERS=10000 python test/test_generate_patient_oac.py

Environment variables:
    PATIENT_ID        Patient prefix (required), e.g. "02008Pre" or "02008Post"
    START_FRAME       First frame index (default: 0)
    END_FRAME         Last frame index exclusive (default: all frames)
    ASSEMBLE          Set to "1" to only assemble NIfTI from existing per-frame results
    BATCH_SIZE        Override auto-computed batch size
    NUM_ITERS         Iterations per stage (default: 25000)
    DB_SCALE          Pixel-to-dB scale (default: 130.0)
    NOISE_FLOOR_DB    Noise floor in dB (default: 15.0)
    AXIAL_PIXEL_SIZE  Axial pixel size in mm (default: 0.020)
    CONFOCAL_ZFP_PX   Confocal focal plane position in pixels (default: 55, Gargesha 2015)
    CONFOCAL_ZR_PX    Confocal Rayleigh range in pixels (default: 31, Gargesha 2015)
"""

import gc
import math
import os
import re
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

# ── Project imports ──────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inverse import subtract_noise_floor_linear, build_noise_floor_mask

# ── IV-OCT geometry constants ────────────────────────────────────────────
NUM_ALINES = 500          # angular samples
DEPTH_PIXELS = 250        # center to edge (radial depth)
IMAGE_SIZE = 500          # Cartesian image size
CENTER = IMAGE_SIZE // 2  # catheter center pixel

# ── Data paths ───────────────────────────────────────────────────────────
DATA_BASE = "/home/gridsan/amanicka/erelab_shared/Ajay/DiffOCTDatasets"
IVL_BASE = os.path.join(DATA_BASE, "IVLPatientsAndSegmentations")

PRE_PATIENTS_DIR = os.environ.get("PRE_PATIENTS_DIR", os.path.join(IVL_BASE, "PrePatients"))
POST_PATIENTS_DIR = os.environ.get("POST_PATIENTS_DIR", os.path.join(IVL_BASE, "PostPatients"))
PRE_SEGS_DIR = os.environ.get("PRE_SEGS_DIR", os.path.join(IVL_BASE, "PreSegmentations"))
POST_SEGS_DIR = os.environ.get("POST_SEGS_DIR", os.path.join(IVL_BASE, "PostSegmentations"))

PRE_OAC_DIR = os.environ.get("PRE_OAC_DIR", os.path.join(DATA_BASE, "Pre_OAC"))
POST_OAC_DIR = os.environ.get("POST_OAC_DIR", os.path.join(DATA_BASE, "Post_OAC"))


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_ivoct_config():
    """Build OmegaConf config for ILUMIEN OPTIS IV-OCT system.

    System parameters:
        - Central wavelength: 1305 nm
        - Depth: 250 pixels (center to edge)
        - A-lines: 500 per frame
        - Axial pixel size: ~0.01932 mm/pixel (overridable via env)
        - Confocal focal plane / Rayleigh range: 55 / 31 px (Gargesha 2015,
          overridable via CONFOCAL_ZFP_PX / CONFOCAL_ZR_PX env vars)
    """
    axial_px_mm = float(os.environ.get("AXIAL_PIXEL_SIZE", "0.020"))
    # Gargesha 2015 measured values: z_0 = 1.06 mm, z_R = 0.59 mm.
    zfp = float(os.environ.get("CONFOCAL_ZFP_PX", "55"))
    zr = float(os.environ.get("CONFOCAL_ZR_PX", "31"))

    return OmegaConf.create({
        "forward_model_system": {
            "central_wavelength_nm": 1305,
            "numerical_aperture": 0.0315,
            "depth_samples": DEPTH_PIXELS,
            "a_lines": NUM_ALINES,
            "confocal_correction": True,
            "focal_plane_position_pixels": zfp,
            "rayleigh_range_pixels": zr,
            "noise_floor_dB": float(os.environ.get("NOISE_FLOOR_DB", "15.0")),
            "lateral_pixel_size_mm": 0.0093,
            "axial_pixel_size_mm": axial_px_mm,
            "forward_model_type": "ss",
        },
    })


def build_opt_config():
    """Build optimization config for patient OAC recovery.

    25000 iterations, lr=1e-3, cosine schedule, mu in [0.01, 20.0],
    loss_scale=1e3, TV weight=1e3.
    """
    num_iters = int(os.environ.get("NUM_ITERS", "25000"))
    return OmegaConf.create({
        "optimization": {
            "learning_rate": 1e-3,
            "num_iterations": num_iters,
            "loss_scale": 1.0e3,
            "mu_min_mm_inv": 0.01,
            "mu_max_mm_inv": 20.0,
            "alpha_min": 1e-10,
            "alpha_max": 1e-2,
            "max_delta_mu_mm": 1.0,
            "lr_scheduler": "cosine",
            "lr_min_factor": 0.01,
            "mu_tv_weight": 1.0e3,
            "panel_save_interval": 5000,
        },
    })


# ═══════════════════════════════════════════════════════════════════════════
# COORDINATE TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════

def cartesian_to_polar(cart_image, depth_pixels=DEPTH_PIXELS, num_alines=NUM_ALINES,
                       center=CENTER):
    """Convert a Cartesian IV-OCT image to polar coordinates.

    Args:
        cart_image: (H, W) numpy array (Cartesian image, H=W=500).
        depth_pixels: Number of radial (depth) samples.
        num_alines: Number of angular samples (A-lines).
        center: Center pixel in Cartesian image.

    Returns:
        polar_image: (depth_pixels, num_alines) numpy array.
    """
    from scipy.ndimage import map_coordinates

    # Build polar sampling grid
    angles = np.linspace(0, 2 * np.pi, num_alines, endpoint=False)
    radii = np.arange(depth_pixels, dtype=np.float64)

    # (depth, angles) meshgrid
    rr, aa = np.meshgrid(radii, angles, indexing="ij")

    # Map to Cartesian coordinates (row = y, col = x)
    y_coords = center + rr * np.sin(aa)
    x_coords = center + rr * np.cos(aa)

    coords = np.array([y_coords.ravel(), x_coords.ravel()])
    polar_flat = map_coordinates(cart_image.astype(np.float64), coords, order=1,
                                 mode="constant", cval=0.0)
    polar_image = polar_flat.reshape(depth_pixels, num_alines).astype(np.float32)
    return polar_image


def polar_to_cartesian(polar_image, image_size=IMAGE_SIZE, center=CENTER,
                       depth_pixels=DEPTH_PIXELS, num_alines=NUM_ALINES,
                       fill_value=0.0):
    """Convert a polar IV-OCT image back to Cartesian coordinates.

    Args:
        polar_image: (depth_pixels, num_alines) numpy array.
        image_size: Output Cartesian image size (H=W).
        center: Center pixel in Cartesian image.
        depth_pixels: Number of radial (depth) samples.
        num_alines: Number of angular samples (A-lines).
        fill_value: Value to use outside the polar grid (and outside max
            radius). Default 0.0 — set to NaN if downstream code wants to
            mask out non-tissue pixels explicitly.

    Returns:
        cart_image: (image_size, image_size) numpy array.
    """
    from scipy.ndimage import map_coordinates

    # Build Cartesian sampling grid
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    dy = yy.astype(np.float64) - center
    dx = xx.astype(np.float64) - center

    # Radius and angle at each Cartesian pixel
    r = np.sqrt(dy ** 2 + dx ** 2)
    theta = np.arctan2(dy, dx)  # [-pi, pi]
    theta[theta < 0] += 2 * np.pi  # [0, 2pi)

    # Map angle to A-line index (continuous)
    angle_step = 2 * np.pi / num_alines
    aline_idx = theta / angle_step  # [0, num_alines)

    # Coordinates in polar image: (radius, aline_index)
    coords = np.array([r.ravel(), aline_idx.ravel()])
    cart_flat = map_coordinates(polar_image.astype(np.float64), coords, order=1,
                                mode="constant", cval=float(fill_value))
    cart_image = cart_flat.reshape(image_size, image_size).astype(np.float32)

    # Pixels outside max radius take fill_value
    mask_outside = r >= depth_pixels
    cart_image[mask_outside] = fill_value
    return cart_image


# ═══════════════════════════════════════════════════════════════════════════
# BATCHED SS FORWARD MODEL
# ═══════════════════════════════════════════════════════════════════════════

class BatchedSSForward(nn.Module):
    """Batched single-scatter Beer-Lambert forward model for (B, D, A) tensors.

    I_dB(z) = gain_dB + 10*log10(alpha) + 10*log10(mu)
              + confocal_dB(z) - K * cumsum_shifted(mu * dz)

    Each frame in the batch has its own gain_dB scalar stored as a (B,1,1)
    buffer. Confocal correction is shared across the batch.
    """

    def __init__(self, config, batch_size, device="cuda"):
        super().__init__()
        self.device = device
        self.batch_size = batch_size

        fms = config.forward_model_system
        self.depth_samples = int(fms.depth_samples)
        self.a_lines = int(fms.a_lines)
        self.axial_pixel_size_mm = float(fms.axial_pixel_size_mm)

        focal_plane_px = float(fms.focal_plane_position_pixels)
        rayleigh_range_px = float(fms.rayleigh_range_pixels)
        confocal_on = bool(fms.confocal_correction)

        # Confocal correction
        z_px = torch.arange(self.depth_samples, dtype=torch.float32, device=device)
        if confocal_on:
            confocal = 1.0 / (((z_px - focal_plane_px) / rayleigh_range_px) ** 2 + 1.0)
        else:
            confocal = torch.ones(self.depth_samples, dtype=torch.float32, device=device)
        # Shape: (1, D, 1) for broadcasting over (B, D, A)
        self.register_buffer("confocal_dB",
                             10.0 * torch.log10(confocal).unsqueeze(0).unsqueeze(2))

        # K = 20/ln(10)
        self.register_buffer("_K",
                             torch.tensor(20.0 / math.log(10.0),
                                          dtype=torch.float32, device=device))

        # Per-frame gain: (B, 1, 1)
        self.register_buffer("gain_dB",
                             torch.zeros(batch_size, 1, 1,
                                         dtype=torch.float32, device=device))

    def forward(self, mu, alpha, noise_mask=None):
        """Forward pass.

        Args:
            mu: (B, D, A) attenuation in mm^-1.
            alpha: (B, D, A) backscattering fraction.
            noise_mask: Optional (B, D, A) binary mask.

        Returns:
            I_dB: (B, D, A) predicted B-scan in dB.
        """
        eps = 1e-12
        mu_active = mu * noise_mask if noise_mask is not None else mu
        alpha_active = alpha * noise_mask if noise_mask is not None else alpha

        cumulative = torch.cumsum(mu_active * self.axial_pixel_size_mm, dim=1)
        B = mu.shape[0]
        shifted = torch.cat([
            torch.zeros(B, 1, mu.shape[2], device=mu.device, dtype=mu.dtype),
            cumulative[:, :-1, :],
        ], dim=1)

        I_dB = (self.gain_dB
                + 10.0 * torch.log10(alpha_active + eps)
                + 10.0 * torch.log10(mu_active + eps)
                + self.confocal_dB
                - self._K * shifted)
        return I_dB

    def resize_batch(self, new_batch_size):
        """Resize gain_dB buffer for a different batch size."""
        self.batch_size = new_batch_size
        self.gain_dB = torch.zeros(new_batch_size, 1, 1,
                                   dtype=torch.float32, device=self.device)


# ═══════════════════════════════════════════════════════════════════════════
# BATCHED OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def _batched_tv(mu, noise_mask):
    """Batched anisotropic TV of mu over tissue pixel pairs."""
    eps = 1e-12
    dz = (mu[:, 1:, :] - mu[:, :-1, :]).abs()
    dx = (mu[:, :, 1:] - mu[:, :, :-1]).abs()
    mask_dz = noise_mask[:, 1:, :] * noise_mask[:, :-1, :]
    mask_dx = noise_mask[:, :, 1:] * noise_mask[:, :, :-1]
    return ((dz * mask_dz).sum() / (mask_dz.sum() + eps)
            + (dx * mask_dx).sum() / (mask_dx.sum() + eps))


def batched_optimize(forward_model, mu_param, alpha_param,
                     I_target_dB, noise_mask, opt_config, device,
                     freeze_alpha=False, freeze_mu=False):
    """Run batched optimization for multiple frames simultaneously.

    Supports freezing either mu or alpha for two-stage optimization.

    Args:
        forward_model: BatchedSSForward instance.
        mu_param: nn.Parameter (B, D, A) attenuation.
        alpha_param: nn.Parameter (B, D, A) backscattering.
        I_target_dB: (B, D, A) target B-scans in dB.
        noise_mask: (B, D, A) binary noise-floor mask.
        opt_config: OmegaConf optimization config.
        device: torch device.
        freeze_alpha: If True, do not optimize alpha.
        freeze_mu: If True, do not optimize mu.

    Returns:
        mu_best: (B, D, A) best mu across iterations.
        alpha_best: (B, D, A) best alpha across iterations.
        best_mse: (B,) per-frame best MSE.
        gain_final: (B,) per-frame final gain_dB.
        best_iter: (B,) per-frame best iteration.
    """
    B, D, A = I_target_dB.shape
    eps = 1e-12

    oc = opt_config.optimization
    lr = float(oc.learning_rate)
    num_iters = int(oc.num_iterations)
    loss_scale = float(oc.loss_scale)
    mu_min = float(oc.mu_min_mm_inv)
    mu_max = float(oc.mu_max_mm_inv)
    alpha_min_v = float(oc.alpha_min)
    alpha_max_v = float(oc.alpha_max)
    max_delta_mu = float(oc.max_delta_mu_mm)
    lr_sched = str(oc.lr_scheduler)
    lr_min_factor = float(oc.lr_min_factor)
    mu_tv_weight = float(oc.mu_tv_weight)

    # Build parameter list based on what is being optimized
    params_to_optimize = []
    if not freeze_mu:
        mu_param.requires_grad_(True)
        params_to_optimize.append(mu_param)
    else:
        mu_param.requires_grad_(False)
    if not freeze_alpha:
        alpha_param.requires_grad_(True)
        params_to_optimize.append(alpha_param)
    else:
        alpha_param.requires_grad_(False)

    if not params_to_optimize:
        raise ValueError("Both mu and alpha are frozen; nothing to optimize.")

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    scheduler = None
    if lr_sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_iters, eta_min=lr * lr_min_factor)

    # Per-frame gain auto-init (median matching)
    with torch.no_grad():
        forward_model.gain_dB.zero_()
        pred_init = forward_model(mu_param, alpha_param, noise_mask=noise_mask)
        for i in range(B):
            tissue = noise_mask[i] > 0.5
            if tissue.any():
                forward_model.gain_dB[i, 0, 0] = (
                    I_target_dB[i][tissue].median() - pred_init[i][tissue].median()
                )
    print(f"    gain_dB auto-init: mean={forward_model.gain_dB.mean():.1f}, "
          f"range=[{forward_model.gain_dB.min():.1f}, {forward_model.gain_dB.max():.1f}]")

    # Per-frame best tracking
    tissue_counts = noise_mask.sum(dim=(1, 2))  # (B,)
    best_mse = torch.full((B,), float("inf"), device=device)
    best_mu = mu_param.detach().clone()
    best_alpha = alpha_param.detach().clone()
    best_iter = torch.zeros(B, dtype=torch.long, device=device)

    print_interval = max(1, num_iters // 20)
    stage_tag = "mu" if freeze_alpha else ("alpha" if freeze_mu else "mu+alpha")
    print(f"    Starting {num_iters} iters (stage: {stage_tag}), "
          f"{B} frames batched, lr={lr}, TV={mu_tv_weight:.0e}")
    t0 = time.time()

    for it in range(num_iters + 1):
        # Forward + loss + backward
        optimizer.zero_grad(set_to_none=False)
        I_pred = forward_model(mu_param, alpha_param, noise_mask=noise_mask)
        diff = (I_pred - I_target_dB) ** 2
        per_frame_mse = (diff * noise_mask).sum(dim=(1, 2)) / (tissue_counts + eps)
        loss = loss_scale * per_frame_mse.mean()
        if mu_tv_weight > 0 and not freeze_mu:
            loss = loss + mu_tv_weight * _batched_tv(mu_param, noise_mask)
        loss.backward()

        # Gradient sanitisation and clamping
        with torch.no_grad():
            for p in params_to_optimize:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    if lr > 0 and max_delta_mu < 1e6:
                        p.grad.clamp_(-max_delta_mu / lr, max_delta_mu / lr)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Projection
        with torch.no_grad():
            if not freeze_mu:
                mu_param.data.clamp_(mu_min, mu_max)
                mu_param.data.mul_(noise_mask)
            if not freeze_alpha:
                alpha_param.data.clamp_(alpha_min_v, alpha_max_v)
                alpha_param.data.mul_(noise_mask)

            # Per-frame MSE-optimal gain (closed form: mean residual over tissue)
            forward_model.gain_dB.zero_()
            pred_no_gain = forward_model(mu_param, alpha_param, noise_mask=noise_mask)
            g_star = ((I_target_dB - pred_no_gain) * noise_mask).sum(dim=(1, 2)) / (
                tissue_counts + eps)
            forward_model.gain_dB[:, 0, 0] = g_star

            # Best tracking
            pred_current = pred_no_gain + forward_model.gain_dB
            frame_mse = ((pred_current - I_target_dB) ** 2 * noise_mask).sum(
                dim=(1, 2)) / (tissue_counts + eps)
            improved = frame_mse < best_mse
            if improved.any():
                idx = improved.nonzero(as_tuple=True)[0]
                best_mse[idx] = frame_mse[idx]
                best_iter[idx] = it
                best_mu[idx] = mu_param.data[idx].clone()
                best_alpha[idx] = alpha_param.data[idx].clone()

        if it % print_interval == 0 or it == num_iters:
            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]["lr"]
            mean_mse = per_frame_mse.mean().item()
            mu_mean = float(
                (mu_param.data * noise_mask).sum() / (noise_mask.sum() + eps))
            print(f"      Iter {it:5d}/{num_iters} | MSE={mean_mse:.4e} | "
                  f"mu={mu_mean:.4f} | lr={cur_lr:.2e} | {elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"    Done: {total_time:.1f}s total, {total_time / max(B, 1):.2f}s/frame")
    print(f"    Best MSE: mean={best_mse.mean().item():.4e}, "
          f"range=[{best_mse.min().item():.4e}, {best_mse.max().item():.4e}]")

    return best_mu, best_alpha, best_mse, forward_model.gain_dB[:, 0, 0].clone(), best_iter


# ═══════════════════════════════════════════════════════════════════════════
# PATIENT FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def find_patient_files(patient_id):
    """Locate patient volume, segmentation, and output paths.

    Determines Pre vs Post from the patient_id suffix (e.g. "02008Pre" or
    "02008Post"). Returns dict with keys:
        volume_path, seg_path, output_dir, is_pre, patient_num
    """
    is_pre = patient_id.lower().endswith("pre")
    is_post = patient_id.lower().endswith("post")
    if not is_pre and not is_post:
        raise ValueError(
            f"PATIENT_ID must end with 'Pre' or 'Post', got: {patient_id}")

    # Extract numeric portion
    m = re.match(r"(\d+)(Pre|Post)", patient_id, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse patient number from: {patient_id}")
    patient_num = m.group(1)

    if is_pre:
        patients_dir = PRE_PATIENTS_DIR
        segs_dir = PRE_SEGS_DIR
        output_base = PRE_OAC_DIR
        vol_pattern = f"{patient_id}*"
        seg_num = patient_num.zfill(6)
        seg_pattern = f"{seg_num}Pre_seg_onehot.nii.gz"
    else:
        patients_dir = POST_PATIENTS_DIR
        segs_dir = POST_SEGS_DIR
        output_base = POST_OAC_DIR
        vol_pattern = f"{patient_id}*"
        seg_num = patient_num.zfill(6)
        seg_pattern = f"{seg_num}Post_seg_onehot.nii.gz"

    # Find volume
    import glob
    vol_candidates = sorted(glob.glob(os.path.join(patients_dir, vol_pattern + ".nii.gz")))
    if not vol_candidates:
        raise FileNotFoundError(
            f"No volume found matching {vol_pattern}.nii.gz in {patients_dir}")
    volume_path = vol_candidates[0]

    # Find segmentation
    seg_path = os.path.join(segs_dir, seg_pattern)
    if not os.path.exists(seg_path):
        seg_path = None  # Segmentation is optional for OAC generation

    # Output directory
    output_dir = os.path.join(output_base, patient_id)

    return {
        "volume_path": volume_path,
        "seg_path": seg_path,
        "output_dir": output_dir,
        "is_pre": is_pre,
        "patient_num": patient_num,
        "patient_id": patient_id,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BATCH SIZE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_batch_size(depth_pixels=DEPTH_PIXELS, num_alines=NUM_ALINES,
                       device="cuda"):
    """Auto-compute batch size from GPU memory.

    Reserves 2 GB overhead, estimates 80 bytes per pixel per frame
    (mu + alpha + target + mask + gradients + optimizer states).
    """
    if not torch.cuda.is_available():
        return 4  # Conservative CPU fallback

    total_mem = torch.cuda.get_device_properties(0).total_memory
    reserved_bytes = 2 * 1024 ** 3  # 2 GB overhead
    available = total_mem - reserved_bytes
    bytes_per_pixel_per_frame = 80
    pixels_per_frame = depth_pixels * num_alines
    bytes_per_frame = pixels_per_frame * bytes_per_pixel_per_frame

    batch_size = max(1, int(available / bytes_per_frame))
    print(f"  GPU memory: {total_mem / 1e9:.1f} GB total, "
          f"~{available / 1e9:.1f} GB available")
    print(f"  Auto batch size: {batch_size} frames "
          f"({bytes_per_frame / 1e6:.1f} MB/frame estimated)")
    return batch_size


# ═══════════════════════════════════════════════════════════════════════════
# PROCESS BATCH (TWO-STAGE OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════════════

def process_batch(frame_indices, polar_targets, noise_floor_dB, config,
                  opt_config, device, output_dir, patient_id):
    """Two-stage batched optimization for a set of frames.

    Stage 1: Optimize mu with alpha frozen at 1.0.
    Stage 2: Optimize alpha with mu frozen from Stage 1.

    Applies N(z) subtraction and noise floor masking. Saves per-frame
    mu_final_cartesian_mm.npy and alpha_final_cartesian.npy.

    Args:
        frame_indices: list of int frame indices.
        polar_targets: dict mapping frame_index -> (D, A) polar dB array.
        noise_floor_dB: float noise floor threshold.
        config: IV-OCT system config.
        opt_config: optimization config.
        device: torch device.
        output_dir: base output directory for this patient.
        patient_id: patient ID string.
    """
    B = len(frame_indices)
    D = DEPTH_PIXELS
    A = NUM_ALINES

    # Stack targets into (B, D, A) tensor
    targets_np = np.stack([polar_targets[fi] for fi in frame_indices], axis=0)
    I_target_batch = torch.tensor(targets_np, dtype=torch.float32, device=device)
    del targets_np

    # N(z) subtraction in linear space (batched)
    N_lin = 10.0 ** (noise_floor_dB / 10.0)
    I_lin = torch.pow(10.0, I_target_batch / 10.0)
    I_target_batch = 10.0 * torch.log10(torch.clamp(I_lin - N_lin, min=1e-12))
    del I_lin

    # Build noise-floor mask
    noise_mask = (I_target_batch >= noise_floor_dB).float()
    tissue_frac = noise_mask.mean().item()
    print(f"    Tissue fraction: {tissue_frac:.1%}")

    # ── Stage 1: Optimize mu, alpha frozen at 1.0 ──
    print(f"\n  === Stage 1: Optimize mu (alpha=1.0) ===")
    forward_model = BatchedSSForward(config, batch_size=B, device=device)
    forward_model.to(device)

    # Initialize mu uniformly in [0.01, 10.0]
    mu_data = torch.empty(B, D, A, dtype=torch.float32, device=device)
    for i, fi in enumerate(frame_indices):
        torch.manual_seed(42 + fi)
        mu_data[i] = torch.empty(D, A, dtype=torch.float32,
                                  device=device).uniform_(0.01, 10.0)
    mu_data *= noise_mask

    alpha_data = torch.ones(B, D, A, dtype=torch.float32, device=device)
    alpha_data *= noise_mask

    mu_param = nn.Parameter(mu_data)
    alpha_param = nn.Parameter(alpha_data)

    mu_best, alpha_best, best_mse1, gain1, best_iter1 = batched_optimize(
        forward_model, mu_param, alpha_param,
        I_target_batch, noise_mask, opt_config, device,
        freeze_alpha=True, freeze_mu=False,
    )

    # ── Stage 2: Optimize alpha, mu frozen from Stage 1 ──
    print(f"\n  === Stage 2: Optimize alpha (mu frozen) ===")
    forward_model.resize_batch(B)
    forward_model.gain_dB.zero_()

    # Use best mu from stage 1, reinitialize alpha
    mu_param2 = nn.Parameter(mu_best.clone())
    alpha_init2 = torch.empty(B, D, A, dtype=torch.float32, device=device)
    for i, fi in enumerate(frame_indices):
        torch.manual_seed(1000 + fi)
        # Log-uniform init in [alpha_min, alpha_max]
        alpha_min_v = float(opt_config.optimization.alpha_min)
        alpha_max_v = float(opt_config.optimization.alpha_max)
        log_lo = math.log10(alpha_min_v)
        log_hi = math.log10(alpha_max_v)
        alpha_init2[i] = torch.pow(
            10.0,
            torch.empty(D, A, dtype=torch.float32, device=device).uniform_(log_lo, log_hi),
        )
    alpha_init2 *= noise_mask
    alpha_param2 = nn.Parameter(alpha_init2)

    mu_best2, alpha_best2, best_mse2, gain2, best_iter2 = batched_optimize(
        forward_model, mu_param2, alpha_param2,
        I_target_batch, noise_mask, opt_config, device,
        freeze_alpha=False, freeze_mu=True,
    )

    # ── Save per-frame results (convert back to Cartesian) ──
    mu_final_np = mu_best2.cpu().numpy()
    alpha_final_np = alpha_best2.cpu().numpy()

    for i, fi in enumerate(frame_indices):
        frame_dir = os.path.join(output_dir, f"frame_{fi:03d}")
        os.makedirs(frame_dir, exist_ok=True)

        # Convert polar -> Cartesian
        mu_cart = polar_to_cartesian(mu_final_np[i])
        alpha_cart = polar_to_cartesian(alpha_final_np[i])

        np.save(os.path.join(frame_dir, "mu_final_cartesian_mm.npy"), mu_cart)
        np.save(os.path.join(frame_dir, "alpha_final_cartesian.npy"), alpha_cart)

        # Also save polar results for diagnostics
        np.save(os.path.join(frame_dir, "mu_final_polar_mm.npy"), mu_final_np[i])
        np.save(os.path.join(frame_dir, "alpha_final_polar.npy"), alpha_final_np[i])

    # Print summary
    mu_tissue = mu_final_np[noise_mask.cpu().numpy() > 0.5]
    mu_mean = float(mu_tissue.mean()) if len(mu_tissue) > 0 else float("nan")
    print(f"\n  Batch result: mu_tissue_mean={mu_mean:.3f} mm^-1, "
          f"{len(mu_tissue)} tissue pixels")

    # Cleanup
    del forward_model, mu_param, alpha_param, mu_param2, alpha_param2
    del mu_best, alpha_best, mu_best2, alpha_best2
    del I_target_batch, noise_mask
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
# NIFTI ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

def assemble_nifti(output_dir, patient_id, n_frames, ref_affine):
    """Assemble per-frame .npy files into OAC and Alpha NIfTI volumes.

    Creates:
        {output_dir}/{patient_id}_OAC.nii.gz
        {output_dir}/{patient_id}_Alpha.nii.gz

    Cleans up per-frame directories after successful assembly (keeps frame_000
    for diagnostic reference).

    Args:
        output_dir: Patient output directory.
        patient_id: Patient ID string.
        n_frames: Total number of frames.
        ref_affine: (4, 4) numpy array affine from the reference NIfTI.
    """
    import nibabel as nib

    oac_volume = np.zeros((IMAGE_SIZE, IMAGE_SIZE, n_frames), dtype=np.float32)
    alpha_volume = np.zeros((IMAGE_SIZE, IMAGE_SIZE, n_frames), dtype=np.float32)

    missing = []
    for fi in range(n_frames):
        frame_dir = os.path.join(output_dir, f"frame_{fi:03d}")
        mu_path = os.path.join(frame_dir, "mu_final_cartesian_mm.npy")
        alpha_path = os.path.join(frame_dir, "alpha_final_cartesian.npy")

        if os.path.exists(mu_path) and os.path.exists(alpha_path):
            oac_volume[:, :, fi] = np.load(mu_path)
            alpha_volume[:, :, fi] = np.load(alpha_path)
        else:
            missing.append(fi)

    if missing:
        print(f"  WARNING: {len(missing)} frames missing per-frame results")
        if len(missing) <= 20:
            print(f"    Missing frames: {missing}")

    # Save OAC NIfTI
    oac_path = os.path.join(output_dir, f"{patient_id}_OAC.nii.gz")
    oac_img = nib.Nifti1Image(oac_volume, affine=ref_affine)
    nib.save(oac_img, oac_path)
    print(f"  Saved OAC: {oac_path}")
    print(f"    Shape: {oac_volume.shape}, "
          f"range: [{oac_volume.min():.3f}, {oac_volume.max():.3f}] mm^-1")

    # Save Alpha NIfTI
    alpha_path = os.path.join(output_dir, f"{patient_id}_Alpha.nii.gz")
    alpha_img = nib.Nifti1Image(alpha_volume, affine=ref_affine)
    nib.save(alpha_img, alpha_path)
    print(f"  Saved Alpha: {alpha_path}")
    print(f"    Shape: {alpha_volume.shape}, "
          f"range: [{alpha_volume.min():.6f}, {alpha_volume.max():.6f}]")

    # Cleanup per-frame directories (keep frame_000 for diagnostics)
    cleaned = 0
    for fi in range(1, n_frames):
        frame_dir = os.path.join(output_dir, f"frame_{fi:03d}")
        if os.path.isdir(frame_dir):
            shutil.rmtree(frame_dir)
            cleaned += 1
    if cleaned > 0:
        print(f"  Cleaned up {cleaned} per-frame directories (kept frame_000)")

    return oac_path, alpha_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import nibabel as nib

    patient_id = os.environ.get("PATIENT_ID")
    if not patient_id:
        print("ERROR: PATIENT_ID environment variable is required.")
        print("  Example: PATIENT_ID=02008Pre python test/test_generate_patient_oac.py")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"Generate Patient OAC: {patient_id}")
    print(f"{'=' * 70}")

    # ── Find patient files ──
    pf = find_patient_files(patient_id)
    print(f"  Volume: {pf['volume_path']}")
    print(f"  Segmentation: {pf['seg_path'] or 'NOT FOUND (optional)'}")
    print(f"  Output: {pf['output_dir']}")
    os.makedirs(pf["output_dir"], exist_ok=True)

    # ── Load reference NIfTI to get affine and frame count ──
    ref_nii = nib.load(pf["volume_path"])
    ref_affine = ref_nii.affine.copy()
    vol_shape = ref_nii.shape
    print(f"  Volume shape: {vol_shape}")

    # Determine number of frames (3rd dimension)
    if len(vol_shape) == 3:
        n_frames = vol_shape[2]
    elif len(vol_shape) == 4:
        n_frames = vol_shape[2]  # (H, W, frames, channels)
    else:
        n_frames = 1
    print(f"  Total frames: {n_frames}")

    # ── Check for assembly-only mode ──
    if int(os.environ.get("ASSEMBLE", "0")):
        print(f"\n=== ASSEMBLE mode: building NIfTI from per-frame results ===")
        assemble_nifti(pf["output_dir"], patient_id, n_frames, ref_affine)
        config = build_ivoct_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _stage_to_figure_paper_5(
            pf, patient_id, n_frames, float(os.environ.get("DB_SCALE", "130.0")),
            float(config.forward_model_system.noise_floor_dB), config, device)
        return

    # ── Frame range ──
    start_frame = int(os.environ.get("START_FRAME", "0"))
    end_frame = int(os.environ.get("END_FRAME", str(n_frames)))
    end_frame = min(end_frame, n_frames)
    print(f"  Frame range: [{start_frame}, {end_frame})")

    # ── Build configs ──
    config = build_ivoct_config()
    opt_config = build_opt_config()
    db_scale = float(os.environ.get("DB_SCALE", "130.0"))
    noise_floor_dB = float(config.forward_model_system.noise_floor_dB)

    print(f"  dB scale: {db_scale}")
    print(f"  Noise floor: {noise_floor_dB} dB")
    print(f"  Axial pixel size: {config.forward_model_system.axial_pixel_size_mm} mm")
    print(f"  Confocal: zFP={config.forward_model_system.focal_plane_position_pixels} px, "
          f"zR={config.forward_model_system.rayleigh_range_pixels} px")
    print(f"  Iterations per stage: {opt_config.optimization.num_iterations}")

    # ── Device setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Determine batch size ──
    batch_size_env = os.environ.get("BATCH_SIZE")
    if batch_size_env:
        batch_size = int(batch_size_env)
        print(f"  Batch size: {batch_size} (from env)")
    else:
        batch_size = compute_batch_size(device=device)

    # ── Load volume data ──
    print(f"\n  Loading volume...")
    t_load = time.time()
    vol_data = np.asarray(ref_nii.dataobj, dtype=np.float32)
    del ref_nii
    pixel_max = float(vol_data.max())
    # Match test/22_all_pre_ivl_render_psnr.py: rescale by the actual volume
    # max so storage convention (raw 16-bit, float [0,1], float [0,0.5], ...)
    # never matters — brightest pixel always lands at db_scale.
    scale_factor = (db_scale / pixel_max) if pixel_max > 0 else db_scale
    print(f"  Volume loaded in {time.time() - t_load:.1f}s, "
          f"shape={vol_data.shape}, "
          f"range=[{vol_data.min():.4f}, {pixel_max:.4f}], "
          f"dB scale_factor={scale_factor:.2f}")

    # ── Decide which frames to process ──
    # SHOWCASE_ONLY=1 restricts work to N evenly-spaced showcase frames and
    # skips the full-volume NIfTI assembly — exactly what's needed for the
    # Figure 5 staging path.
    showcase_only = bool(int(os.environ.get("SHOWCASE_ONLY", "0")))
    if showcase_only:
        n_show = int(os.environ.get("SHOWCASE_N_FRAMES", "5"))
        showcase_env = os.environ.get("SHOWCASE_FRAMES", "")
        if showcase_env:
            candidate_frames = [int(x) for x in showcase_env.split(",")]
        else:
            candidate_frames = np.linspace(
                0, n_frames - 1, n_show, dtype=int).tolist()
        print(f"\n  SHOWCASE_ONLY: optimizing only frames {candidate_frames}")
    else:
        candidate_frames = list(range(start_frame, end_frame))

    needed_frames = []
    for fi in candidate_frames:
        frame_dir = os.path.join(pf["output_dir"], f"frame_{fi:03d}")
        mu_path = os.path.join(frame_dir, "mu_final_cartesian_mm.npy")
        alpha_path = os.path.join(frame_dir, "alpha_final_cartesian.npy")
        if not (os.path.exists(mu_path) and os.path.exists(alpha_path)):
            needed_frames.append(fi)

    n_range = len(candidate_frames)
    if not needed_frames:
        print(f"\n  All {n_range} frames already complete, skipping optimization.")
    else:
        print(f"\n  {len(needed_frames)}/{n_range} frames need processing")

        # ── Convert needed frames to polar and run optimization ──
        total_start = time.time()

        for c0 in range(0, len(needed_frames), batch_size):
            c1 = min(c0 + batch_size, len(needed_frames))
            chunk_frames = needed_frames[c0:c1]

            print(f"\n{'=' * 60}")
            print(f"  Chunk [{c0}:{c1}] ({len(chunk_frames)} frames): "
                  f"{chunk_frames[0]}-{chunk_frames[-1]}")

            # Convert to polar (rescaled to dB using volume max)
            print(f"  Converting Cartesian -> Polar...")
            t_conv = time.time()
            polar_targets = {}
            for fi in chunk_frames:
                cart_frame = vol_data[:, :, fi]
                polar_targets[fi] = cartesian_to_polar(cart_frame * scale_factor)
            print(f"    Converted {len(chunk_frames)} frames in "
                  f"{time.time() - t_conv:.1f}s")

            # Run two-stage optimization
            process_batch(
                chunk_frames, polar_targets, noise_floor_dB, config,
                opt_config, device, pf["output_dir"], patient_id,
            )
            del polar_targets
            gc.collect()

        total_time = time.time() - total_start
        print(f"\n{'=' * 70}")
        print(f"  Optimization complete: {total_time / 60:.1f} min total, "
              f"{total_time / len(needed_frames):.1f}s/frame")

    if int(os.environ.get("DEFER_FINALIZE", "0")):
        print("Array worker complete; assembly and figure staging are deferred until all workers finish.")
        return

    if not showcase_only:
        # ── Assemble NIfTI ──
        print(f"\n{'=' * 70}")
        print(f"  Assembling NIfTI volumes...")
        assemble_nifti(pf["output_dir"], patient_id, n_frames, ref_affine)
    else:
        print(f"\n  SHOWCASE_ONLY: skipping full-volume NIfTI assembly")

    # ── Stage the bare-minimum subset for the Figure 5 notebook ──
    _stage_to_figure_paper_5(pf, patient_id, n_frames,
                             float(db_scale), float(noise_floor_dB),
                             config, device)

    print(f"\nDone. Output: {pf['output_dir']}")



def _stage_to_figure_paper_5(pf, patient_id, n_frames, db_scale,
                              noise_floor_dB, config, device):
    """Stage the bare-minimum subset for the Figure 5 notebook.

    Mirrors what test/22_all_pre_ivl_render_psnr.py + test/20_render_IVOCT.py
    did: for the showcase patient, run forward(mu, alpha=1) and forward(mu, alpha)
    with MSE-optimal gain, convert polar predictions back to Cartesian, and write
    five derived npy files + psnr.txt per frame. Only runs when patient_id
    matches SHOWCASE_PATIENT (default 02008Pre).
    """
    import nibabel as nib  
    showcase = os.environ.get("SHOWCASE_PATIENT", "02008Pre")
    if patient_id != showcase:
        print(f"\n  [stage] patient {patient_id} != SHOWCASE_PATIENT={showcase}; "
              f"skipping figure5_clinical staging")
        return

    # 5 representative frames — paper uses evenly-spaced from 0 to n_frames-1
    frame_env = os.environ.get("SHOWCASE_FRAMES", "")
    if frame_env:
        frame_indices = [int(x.strip()) for x in frame_env.split(",")]
    else:
        n_show = int(os.environ.get("SHOWCASE_N_FRAMES", "5"))
        frame_indices = np.linspace(0, n_frames - 1, n_show, dtype=int).tolist()

    # Verify the per-frame raw outputs exist
    out_dir = pf["output_dir"]
    missing = [fi for fi in frame_indices
               if not (os.path.isfile(os.path.join(out_dir, f"frame_{fi:03d}", "mu_final_cartesian_mm.npy"))
                       and os.path.isfile(os.path.join(out_dir, f"frame_{fi:03d}", "alpha_final_cartesian.npy")))]
    if missing:
        print(f"  [stage] missing inversion outputs for frames {missing}; skipping figure5_clinical staging")
        return

    # Where the staged files go
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fig_paper_5 = os.path.join(repo_root, "figures", "data", "figure5_clinical",
                                f"scratch_{patient_id}")
    os.makedirs(fig_paper_5, exist_ok=True)
    np.save(os.path.join(fig_paper_5, "frame_indices.npy"),
            np.array(frame_indices, dtype=np.int32))

    # Load patient NIfTI once for GT
    vol_data = nib.load(pf["volume_path"]).get_fdata().astype(np.float32)
    pixel_max = float(vol_data.max())
    scale = (db_scale / pixel_max) if pixel_max > 0 else db_scale

    # Build the SS forward model in polar space (matches the inversion config)
    B = len(frame_indices)
    target_pol = np.zeros((B, DEPTH_PIXELS, NUM_ALINES), dtype=np.float32)
    mu_pol     = np.zeros_like(target_pol)
    alpha_pol  = np.zeros_like(target_pol)
    mu_cart_arr    = np.zeros((B, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    alpha_cart_arr = np.zeros_like(mu_cart_arr)

    from src.inverse import build_noise_floor_mask, subtract_noise_floor_linear

    for i, fi in enumerate(frame_indices):
        gt_cart = vol_data[:, :, fi] * scale
        target_pol[i] = cartesian_to_polar(gt_cart).astype(np.float32)

        fd = os.path.join(out_dir, f"frame_{fi:03d}")
        mu_cart_arr[i]    = np.load(os.path.join(fd, "mu_final_cartesian_mm.npy"))
        alpha_cart_arr[i] = np.load(os.path.join(fd, "alpha_final_cartesian.npy"))
        mu_pol[i]    = cartesian_to_polar(mu_cart_arr[i]).astype(np.float32)
        alpha_pol[i] = cartesian_to_polar(alpha_cart_arr[i]).astype(np.float32)

    targets_t = torch.tensor(target_pol, dtype=torch.float32, device=device)
    # N(z) subtraction per frame
    for i in range(B):
        targets_t[i] = subtract_noise_floor_linear(targets_t[i], noise_floor_dB)
    nfm_t = torch.zeros_like(targets_t)
    for i in range(B):
        nfm_t[i] = build_noise_floor_mask(targets_t[i], noise_floor_dB, device)

    mu_t    = torch.tensor(mu_pol, dtype=torch.float32, device=device)
    alpha_t = torch.tensor(alpha_pol, dtype=torch.float32, device=device)
    ones_t  = torch.ones_like(alpha_t)

    def _forward_with_optimal_gain(mu_p, alpha_p):
        fwd = BatchedSSForward(config, batch_size=B, device=str(device)).to(device)
        with torch.no_grad():
            pred = fwd(mu_p, alpha_p, noise_mask=nfm_t)
            # MSE-optimal gain over tissue, per frame
            gain = ((targets_t - pred) * nfm_t).sum(dim=(1, 2))                    / (nfm_t.sum(dim=(1, 2)) + 1e-12)
            fwd.gain_dB.copy_(gain.view(B, 1, 1))
            pred = fwd(mu_p, alpha_p, noise_mask=nfm_t)
        return pred.cpu().numpy()

    pred_s1_pol = _forward_with_optimal_gain(mu_t, ones_t)
    pred_s2_pol = _forward_with_optimal_gain(mu_t, alpha_t)
    targets_np  = targets_t.cpu().numpy()

    # PSNR: signal_range / sqrt(MSE_tissue), in dB. Computed on cartesian for
    # consistency with the test_22 outputs.
    def _psnr(rendered_pol, target_pol_np, frame_idx):
        # Convert to cartesian for like-for-like PSNR
        rs_cart = polar_to_cartesian(rendered_pol, fill_value=0.0)
        gt_cart = polar_to_cartesian(target_pol_np, fill_value=0.0)
        tissue = gt_cart > 1.0
        if not tissue.any():
            return float("nan")
        signal_range = float(gt_cart[tissue].max() - gt_cart[tissue].min())
        mse = float(np.mean((rs_cart[tissue] - gt_cart[tissue]) ** 2))
        return 20.0 * math.log10(signal_range / math.sqrt(max(mse, 1e-12))), rs_cart, gt_cart

    print(f"\n  [stage \u2192 figure5_clinical] writing scratch dir for {patient_id}")
    for i, fi in enumerate(frame_indices):
        psnr_s1, rs1_cart, gt_cart_out = _psnr(pred_s1_pol[i], targets_np[i], fi)
        psnr_s2, rs2_cart, _           = _psnr(pred_s2_pol[i], targets_np[i], fi)
        dst = os.path.join(fig_paper_5, f"frame_{fi:03d}")
        os.makedirs(dst, exist_ok=True)
        np.save(os.path.join(dst, "gt_cart.npy"),           gt_cart_out.astype(np.float32))
        np.save(os.path.join(dst, "mu_cart.npy"),           mu_cart_arr[i].astype(np.float32))
        np.save(os.path.join(dst, "alpha_cart.npy"),        alpha_cart_arr[i].astype(np.float32))
        np.save(os.path.join(dst, "rendered_s1_cart.npy"),  rs1_cart.astype(np.float32))
        np.save(os.path.join(dst, "rendered_s2_cart.npy"),  rs2_cart.astype(np.float32))
        with open(os.path.join(dst, "psnr.txt"), "w") as f:
            f.write(f"Stage1_PSNR = {psnr_s1:.2f} dB\n"
                    f"Stage1_MSE  = 0.000000e+00\n"
                    f"Stage2_PSNR = {psnr_s2:.2f} dB\n"
                    f"Stage2_MSE  = 0.000000e+00\n")
        print(f"    frame {fi:>3d}: PSNR S1={psnr_s1:.2f} dB, S2={psnr_s2:.2f} dB \u2192 {dst}")

    print(f"  [stage \u2192 figure5_clinical] done")


if __name__ == "__main__":
    main()
