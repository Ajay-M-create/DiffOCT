#!/usr/bin/env python3
"""
22_all_pre_ivl_render_psnr.py  —  Paper-config PSNR across all 128 Pre-IVL pullbacks
====================================================================================
Re-runs the 20_render_IVOCT.py two-stage optimization (mu then alpha) on every
frame of every Pre-IVL pullback with the paper's configuration:

    NOISE_FLOOR_DB          0.0   (no scatterers filtered)
    AXIAL_PIXEL_SIZE_MM     0.01932
    NUM_ITERS_S1            25000 (mu, alpha=1.0)
    NUM_ITERS_S2            25000 (alpha, mu frozen)
    mu_tv_weight            0
    mu_l1_weight            0
    CONFOCAL_ZFP / ZR       55 / 31   (Gargesha 2015 measured values: z_0=1.06 mm, z_R=0.59 mm)

Reuses the batched GPU forward model and optimizer from
`test-shockwave-arteries/shockwave_arteries_batch_gpu.py` via importlib.

Outputs, per patient:
    outputs/22_all_pre_render_psnr/{patient_id}/frame_{NNN}/
        mu_final_cartesian_mm.npy
        alpha_final_cartesian.npy
        psnr.npz   (psnr_s1, psnr_s2, mse_s1, mse_s2, mae_s1, mae_s2,
                    dice_s1, dice_s2, signal_range, n_tissue_px, frame_idx)
    outputs/22_all_pre_render_psnr/{patient_id}/psnr_summary.npz (after assembly)

Modes:
    Worker:     PATIENT_ID=... START_FRAME=... END_FRAME=... python experiments/all_pre_ivl_render_psnr.py
    Assemble:   PATIENT_ID=... ASSEMBLE=1 python experiments/all_pre_ivl_render_psnr.py
    Summary:    SUMMARIZE=1 python experiments/all_pre_ivl_render_psnr.py

SLURM wrapper: scripts/22_AllPreIVLRenderPsnr.sh
"""

import os
import sys
import gc
import glob
import math
import time
import shutil
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import shockwave helpers (sibling directory with hyphens in its name)
SHOCKWAVE_DIR = os.path.join(PROJECT_ROOT, 'test-shockwave-arteries')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sap = _load_module(
    'shockwave_arteries_process',
    os.path.join(SHOCKWAVE_DIR, 'shockwave_arteries_process.py'))
sapb = _load_module(
    'shockwave_arteries_batch_gpu',
    os.path.join(SHOCKWAVE_DIR, 'shockwave_arteries_batch_gpu.py'))

BatchedSSForward = sapb.BatchedSSForward
batched_optimize = sapb.batched_optimize
compute_batch_size = sapb.compute_batch_size

from src.inverse import subtract_noise_floor_linear, build_noise_floor_mask


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATIENT_DIR = os.environ.get(
    'PATIENT_DATA_DIR',
    '/home/gridsan/amanicka/erelab_shared/Ajay/'
    'DiffOCT-IVLDatasetsAndSegsPrePost/Pre_Post')

OUT_BASE_DEFAULT = os.path.join(PROJECT_ROOT, 'outputs', '22_all_pre_render_psnr')

DB_SCALE = float(os.environ.get('DB_SCALE', '130.0'))
NOISE_FLOOR_DB = float(os.environ.get('NOISE_FLOOR_DB', '15.0'))
AXIAL_PIXEL_SIZE_MM = float(os.environ.get('AXIAL_PIXEL_SIZE_MM', '0.01932'))
NUM_ITERS_S1 = int(os.environ.get('NUM_ITERS_S1', '25000'))
NUM_ITERS_S2 = int(os.environ.get('NUM_ITERS_S2', '25000'))
CONFOCAL_ZFP_PX = int(os.environ.get('CONFOCAL_ZFP_PX', '55'))
CONFOCAL_ZR_PX = int(os.environ.get('CONFOCAL_ZR_PX', '31'))

# Dice threshold is fixed at 15 dB to match 20_render_IVOCT._dice_noise_floor
DICE_THRESHOLD_DB = 15.0


# ===================================================================
# Patient discovery
# ===================================================================

def discover_pre_patients(patient_dir=PATIENT_DIR):
    """Return sorted list of Pre patient IDs found in patient_dir."""
    files = sorted(glob.glob(os.path.join(patient_dir, '*Pre*.nii.gz')))
    ids = []
    seen = set()
    for f in files:
        base = os.path.basename(f)
        # Accept {ID}Pre.nii.gz, {ID}Pre_target_Post.nii.gz, etc.
        # Split on first '_' or '.' after 'Pre'
        idx = base.find('Pre')
        if idx < 0:
            continue
        pid = base[:idx + 3]
        if pid not in seen:
            seen.add(pid)
            ids.append(pid)
    return ids


def find_patient_file(patient_id, patient_dir=PATIENT_DIR):
    pattern = os.path.join(patient_dir, f'{patient_id}*.nii.gz')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No patient file matching {pattern}')
    return files[0]


# ===================================================================
# Config builders
# ===================================================================

def build_paper_ivoct_config():
    """IV-OCT forward-model config matching 20_render_IVOCT.py."""
    return OmegaConf.create({
        'forward_model_system': {
            'central_wavelength_nm': 1305,
            'bandwidth_nm': 110,
            'numerical_aperture': 0.017,
            'depth_samples': sap.DEPTH_PIXELS,
            'a_lines': sap.NUM_ALINES,
            'b_lines': 1,
            'sensitivity_falloff': False,
            'confocal_correction': True,
            'focal_plane_position_pixels': CONFOCAL_ZFP_PX,
            'rayleigh_range_pixels': CONFOCAL_ZR_PX,
            'noise_floor_dB': NOISE_FLOOR_DB,
            'lateral_pixel_size_mm': AXIAL_PIXEL_SIZE_MM,
            'axial_pixel_size_mm': AXIAL_PIXEL_SIZE_MM,
            'forward_model_type': 'ss',
        },
        'loss': 'mse',
    })


def build_paper_opt_config(num_iterations, alpha_min=1.0, alpha_max=1.0):
    """Optimization config matching 20_render_IVOCT.py (no TV/L1, fully overfit)."""
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
            'mu_tv_weight': 0.0,   # paper uses none
            'mu_l1_weight': 0.0,   # paper uses none
            'gain_dB': 'auto',
            'gradient_scaling_enabled': False,
            'gradient_scaling_factor': 2.0,
            'gradient_scaling_cap': 50.0,
            'mu_stats_threshold_mm_inv': 0,
        }
    })


# ===================================================================
# PSNR / Dice helpers
# ===================================================================

def _psnr_for_frame(pred_polar, target_polar, tissue_polar):
    """Per-frame PSNR / MSE / MAE computed within the tissue mask (polar)."""
    if not tissue_polar.any():
        return {'mse': 0.0, 'psnr': 0.0, 'mae': 0.0,
                'signal_range': 0.0, 'n_tissue_px': 0}
    tgt = target_polar[tissue_polar]
    pr = pred_polar[tissue_polar]
    d = pr - tgt
    mse = float(np.mean(d * d))
    signal_range = float(tgt.max() - tgt.min())
    psnr = 10.0 * np.log10((signal_range ** 2) / (mse + 1e-12)) if signal_range > 0 else 0.0
    mae = float(np.mean(np.abs(d)))
    return {'mse': mse, 'psnr': psnr, 'mae': mae,
            'signal_range': signal_range,
            'n_tissue_px': int(tissue_polar.sum())}


def _dice_at_threshold(pred_cart, target_cart, thresh_dB=DICE_THRESHOLD_DB):
    A = target_cart > thresh_dB
    B = pred_cart > thresh_dB
    denom = int(A.sum()) + int(B.sum())
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(A, B).sum() / denom)


# ===================================================================
# Per-batch processing
# ===================================================================

def process_batch(frame_indices, patient_data, pixel_max, device, out_base,
                  patient_id, config, opt_config_s1, opt_config_s2,
                  save_mu_alpha=True):
    """Two-stage optimization on a batch of frames; saves mu, alpha, and PSNR."""
    B = len(frame_indices)
    D = sap.DEPTH_PIXELS
    A = sap.NUM_ALINES
    prune_threshold = 0.05

    print(f"\n{'=' * 70}")
    print(f"Batch of {B} frames: {frame_indices[0]}-{frame_indices[-1]}")
    print(f"{'=' * 70}")

    # --- Skip frames already done ---
    frames_to_process = []
    for fi in frame_indices:
        frame_dir = os.path.join(out_base, f'frame_{fi:03d}')
        psnr_out = os.path.join(frame_dir, 'psnr.npz')
        if os.path.exists(psnr_out):
            continue
        frames_to_process.append(fi)

    if not frames_to_process:
        print("  All frames already processed.")
        return

    B_actual = len(frames_to_process)
    print(f"  {B_actual} frames to process ({B - B_actual} already done)")

    # --- Cartesian -> polar -> dB -> N(z) subtraction ---
    polar_imgs_dB = np.zeros((B_actual, D, A), dtype=np.float32)
    cart_targets = np.zeros((B_actual, sap.IMAGE_SIZE, sap.IMAGE_SIZE),
                             dtype=np.float32)

    t_prep = time.time()
    for i, fi in enumerate(frames_to_process):
        frame = patient_data[:, :, fi]
        cart_targets[i] = frame.astype(np.float32)
        polar_img = sap.cartesian_to_polar(frame)
        if pixel_max > 0:
            polar_dB = polar_img * (DB_SCALE / pixel_max)
        else:
            polar_dB = polar_img * DB_SCALE
        polar_imgs_dB[i] = polar_dB.astype(np.float32)
    print(f"  Polar conversion: {time.time() - t_prep:.1f}s")

    targets_orig_gpu = torch.tensor(polar_imgs_dB, dtype=torch.float32, device=device)

    # N(z) subtraction per frame (linear-space; nf=0.0 subtracts 1.0 linear).
    targets_gpu = targets_orig_gpu.clone()
    for i in range(B_actual):
        targets_gpu[i] = subtract_noise_floor_linear(targets_gpu[i], NOISE_FLOOR_DB)
    noise_masks_gpu = torch.zeros_like(targets_gpu)
    for i in range(B_actual):
        noise_masks_gpu[i] = build_noise_floor_mask(targets_gpu[i], NOISE_FLOOR_DB, device)

    # --- Stage 1: optimize mu, alpha=1.0 ---
    print(f"\n  === Stage 1: mu (B={B_actual}, iters={NUM_ITERS_S1}) ===")
    forward_s1 = BatchedSSForward(config, batch_size=B_actual,
                                   device=str(device)).to(device)
    mu = nn.Parameter(torch.empty(B_actual, D, A, device=device).uniform_(0.0, 10.0))
    alpha_s1 = nn.Parameter(torch.ones(B_actual, D, A, device=device))

    t_s1 = time.time()
    mu_opt, _, _ = batched_optimize(
        forward_s1, mu, alpha_s1, targets_gpu, noise_masks_gpu,
        opt_config_s1, prune_threshold, device, recompute_gain=True)
    t_s1_elapsed = time.time() - t_s1
    print(f"  Stage 1 total: {t_s1_elapsed:.1f}s "
          f"({t_s1_elapsed / max(B_actual, 1):.2f}s/frame)")

    # Pred s1 (mu only)
    with torch.no_grad():
        pred_s1 = forward_s1(mu_opt, alpha_s1, noise_mask=noise_masks_gpu)
    pred_s1_np = pred_s1.cpu().numpy()

    mu_frozen_np = mu_opt.detach().cpu().numpy()
    del forward_s1, mu, alpha_s1, pred_s1
    torch.cuda.empty_cache()

    # --- Stage 2: optimize alpha, mu frozen ---
    print(f"\n  === Stage 2: alpha (B={B_actual}, iters={NUM_ITERS_S2}) ===")
    forward_s2 = BatchedSSForward(config, batch_size=B_actual,
                                   device=str(device)).to(device)
    mu_param_s2 = nn.Parameter(torch.tensor(mu_frozen_np, device=device))
    mu_param_s2.requires_grad_(False)
    alpha_s2 = nn.Parameter(torch.full((B_actual, D, A), 1e-3, device=device))

    t_s2 = time.time()
    _, alpha_opt, _ = batched_optimize(
        forward_s2, mu_param_s2, alpha_s2, targets_gpu, noise_masks_gpu,
        opt_config_s2, prune_threshold, device, recompute_gain=True)
    t_s2_elapsed = time.time() - t_s2
    print(f"  Stage 2 total: {t_s2_elapsed:.1f}s "
          f"({t_s2_elapsed / max(B_actual, 1):.2f}s/frame)")

    with torch.no_grad():
        pred_s2 = forward_s2(mu_param_s2, alpha_opt, noise_mask=noise_masks_gpu)
    pred_s2_np = pred_s2.cpu().numpy()
    alpha_opt_np = alpha_opt.detach().cpu().numpy()
    targets_np = targets_gpu.cpu().numpy()
    noise_masks_np = noise_masks_gpu.cpu().numpy()
    del forward_s2, mu_param_s2, alpha_s2, alpha_opt, mu_opt, pred_s2
    del targets_gpu, noise_masks_gpu, targets_orig_gpu
    torch.cuda.empty_cache()

    # --- Per-frame save: mu_cart, alpha_cart, PSNR/Dice ---
    print(f"\n  Saving {B_actual} frames...")
    t_save = time.time()
    for i, fi in enumerate(frames_to_process):
        frame_dir = os.path.join(out_base, f'frame_{fi:03d}')
        os.makedirs(frame_dir, exist_ok=True)

        mu_cart = sap.polar_to_cartesian(mu_frozen_np[i],
                                          fill_value=0.0).astype(np.float32)
        alpha_cart = sap.polar_to_cartesian(alpha_opt_np[i],
                                             fill_value=0.0).astype(np.float32)
        pred_s1_cart = sap.polar_to_cartesian(pred_s1_np[i], fill_value=0.0)
        pred_s2_cart = sap.polar_to_cartesian(pred_s2_np[i], fill_value=0.0)
        target_cart = sap.polar_to_cartesian(targets_np[i], fill_value=0.0)

        if save_mu_alpha:
            np.save(os.path.join(frame_dir, 'mu_final_cartesian_mm.npy'), mu_cart)
            np.save(os.path.join(frame_dir, 'alpha_final_cartesian.npy'), alpha_cart)

        tissue_polar = noise_masks_np[i] > 0.5
        stats_s1 = _psnr_for_frame(pred_s1_np[i], targets_np[i], tissue_polar)
        stats_s2 = _psnr_for_frame(pred_s2_np[i], targets_np[i], tissue_polar)
        dice_s1 = _dice_at_threshold(pred_s1_cart, target_cart, DICE_THRESHOLD_DB)
        dice_s2 = _dice_at_threshold(pred_s2_cart, target_cart, DICE_THRESHOLD_DB)

        np.savez(os.path.join(frame_dir, 'psnr.npz'),
                 psnr_s1=stats_s1['psnr'], psnr_s2=stats_s2['psnr'],
                 mse_s1=stats_s1['mse'], mse_s2=stats_s2['mse'],
                 mae_s1=stats_s1['mae'], mae_s2=stats_s2['mae'],
                 signal_range=stats_s1['signal_range'],
                 n_tissue_px=stats_s1['n_tissue_px'],
                 dice_s1=dice_s1, dice_s2=dice_s2,
                 frame_idx=int(fi))

    print(f"  Save: {time.time() - t_save:.1f}s")


# ===================================================================
# Patient-level worker
# ===================================================================

def run_worker(patient_id, start_frame, end_frame, out_base):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    patient_path = find_patient_file(patient_id)
    print(f"Loading {patient_path}")
    patient_data = sap.load_nifti(patient_path)
    total_frames = patient_data.shape[2]
    pixel_max = float(patient_data.max())
    print(f"  shape={patient_data.shape}  range=[0, {pixel_max:.4f}]  "
          f"total_frames={total_frames}")

    end_frame = min(end_frame, total_frames)
    if start_frame >= end_frame:
        print(f"Nothing to do: start={start_frame} end={end_frame}")
        return

    config = build_paper_ivoct_config()
    opt_config_s1 = build_paper_opt_config(NUM_ITERS_S1, alpha_min=1.0, alpha_max=1.0)
    opt_config_s2 = build_paper_opt_config(NUM_ITERS_S2,
                                            alpha_min=1e-10, alpha_max=1e-2)

    max_batch = compute_batch_size(sap.DEPTH_PIXELS, sap.NUM_ALINES, device)
    frames = list(range(start_frame, end_frame))
    total = len(frames)
    batch_size = min(max_batch, total)
    num_batches = (total + batch_size - 1) // batch_size
    print(f"\nProcessing {total} frames in {num_batches} batch(es) of "
          f"up to {batch_size}")

    t0 = time.time()
    for bi in range(num_batches):
        lo = bi * batch_size
        hi = min(lo + batch_size, total)
        batch_frames = frames[lo:hi]
        print(f"\n{'#' * 70}\n# Batch {bi + 1}/{num_batches}  "
              f"frames {batch_frames[0]}..{batch_frames[-1]}\n{'#' * 70}")
        process_batch(batch_frames, patient_data, pixel_max, device,
                      out_base, patient_id, config, opt_config_s1,
                      opt_config_s2)
        if bi == 0 and device.type == 'cuda':
            peak = torch.cuda.max_memory_allocated(device) / 1e6
            print(f"\n  Peak GPU memory (batch 1): {peak:.0f} MB")
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nWorker complete: {total} frames in {time.time() - t0:.1f}s")


# ===================================================================
# Assembly (per-patient)
# ===================================================================

def run_assembly(patient_id, out_base):
    """Gather per-frame mu, alpha, and PSNR into per-patient NIfTI + npz."""
    import nibabel as nib

    patient_path = find_patient_file(patient_id)
    patient_data = sap.load_nifti(patient_path)
    total_frames = patient_data.shape[2]

    mu_vol = np.zeros((sap.IMAGE_SIZE, sap.IMAGE_SIZE, total_frames),
                       dtype=np.float32)
    alpha_vol = np.zeros_like(mu_vol)

    psnr_s1 = np.full(total_frames, np.nan, dtype=np.float32)
    psnr_s2 = np.full(total_frames, np.nan, dtype=np.float32)
    mse_s1 = np.full(total_frames, np.nan, dtype=np.float32)
    mse_s2 = np.full(total_frames, np.nan, dtype=np.float32)
    mae_s1 = np.full(total_frames, np.nan, dtype=np.float32)
    mae_s2 = np.full(total_frames, np.nan, dtype=np.float32)
    dice_s1 = np.full(total_frames, np.nan, dtype=np.float32)
    dice_s2 = np.full(total_frames, np.nan, dtype=np.float32)
    signal_range = np.full(total_frames, np.nan, dtype=np.float32)
    n_tissue = np.zeros(total_frames, dtype=np.int64)

    missing = []
    for fi in range(total_frames):
        frame_dir = os.path.join(out_base, f'frame_{fi:03d}')
        psnr_path = os.path.join(frame_dir, 'psnr.npz')
        mu_path = os.path.join(frame_dir, 'mu_final_cartesian_mm.npy')
        alpha_path = os.path.join(frame_dir, 'alpha_final_cartesian.npy')
        if not os.path.exists(psnr_path):
            missing.append(fi)
            continue
        with np.load(psnr_path) as d:
            psnr_s1[fi] = d['psnr_s1']; psnr_s2[fi] = d['psnr_s2']
            mse_s1[fi] = d['mse_s1'];   mse_s2[fi] = d['mse_s2']
            mae_s1[fi] = d['mae_s1'];   mae_s2[fi] = d['mae_s2']
            dice_s1[fi] = d['dice_s1']; dice_s2[fi] = d['dice_s2']
            signal_range[fi] = d['signal_range']
            n_tissue[fi] = d['n_tissue_px']
        if os.path.exists(mu_path):
            mu_vol[:, :, fi] = np.load(mu_path)
        if os.path.exists(alpha_path):
            alpha_vol[:, :, fi] = np.load(alpha_path)

    if missing:
        print(f"  WARNING: {len(missing)} frames missing: "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Save per-patient summary
    summary_path = os.path.join(out_base, 'psnr_summary.npz')
    np.savez(summary_path,
             psnr_s1=psnr_s1, psnr_s2=psnr_s2,
             mse_s1=mse_s1, mse_s2=mse_s2,
             mae_s1=mae_s1, mae_s2=mae_s2,
             dice_s1=dice_s1, dice_s2=dice_s2,
             signal_range=signal_range, n_tissue_px=n_tissue,
             frame_indices=np.arange(total_frames),
             missing=np.array(missing, dtype=int))
    print(f"  Saved: {summary_path}")

    # Save mu/alpha volumes as NIfTI (mimic existing OAC/Alpha layout)
    try:
        ref_img = nib.load(patient_path)
        mu_img = nib.Nifti1Image(mu_vol, ref_img.affine, ref_img.header)
        alpha_img = nib.Nifti1Image(alpha_vol, ref_img.affine, ref_img.header)
        nib.save(mu_img, os.path.join(out_base, f'{patient_id}_OAC.nii.gz'))
        nib.save(alpha_img, os.path.join(out_base, f'{patient_id}_Alpha.nii.gz'))
        print(f"  Saved: {patient_id}_OAC.nii.gz, {patient_id}_Alpha.nii.gz")
    except Exception as e:
        print(f"  WARNING: could not save NIfTI volumes: {e}")

    # Valid frames only (not NaN)
    valid = ~np.isnan(psnr_s1)
    n_valid = int(valid.sum())
    if n_valid > 0:
        print(f"\n  PSNR summary ({n_valid}/{total_frames} frames):")
        print(f"    Stage 1 (mu):    mean={np.nanmean(psnr_s1):6.2f}  "
              f"std={np.nanstd(psnr_s1):5.2f}  "
              f"min={np.nanmin(psnr_s1):6.2f}  max={np.nanmax(psnr_s1):6.2f}")
        print(f"    Stage 2 (mu+a):  mean={np.nanmean(psnr_s2):6.2f}  "
              f"std={np.nanstd(psnr_s2):5.2f}  "
              f"min={np.nanmin(psnr_s2):6.2f}  max={np.nanmax(psnr_s2):6.2f}")
        print(f"    Dice s1:         mean={np.nanmean(dice_s1):6.3f}  "
              f"std={np.nanstd(dice_s1):5.3f}")
        print(f"    Dice s2:         mean={np.nanmean(dice_s2):6.3f}  "
              f"std={np.nanstd(dice_s2):5.3f}")


# ===================================================================
# Paper-figure regeneration (for 02008Pre, using test/20_render_IVOCT layout)
# ===================================================================

def _rebuild_paper_figures(patient_id, out_root,
                            frame_indices_src=None,
                            target_dirs=None):
    """Regenerate clinical_src.pdf + supp_fig_alpha_refinement.pdf for `patient_id`
    using the test_22 mu/alpha maps, with the exact same frames and layout as
    experiments/render_ivoct.save_paper_figures.

    Steps:
      1. Read the 12 frame indices from the original 20_render_ivoct run.
      2. For each frame, rebuild the 20_render-compatible per-frame files
         (gt_cart.npy, rendered_s{1,2}_cart.npy, mu_cart.npy, alpha_cart.npy,
         psnr.txt) from the test_22 outputs in a scratch dir.
      3. Call the save_paper_figures() function from 20_render_IVOCT.py on
         that scratch dir — gives pixel-identical layout to the paper.
      4. Copy the two PDFs to test_22 patient dir and (default) also to
         outputs/20_render_ivoct/{patient_id}/.
    """
    t22_patient_dir = os.path.join(out_root, patient_id)
    if not os.path.isdir(t22_patient_dir):
        print(f"  Skipping figure regen: {t22_patient_dir} not found.")
        return

    # Select 8 evenly-spaced example frames from the patient's full range.
    # Override via PAPER_FIG_N_FRAMES (must be even) or PAPER_FIG_FRAME_INDICES
    # (comma-separated list).
    n_frames_fig = int(os.environ.get('PAPER_FIG_N_FRAMES', '5'))

    frame_indices_env = os.environ.get('PAPER_FIG_FRAME_INDICES', '')
    if frame_indices_env:
        frame_indices = [int(x.strip()) for x in frame_indices_env.split(',')]
    else:
        try:
            patient_path = find_patient_file(patient_id)
            total_frames = sap.load_nifti(patient_path).shape[2]
        except Exception as e:
            print(f"  Skipping figure regen: could not load patient ({e})")
            return
        frame_indices = np.linspace(0, total_frames - 1, n_frames_fig,
                                     dtype=int).tolist()
    if frame_indices_src is not None and os.path.exists(frame_indices_src):
        # Optional explicit override (e.g. a saved frame_indices.npy)
        frame_indices = np.load(frame_indices_src).tolist()
    if len(frame_indices) % 2 != 0:
        print(f"  WARNING: expected even number of frames, got {len(frame_indices)}")

    # Check all the needed frames are present in test_22
    for fi in frame_indices:
        fd = os.path.join(t22_patient_dir, f'frame_{fi:03d}')
        if not (os.path.exists(os.path.join(fd, 'mu_final_cartesian_mm.npy')) and
                os.path.exists(os.path.join(fd, 'alpha_final_cartesian.npy')) and
                os.path.exists(os.path.join(fd, 'psnr.npz'))):
            print(f"  Skipping figure regen: missing files for frame {fi}")
            return

    # Import save_paper_figures from experiments/render_ivoct.py
    t20 = _load_module(
        'test_20_render_IVOCT',
        os.path.join(PROJECT_ROOT, 'experiments', 'render_ivoct.py'))

    # Scratch dir populated with 20_render-compatible filenames
    scratch = os.path.join(out_root, '_paper_figs_scratch', patient_id)
    os.makedirs(scratch, exist_ok=True)
    np.save(os.path.join(scratch, 'frame_indices.npy'),
            np.array(frame_indices, dtype=int))

    print(f"\n{'=' * 72}\n  Rebuilding paper figures for {patient_id} "
          f"({len(frame_indices)} frames)\n{'=' * 72}")

    # Load patient data once
    patient_path = find_patient_file(patient_id)
    patient_data = sap.load_nifti(patient_path)
    pixel_max = float(patient_data.max())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = len(frame_indices)
    D = sap.DEPTH_PIXELS
    A = sap.NUM_ALINES

    # Build target polar dB + noise mask per frame (batched)
    target_pol = np.zeros((B, D, A), dtype=np.float32)
    mu_pol = np.zeros((B, D, A), dtype=np.float32)
    alpha_pol = np.zeros((B, D, A), dtype=np.float32)
    psnr_s1_vals = np.zeros(B, dtype=np.float32)
    psnr_s2_vals = np.zeros(B, dtype=np.float32)

    for i, fi in enumerate(frame_indices):
        fd = os.path.join(t22_patient_dir, f'frame_{fi:03d}')
        gt_cart = patient_data[:, :, fi].astype(np.float32)
        pol = sap.cartesian_to_polar(gt_cart)
        scale = (DB_SCALE / pixel_max) if pixel_max > 0 else DB_SCALE
        target_pol[i] = (pol * scale).astype(np.float32)

        mu_cart = np.load(os.path.join(fd, 'mu_final_cartesian_mm.npy'))
        alpha_cart = np.load(os.path.join(fd, 'alpha_final_cartesian.npy'))
        mu_pol[i] = sap.cartesian_to_polar(mu_cart).astype(np.float32)
        alpha_pol[i] = sap.cartesian_to_polar(alpha_cart).astype(np.float32)

        with np.load(os.path.join(fd, 'psnr.npz')) as d:
            psnr_s1_vals[i] = float(d['psnr_s1'])
            psnr_s2_vals[i] = float(d['psnr_s2'])

    targets_gpu = torch.tensor(target_pol, dtype=torch.float32, device=device)
    # N(z) subtraction per frame
    for i in range(B):
        targets_gpu[i] = subtract_noise_floor_linear(targets_gpu[i], NOISE_FLOOR_DB)
    nfm_gpu = torch.zeros_like(targets_gpu)
    for i in range(B):
        nfm_gpu[i] = build_noise_floor_mask(targets_gpu[i], NOISE_FLOOR_DB, device)

    mu_gpu = torch.tensor(mu_pol, dtype=torch.float32, device=device)
    alpha_gpu = torch.tensor(alpha_pol, dtype=torch.float32, device=device)
    alpha_ones = torch.ones_like(alpha_gpu)

    config = build_paper_ivoct_config()

    def _forward_with_optimal_gain(mu_t, alpha_t):
        fwd = BatchedSSForward(config, batch_size=B, device=str(device)).to(device)
        with torch.no_grad():
            pred = fwd(mu_t, alpha_t, noise_mask=nfm_gpu)
            # Analytical per-frame MSE-optimal gain over tissue
            gain = ((targets_gpu - pred) * nfm_gpu).sum(dim=(1, 2)) \
                   / (nfm_gpu.sum(dim=(1, 2)) + 1e-12)
            fwd.gain_dB.copy_(gain)
            pred = fwd(mu_t, alpha_t, noise_mask=nfm_gpu)
        return pred.cpu().numpy()

    pred_s1_pol = _forward_with_optimal_gain(mu_gpu, alpha_ones)
    pred_s2_pol = _forward_with_optimal_gain(mu_gpu, alpha_gpu)
    targets_np = targets_gpu.cpu().numpy()

    # Populate scratch/frame_{fi:03d}/ with 20_render-compatible files
    for i, fi in enumerate(frame_indices):
        dst = os.path.join(scratch, f'frame_{fi:03d}')
        os.makedirs(dst, exist_ok=True)

        gt_cart_out = sap.polar_to_cartesian(targets_np[i], fill_value=0.0)
        rs1_cart = sap.polar_to_cartesian(pred_s1_pol[i], fill_value=0.0)
        rs2_cart = sap.polar_to_cartesian(pred_s2_pol[i], fill_value=0.0)

        src_fd = os.path.join(t22_patient_dir, f'frame_{fi:03d}')
        mu_cart = np.load(os.path.join(src_fd, 'mu_final_cartesian_mm.npy'))
        alpha_cart = np.load(os.path.join(src_fd, 'alpha_final_cartesian.npy'))

        np.save(os.path.join(dst, 'gt_cart.npy'), gt_cart_out.astype(np.float32))
        np.save(os.path.join(dst, 'rendered_s1_cart.npy'), rs1_cart.astype(np.float32))
        np.save(os.path.join(dst, 'rendered_s2_cart.npy'), rs2_cart.astype(np.float32))
        np.save(os.path.join(dst, 'mu_cart.npy'), mu_cart.astype(np.float32))
        np.save(os.path.join(dst, 'alpha_cart.npy'), alpha_cart.astype(np.float32))

        with open(os.path.join(dst, 'psnr.txt'), 'w') as f:
            f.write(f"Stage1_PSNR = {float(psnr_s1_vals[i]):.2f} dB\n"
                    f"Stage1_MSE  = 0.000000e+00\n"
                    f"Stage2_PSNR = {float(psnr_s2_vals[i]):.2f} dB\n"
                    f"Stage2_MSE  = 0.000000e+00\n")

    # Call save_paper_figures from experiments/render_ivoct.py
    t20.save_paper_figures(scratch, frame_indices, patient_id)

    # Copy the two PDFs only to the test_22 patient output dir.
    # (paper/Figs/ is already updated by save_paper_figures above.)
    default_targets = [t22_patient_dir]
    targets = target_dirs if target_dirs is not None else default_targets
    for tgt in targets:
        os.makedirs(tgt, exist_ok=True)
        for fname in ('clinical_src.pdf',
                      'supp_fig_alpha_refinement.pdf',
                      'supp_fig_alpha_refinement_rows.pdf'):
            src = os.path.join(scratch, fname)
            dst = os.path.join(tgt, fname)
            if os.path.exists(src):
                shutil.copyfile(src, dst)
                print(f"  Wrote: {dst}")


# ===================================================================
# Global summary across patients
# ===================================================================

def run_global_summary(out_root):
    """Aggregate per-patient psnr_summary.npz across all 128 Pre patients."""
    patient_dirs = sorted(
        d for d in glob.glob(os.path.join(out_root, '*'))
        if os.path.isdir(d) and os.path.exists(
            os.path.join(d, 'psnr_summary.npz')))

    if not patient_dirs:
        print(f"No per-patient summaries under {out_root}")
        return

    all_s1, all_s2 = [], []
    all_mse_s1, all_mse_s2 = [], []
    all_mae_s1, all_mae_s2 = [], []
    all_dice_s1, all_dice_s2 = [], []
    per_patient_mean_s1, per_patient_mean_s2 = [], []
    per_patient_n = []
    patient_ids = []

    for d in patient_dirs:
        pid = os.path.basename(d)
        with np.load(os.path.join(d, 'psnr_summary.npz')) as s:
            s1 = s['psnr_s1']; s2 = s['psnr_s2']
            mse1 = s['mse_s1']; mse2 = s['mse_s2']
            mae1 = s['mae_s1']; mae2 = s['mae_s2']
            d1 = s['dice_s1']; d2 = s['dice_s2']
        valid = ~np.isnan(s1)
        if not valid.any():
            continue
        all_s1.append(s1[valid]);     all_s2.append(s2[valid])
        all_mse_s1.append(mse1[valid]); all_mse_s2.append(mse2[valid])
        all_mae_s1.append(mae1[valid]); all_mae_s2.append(mae2[valid])
        all_dice_s1.append(d1[valid]);  all_dice_s2.append(d2[valid])
        per_patient_mean_s1.append(float(np.nanmean(s1)))
        per_patient_mean_s2.append(float(np.nanmean(s2)))
        per_patient_n.append(int(valid.sum()))
        patient_ids.append(pid)

    if not all_s1:
        print("No valid PSNR data found.")
        return

    all_s1 = np.concatenate(all_s1)
    all_s2 = np.concatenate(all_s2)
    all_mse_s1 = np.concatenate(all_mse_s1)
    all_mse_s2 = np.concatenate(all_mse_s2)
    all_mae_s1 = np.concatenate(all_mae_s1)
    all_mae_s2 = np.concatenate(all_mae_s2)
    all_dice_s1 = np.concatenate(all_dice_s1)
    all_dice_s2 = np.concatenate(all_dice_s2)

    n_patients = len(patient_ids)
    n_frames = len(all_s1)

    print("\n" + "=" * 72)
    print(f"  Pre-IVL rendering quality — all frames × all patients")
    print("=" * 72)
    print(f"  Patients:       {n_patients}")
    print(f"  Frames total:   {n_frames}")
    print(f"  NOISE_FLOOR_DB: {NOISE_FLOOR_DB}")
    print(f"  AXIAL_PIX_MM:   {AXIAL_PIXEL_SIZE_MM}")
    print(f"  NUM_ITERS s1:   {NUM_ITERS_S1}")
    print(f"  NUM_ITERS s2:   {NUM_ITERS_S2}")
    print("-" * 72)

    def _fmt(name, arr, unit=''):
        print(f"  {name:<25s}  mean={np.mean(arr):8.3f}  std={np.std(arr):7.3f}  "
              f"min={np.min(arr):8.3f}  max={np.max(arr):8.3f}{unit}")

    _fmt('PSNR stage 1 (mu)', all_s1, ' dB')
    _fmt('PSNR stage 2 (mu+a)', all_s2, ' dB')
    _fmt('MSE  stage 1', all_mse_s1)
    _fmt('MSE  stage 2', all_mse_s2)
    _fmt('MAE  stage 1', all_mae_s1, ' dB')
    _fmt('MAE  stage 2', all_mae_s2, ' dB')
    _fmt('Dice stage 1', all_dice_s1)
    _fmt('Dice stage 2', all_dice_s2)
    print("-" * 72)

    pp1 = np.array(per_patient_mean_s1)
    pp2 = np.array(per_patient_mean_s2)
    print(f"  Per-patient mean-PSNR stage 1: "
          f"mean={pp1.mean():.2f} std={pp1.std():.2f}")
    print(f"  Per-patient mean-PSNR stage 2: "
          f"mean={pp2.mean():.2f} std={pp2.std():.2f}")
    print("=" * 72)

    out_csv = os.path.join(out_root, 'summary_all_patients.csv')
    with open(out_csv, 'w') as f:
        f.write('patient_id,n_frames,psnr_s1_mean,psnr_s2_mean\n')
        for pid, n, m1, m2 in zip(patient_ids, per_patient_n,
                                   per_patient_mean_s1, per_patient_mean_s2):
            f.write(f'{pid},{n},{m1:.4f},{m2:.4f}\n')
    print(f"Wrote per-patient CSV: {out_csv}")

    out_npz = os.path.join(out_root, 'summary_all_patients.npz')
    np.savez(out_npz,
             all_psnr_s1=all_s1, all_psnr_s2=all_s2,
             all_mse_s1=all_mse_s1, all_mse_s2=all_mse_s2,
             all_mae_s1=all_mae_s1, all_mae_s2=all_mae_s2,
             all_dice_s1=all_dice_s1, all_dice_s2=all_dice_s2,
             per_patient_mean_s1=pp1, per_patient_mean_s2=pp2,
             per_patient_n=np.array(per_patient_n),
             patient_ids=np.array(patient_ids))
    print(f"Wrote concatenated npz:  {out_npz}")

    # ---- Paper figures (clinical_src.pdf + supp_fig_alpha_refinement.pdf) ----
    if os.environ.get('SKIP_PAPER_FIGS', '') != '1':
        fig_patient = os.environ.get('PAPER_FIG_PATIENT', '02008Pre')
        try:
            _rebuild_paper_figures(fig_patient, out_root)
        except Exception as e:
            print(f"  WARNING: paper-figure regeneration failed: {e}")


# ===================================================================
# Main
# ===================================================================

def main():
    summarize = os.environ.get('SUMMARIZE', '') == '1'
    assemble = os.environ.get('ASSEMBLE', '') == '1'
    out_root = os.environ.get('OUT_ROOT', OUT_BASE_DEFAULT)
    os.makedirs(out_root, exist_ok=True)

    print("=" * 72)
    print(" 22_all_pre_ivl_render_psnr.py")
    print("=" * 72)
    print(f" Mode:           {'SUMMARIZE' if summarize else ('ASSEMBLE' if assemble else 'WORKER')}")
    print(f" Out root:       {out_root}")
    print(f" Patient dir:    {PATIENT_DIR}")
    print(f" NOISE_FLOOR_DB: {NOISE_FLOOR_DB}")
    print(f" AXIAL_PIX_MM:   {AXIAL_PIXEL_SIZE_MM}")
    print(f" NUM_ITERS s1:   {NUM_ITERS_S1}")
    print(f" NUM_ITERS s2:   {NUM_ITERS_S2}")
    print("=" * 72)

    if summarize:
        run_global_summary(out_root)
        return

    patient_id = os.environ.get('PATIENT_ID')
    if not patient_id:
        print("ERROR: PATIENT_ID must be set (or SUMMARIZE=1).")
        sys.exit(1)

    patient_out = os.path.join(out_root, patient_id)
    os.makedirs(patient_out, exist_ok=True)

    if assemble:
        run_assembly(patient_id, patient_out)
        return

    start_frame = int(os.environ.get('START_FRAME', '0'))
    end_frame_env = os.environ.get('END_FRAME', '')
    end_frame = int(end_frame_env) if end_frame_env else 10**9
    run_worker(patient_id, start_frame, end_frame, patient_out)


if __name__ == '__main__':
    main()
