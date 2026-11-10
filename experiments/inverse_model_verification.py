#!/usr/bin/env python3
"""
inverse_model_verification.py — synthetic mu recovery sweep (Figure 2).

Generates 500 synthetic OCT B-scans from known random-pixel mu maps, scrambles
the mu values, then runs the inverse optimizer to recover them. Reports per-
phantom and aggregate error statistics.

Output: outputs/inverse_model_verification/phantom_NNNN/

Usage:
    python experiments/inverse_model_verification.py                 # all 500
    START_INDEX=0 END_INDEX=50 python experiments/inverse_model_verification.py
    SUMMARIZE=1 python experiments/inverse_model_verification.py     # build summary + stage
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── path setup ──────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.forward import create_forward_model
from src.inverse import optimize, build_noise_floor_mask

# ── constants ───────────────────────────────────────────────────────────────
OUT_ROOT = os.path.join(PHANTOM_ROOT, "outputs", "inverse_model_verification")
N_PHANTOMS = 500
DEPTH = 500
A_LINES = 500
TISSUE_START = 230      # tissue band start pixel (matches real IntMask)
TISSUE_END = 270        # tissue band end pixel (~40px, matches real phantoms)
MU_AIR = 1e-4           # background mu (mm^-1)
NUM_ITERS = 30000       # optimization iterations per phantom
PANEL_INTERVAL = 5000   # save diagnostic panels less frequently


def make_random_pixel_phantom(idx, rng):
    """Random per-pixel mu within the tissue band, smoothed via low-res zoom."""
    mu_map = np.full((DEPTH, A_LINES), MU_AIR, dtype=np.float32)
    low_res = rng.uniform(0.0, 10.0, size=(20, 50)).astype(np.float32)
    from scipy.ndimage import zoom
    tissue_height = TISSUE_END - TISSUE_START
    upsampled = zoom(low_res, (tissue_height / 20, A_LINES / 50), order=1)
    mu_map[TISSUE_START:TISSUE_END, :] = upsampled[:tissue_height, :A_LINES]
    alpha_val = float(rng.uniform(1e-10, 1e-2))
    alpha_map = np.full((DEPTH, A_LINES), alpha_val, dtype=np.float32)
    meta = {
        'index': idx, 'type': 'random_pixel',
        'alpha_true': alpha_val,
        'mu_tissue_mean': float(mu_map[TISSUE_START:TISSUE_END, :].mean()),
        'mu_tissue_std':  float(mu_map[TISSUE_START:TISSUE_END, :].std()),
    }
    return mu_map, alpha_map, meta


# ═══════════════════════════════════════════════════════════════════════════
# Core: generate synthetic target and run inverse
# ═══════════════════════════════════════════════════════════════════════════

def run_single_phantom(idx, device):
    """Generate one random-pixel phantom, scramble mu, recover, report error."""
    run_dir = os.path.join(OUT_ROOT, f"phantom_{idx:04d}")
    results_file = os.path.join(run_dir, "results.json")

    snapshots_file = os.path.join(run_dir, "snapshots.npz")
    if os.path.exists(results_file) and os.path.exists(snapshots_file):
        print(f"[#{idx}] Already done, skipping.")
        return

    os.makedirs(run_dir, exist_ok=True)
    rng = np.random.default_rng(seed=30000 + idx)  # fixed seed family

    mu_gt_np, alpha_gt_np, meta = make_random_pixel_phantom(idx, rng)

    config = OmegaConf.load(os.path.join(PHANTOM_ROOT, 'configs', 'config.yaml'))
    config.forward_model_system.depth_samples = DEPTH
    config.forward_model_system.a_lines = A_LINES
    config.forward_model_system.forward_model_type = 'ss'

    forward_model = create_forward_model(config, device=device)

    mu_gt = torch.tensor(mu_gt_np, dtype=torch.float32, device=device)
    alpha_gt = torch.tensor(alpha_gt_np, dtype=torch.float32, device=device)

    # Auto-init gain so tissue median sits 15 dB above noise floor; freeze it.
    noise_floor_dB = float(config.forward_model_system.noise_floor_dB)
    with torch.no_grad():
        forward_model.gain_dB.fill_(0.0)
        I_raw = forward_model(mu_gt, alpha_gt)
        tissue_mask_t = (mu_gt > MU_AIR * 10)
        if tissue_mask_t.any():
            raw_tissue_median = I_raw[tissue_mask_t].median().item()
        else:
            raw_tissue_median = I_raw.median().item()
        gain_dB_val = (noise_floor_dB + 15.0) - raw_tissue_median
        forward_model.gain_dB.fill_(gain_dB_val)
        I_target_dB = forward_model(mu_gt, alpha_gt)
    print(f"  gain_dB = {gain_dB_val:.1f} (auto-init, frozen for inverse)")

    noise_mask = build_noise_floor_mask(I_target_dB, noise_floor_dB, device)

    np.save(os.path.join(run_dir, 'mu_gt_mm.npy'), mu_gt_np)
    np.save(os.path.join(run_dir, 'alpha_gt.npy'), alpha_gt_np)
    np.save(os.path.join(run_dir, 'target_dB.npy'), I_target_dB.cpu().numpy())

    is_showcase = idx in (0, 100, 199)

    tissue_mask_np = (mu_gt_np > MU_AIR * 10).astype(np.float32)
    mu_init_np = np.full((DEPTH, A_LINES), MU_AIR, dtype=np.float32)
    mu_scrambled = rng.uniform(0.0, 10.0, size=(DEPTH, A_LINES)).astype(np.float32)
    mu_init_np = mu_scrambled * tissue_mask_np + MU_AIR * (1.0 - tissue_mask_np)

    if is_showcase:
        np.save(os.path.join(run_dir, 'mu_init_mm.npy'), mu_init_np)

    mu = torch.nn.Parameter(
        torch.tensor(mu_init_np, dtype=torch.float32, device=device))
    # Freeze alpha at ground truth: only mu is being verified.
    alpha = torch.nn.Parameter(
        torch.tensor(alpha_gt_np, dtype=torch.float32, device=device),
        requires_grad=False)

    opt_config = OmegaConf.create({
        'optimization': {
            'learning_rate': 1e-3,
            'num_iterations': NUM_ITERS,
            'loss_scale': 1e3,
            'mu_min_mm_inv': 1e-2,
            'mu_max_mm_inv': 20.0,
            'alpha_min': 1e-10,
            'alpha_max': 1e-2,
            'max_delta_mu_mm': 1.0,
            'panel_save_interval': PANEL_INTERVAL,
            'lr_scheduler': 'cosine',
            'lr_min_factor': 0.01,
            'mu_tv_weight': 0.0,
            'gain_dB': gain_dB_val,
        }
    })

    print(f"\n{'='*70}\n[Phantom {idx}] random_pixel\n{'='*70}")

    # Snapshot iterations for the convergence filmstrip (Fig 2 panels p–t).
    _snap_iters = np.unique(np.geomspace(1, NUM_ITERS, num=120, dtype=int))
    _filmstrip_iters = np.arange(0, NUM_ITERS + 1, 2000)
    _snap_iters = np.sort(np.unique(np.concatenate([[0], _snap_iters, _filmstrip_iters])))
    snap_set = set(_snap_iters.tolist())
    tissue_bool = tissue_mask_np > 0.5
    snap_buffer = {}

    def _mu_snapshot_cb(it, mu_tensor):
        if it in snap_set:
            snap_buffer[it] = mu_tensor.detach().cpu().numpy()[tissue_bool].astype(np.float32)

    t0 = time.time()
    mu_opt, alpha_opt, metrics, *_ = optimize(
        forward_model, mu, alpha, I_target_dB, opt_config,
        prune_threshold=0.0, device=device, out_dir=run_dir,
        noise_floor_mask=noise_mask,
        recompute_gain_mse_optimal_each_iter=False,
        mu_snapshot_callback=_mu_snapshot_cb,
    )
    elapsed = time.time() - t0

    np.savez(os.path.join(run_dir, 'metrics.npz'),
             **{k: np.array(v) for k, v in metrics.items()})

    if snap_buffer:
        snap_iters_sorted = sorted(snap_buffer.keys())
        snap_data = np.stack([snap_buffer[it] for it in snap_iters_sorted])
        np.savez(os.path.join(run_dir, 'snapshots.npz'),
                 iters=np.array(snap_iters_sorted, dtype=np.int32),
                 tissue_gt=mu_gt_np[tissue_bool].astype(np.float32),
                 tissue_mu=snap_data)

    mu_recovered_np = mu_opt.detach().cpu().numpy()
    np.save(os.path.join(run_dir, 'mu_recovered_mm.npy'), mu_recovered_np)
    np.save(os.path.join(run_dir, 'alpha_recovered.npy'),
            alpha_opt.detach().cpu().numpy())

    mu_gt_tissue  = mu_gt_np[tissue_bool]
    mu_rec_tissue = mu_recovered_np[tissue_bool]
    abs_err = np.abs(mu_rec_tissue - mu_gt_tissue)
    rel_err = abs_err / (mu_gt_tissue + 1e-12)

    results = {
        **meta,
        'gain_dB_frozen': float(gain_dB_val),
        'mu_gt_tissue_mean':       float(mu_gt_tissue.mean()),
        'mu_recovered_tissue_mean': float(mu_rec_tissue.mean()),
        'mae':              float(abs_err.mean()),
        'rmse':             float(np.sqrt((abs_err ** 2).mean())),
        'mape':             float(rel_err.mean() * 100),
        'median_abs_error': float(np.median(abs_err)),
        'median_rel_error': float(np.median(rel_err) * 100),
        'max_abs_error':    float(abs_err.max()),
        'final_mse_dB':     float(metrics['mse'][-1]),
        'best_mse_dB':      float(min(metrics['mse'])),
        'elapsed_s':        float(elapsed),
        'n_tissue_pixels':  int(tissue_bool.sum()),
    }
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    _save_comparison_panel(mu_gt_np, mu_recovered_np, alpha_gt_np,
                           alpha_opt.detach().cpu().numpy(),
                           tissue_mask_np, results, run_dir)

    print(f"  => MAPE={results['mape']:.1f}%, MAE={results['mae']:.3f} mm^-1, "
          f"RMSE={results['rmse']:.3f} mm^-1, time={elapsed:.1f}s")
    return results


def _save_comparison_panel(mu_gt, mu_rec, alpha_gt, alpha_rec,
                           tissue_mask, results, out_dir):
    """2×3 diagnostic figure: GT vs recovered mu + scatter + error histogram."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    tissue_bool = tissue_mask > 0.5
    vmax = max(mu_gt[tissue_bool].max(), mu_rec[tissue_bool].max()) * 1.1

    im0 = axes[0, 0].imshow(mu_gt, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
    axes[0, 0].set_title('Ground Truth $\\mu_t$ (mm$^{-1}$)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(mu_rec, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
    axes[0, 1].set_title('Recovered $\\mu_t$ (mm$^{-1}$)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    err_map = np.abs(mu_rec - mu_gt) * tissue_mask
    im2 = axes[0, 2].imshow(err_map, cmap='hot', aspect='auto', vmin=0)
    axes[0, 2].set_title(f'|Error| (MAE={results["mae"]:.3f})')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    gt_flat  = mu_gt[tissue_bool].ravel()
    rec_flat = mu_rec[tissue_bool].ravel()
    n_pts = min(len(gt_flat), 5000)
    pick = np.random.choice(len(gt_flat), n_pts, replace=False)
    axes[1, 0].scatter(gt_flat[pick], rec_flat[pick], alpha=0.1, s=1, c='blue')
    lims = [0, max(gt_flat.max(), rec_flat.max()) * 1.1]
    axes[1, 0].plot(lims, lims, 'r--', linewidth=1, label='y=x')
    axes[1, 0].set_xlim(lims); axes[1, 0].set_ylim(lims)
    axes[1, 0].set_xlabel('Ground Truth $\\mu_t$')
    axes[1, 0].set_ylabel('Recovered $\\mu_t$')
    axes[1, 0].set_title(f'Concordance (MAPE={results["mape"]:.1f}%)')
    axes[1, 0].legend(); axes[1, 0].set_aspect('equal')

    errs = (mu_rec[tissue_bool] - mu_gt[tissue_bool]).ravel()
    axes[1, 1].hist(errs, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Error (recovered - GT) mm$^{-1}$')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Error Distribution (median={np.median(errs):.3f})')

    depths = np.arange(DEPTH)
    axes[1, 2].plot(depths, mu_gt[:, A_LINES // 2],  'b-',  lw=1.5, label='Ground Truth')
    axes[1, 2].plot(depths, mu_rec[:, A_LINES // 2], 'r--', lw=1.5, label='Recovered')
    axes[1, 2].set_xlabel('Depth (pixels)')
    axes[1, 2].set_ylabel('$\\mu_t$ (mm$^{-1}$)')
    axes[1, 2].set_title('Central A-line Profile')
    axes[1, 2].legend()
    axes[1, 2].set_xlim([max(0, TISSUE_START - 20), min(DEPTH, TISSUE_END + 20)])

    fig.suptitle(f'random_pixel — Phantom {results["index"]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison_panel.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Summary + convergence filmstrip
# ═══════════════════════════════════════════════════════════════════════════

def _load_all_results():
    """Read every results.json under OUT_ROOT/phantom_NNNN/."""
    results = []
    if not os.path.isdir(OUT_ROOT):
        return results
    for d in sorted(os.listdir(OUT_ROOT)):
        rf = os.path.join(OUT_ROOT, d, "results.json")
        if os.path.isfile(rf):
            with open(rf) as f:
                results.append(json.load(f))
    return results


def generate_summary():
    """Aggregate all phantom results, build the convergence filmstrip data,
    and stage the figure2_verification inputs."""
    summary_dir = os.path.join(OUT_ROOT, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    results = _load_all_results()
    if not results:
        print("No results found. Run the optimization first.")
        return

    print(f"Loaded {len(results)} completed phantoms.")

    # Aggregate stats
    mapes = np.array([r['mape'] for r in results])
    maes  = np.array([r['mae']  for r in results])
    rmses = np.array([r['rmse'] for r in results])
    gts   = np.array([r['mu_gt_tissue_mean']        for r in results])
    recs  = np.array([r['mu_recovered_tissue_mean'] for r in results])
    corr  = np.corrcoef(gts, recs)[0, 1] if len(gts) > 1 else float('nan')

    print(f"  MAPE = {mapes.mean():.2f}% (median {np.median(mapes):.2f}%, "
          f"std {mapes.std():.2f}%)")
    print(f"  MAE  = {maes.mean():.3f} +/- {maes.std():.3f} mm^-1")
    print(f"  R    = {corr:.4f}")
    print(f"  <5%: {(mapes < 5).mean():.0%}    "
          f"<10%: {(mapes < 10).mean():.0%}    "
          f"<15%: {(mapes < 15).mean():.0%}")

    summary_table = {
        'n': len(results),
        'mape_mean': float(mapes.mean()), 'mape_std': float(mapes.std()),
        'mape_median': float(np.median(mapes)),
        'mae_mean':  float(maes.mean()),  'mae_std':  float(maes.std()),
        'rmse_mean': float(rmses.mean()), 'rmse_std': float(rmses.std()),
        'correlation': float(corr),
        'frac_below_5': float((mapes < 5).mean()),
        'frac_below_10': float((mapes < 10).mean()),
        'frac_below_15': float((mapes < 15).mean()),
    }
    with open(os.path.join(summary_dir, 'summary_table.json'), 'w') as f:
        json.dump(summary_table, f, indent=2)

    # Diagnostic plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(mapes, bins=40, color='steelblue', edgecolor='none', alpha=0.85)
    axes[0].set_xlabel('MAPE (%)'); axes[0].set_ylabel('Count')
    axes[0].axvline(10, color='green', linestyle='--', alpha=0.6, label='10% error')
    axes[0].set_title(f'MAPE distribution (n={len(mapes)})')
    axes[0].legend()

    lims = [0, max(gts.max(), recs.max()) * 1.1]
    axes[1].scatter(gts, recs, alpha=0.4, s=12, c='steelblue', edgecolors='none')
    axes[1].plot(lims, lims, 'r--', linewidth=1)
    axes[1].set_xlim(lims); axes[1].set_ylim(lims)
    axes[1].set_xlabel('GT mean tissue $\\mu_t$ (mm$^{-1}$)')
    axes[1].set_ylabel('Recovered mean tissue $\\mu_t$')
    axes[1].set_title(f'Concordance (R={corr:.3f})')
    axes[1].set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'summary.png'), dpi=200)
    plt.close()

    # Build the convergence filmstrip (driven by snapshots.npz files)
    filmstrip_data_path = _build_convergence_filmstrip(results, summary_dir)

    # Stage the subset the Figure 2 notebook reads
    _stage_to_figure_paper_2(filmstrip_data_path)


def _build_convergence_filmstrip(results, summary_dir):
    """Aggregate snapshots.npz files into a single npz of pixel trajectories.

    Returns the path to convergence_filmstrip_data.npz so the staging step can
    copy it into figures/data/figure2_verification/.
    """
    PIXELS_PER_PHANTOM = 100  # 500 phantoms x 100 = 50,000 unique pixels per iter
    paper_dir = os.path.join(summary_dir, 'paper')
    os.makedirs(paper_dir, exist_ok=True)

    phantom_gt_pixels = []
    phantom_rec_snaps = []
    snapshot_iters = None
    cleaned_paths = []

    for r in results:
        run_dir = os.path.join(OUT_ROOT, f"phantom_{r['index']:04d}")
        snap_path = os.path.join(run_dir, 'snapshots.npz')
        if not os.path.exists(snap_path):
            continue
        snap = np.load(snap_path)
        tissue_gt = snap['tissue_gt']
        tissue_mu = snap['tissue_mu']  # (n_snap, n_pixels)
        snap_iters = snap['iters']
        if len(tissue_gt) == 0:
            continue
        if snapshot_iters is None:
            snapshot_iters = snap_iters

        rng = np.random.default_rng(seed=r['index'])
        n_tissue = len(tissue_gt)
        if n_tissue > PIXELS_PER_PHANTOM:
            pick = rng.choice(n_tissue, PIXELS_PER_PHANTOM, replace=False)
        else:
            pick = np.arange(n_tissue)
        phantom_gt_pixels.append(tissue_gt[pick])
        phantom_rec_snaps.append(tissue_mu[:, pick])
        cleaned_paths.append(snap_path)

    if not phantom_gt_pixels or snapshot_iters is None:
        print("  [Filmstrip] No snapshots.npz found, skipping.")
        return None

    pixel_gt_all  = np.concatenate(phantom_gt_pixels)
    pixel_rec_all = np.concatenate(phantom_rec_snaps, axis=1)

    filmstrip_data_path = os.path.join(paper_dir, 'convergence_filmstrip_data.npz')
    np.savez(filmstrip_data_path,
             pixel_gt_all=pixel_gt_all,
             pixel_rec_all=pixel_rec_all,
             snapshot_iters=snapshot_iters)
    print(f"  [Filmstrip] {filmstrip_data_path}  "
          f"({len(phantom_gt_pixels)} phantoms, "
          f"{pixel_gt_all.shape[0]} pixels, {len(snapshot_iters)} iters)")

    # Snapshot files are bulky; remove once aggregated.
    for p in cleaned_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    return filmstrip_data_path


# ═══════════════════════════════════════════════════════════════════════════
# Stage to figures/data/figure2_verification
# ═══════════════════════════════════════════════════════════════════════════

def _stage_to_figure_paper_2(filmstrip_data_path):
    """Copy the inverse-verification subset that the Figure 2 notebook reads.

    Picks the showcase phantom (phantom_0000), computes the iter-0 forward-model
    prediction needed for panel (m), and writes the showcase_* files plus a copy
    of convergence_filmstrip_data.npz into figures/data/figure2_verification/.
    """
    import shutil

    fig_paper_2 = os.path.join(PHANTOM_ROOT, 'figures', 'data', 'figure2_verification')
    os.makedirs(fig_paper_2, exist_ok=True)

    if filmstrip_data_path and os.path.isfile(filmstrip_data_path):
        shutil.copy2(filmstrip_data_path,
                     os.path.join(fig_paper_2, 'convergence_filmstrip_data.npz'))

    showcase_dir = os.path.join(OUT_ROOT, 'phantom_0000')
    if not os.path.isdir(showcase_dir):
        print(f"  [stage] showcase dir missing ({showcase_dir}); skipping showcase staging")
        return

    rename_map = {
        'mu_init_mm.npy':       'showcase_mu_init_mm.npy',
        'mu_gt_mm.npy':         'showcase_mu_gt_mm.npy',
        'mu_recovered_mm.npy':  'showcase_mu_recovered_mm.npy',
        'target_dB.npy':        'showcase_target_dB.npy',
        'alpha_gt.npy':         'showcase_alpha_gt.npy',
    }
    for src_name, dst_name in rename_map.items():
        src = os.path.join(showcase_dir, src_name)
        dst = os.path.join(fig_paper_2, dst_name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    required = ['mu_init_mm.npy', 'mu_gt_mm.npy', 'mu_recovered_mm.npy', 'target_dB.npy']
    missing = [n for n in required if not os.path.isfile(os.path.join(showcase_dir, n))]
    if missing:
        print(f"  [stage] showcase missing {missing}; skipping iter-0 derivation.")
        return

    mu_init   = np.load(os.path.join(showcase_dir, 'mu_init_mm.npy'))
    mu_gt     = np.load(os.path.join(showcase_dir, 'mu_gt_mm.npy'))
    mu_rec    = np.load(os.path.join(showcase_dir, 'mu_recovered_mm.npy'))
    target_dB = np.load(os.path.join(showcase_dir, 'target_dB.npy'))
    alpha_gt_path = os.path.join(showcase_dir, 'alpha_gt.npy')
    if os.path.isfile(alpha_gt_path):
        alpha_gt = np.load(alpha_gt_path)
    else:
        alpha_gt = np.ones_like(mu_gt, dtype=np.float32)
        np.save(os.path.join(fig_paper_2, 'showcase_alpha_gt.npy'), alpha_gt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = OmegaConf.load(os.path.join(PHANTOM_ROOT, 'configs', 'config.yaml'))
    cfg.forward_model_system.depth_samples = mu_init.shape[0]
    cfg.forward_model_system.a_lines       = mu_init.shape[1]
    cfg.forward_model_system.forward_model_type = 'ss'
    fwd = create_forward_model(cfg, device=device)
    noise_floor_dB = float(cfg.forward_model_system.noise_floor_dB)

    with torch.no_grad():
        pred_uncal = fwd(
            torch.tensor(mu_init,  dtype=torch.float32, device=device),
            torch.tensor(alpha_gt, dtype=torch.float32, device=device),
        ).cpu().numpy()
    mask = target_dB > noise_floor_dB
    gain_dB_frozen = float(np.median(target_dB[mask]) - np.median(pred_uncal[mask]))
    fwd.gain_dB.fill_(gain_dB_frozen)
    with torch.no_grad():
        pred_iter0 = fwd(
            torch.tensor(mu_init,  dtype=torch.float32, device=device),
            torch.tensor(alpha_gt, dtype=torch.float32, device=device),
        ).cpu().numpy()
    np.save(os.path.join(fig_paper_2, 'showcase_pred_iter0_dB.npy'), pred_iter0)

    tissue = mu_gt > 1e-3
    if tissue.sum() > 0:
        mape = float(np.mean(np.abs(mu_rec[tissue] - mu_gt[tissue])
                             / np.maximum(mu_gt[tissue], 1e-6))) * 100.0
        mae  = float(np.mean(np.abs(mu_rec[tissue] - mu_gt[tissue])))
    else:
        mape = mae = float('nan')
    json.dump(
        dict(phase='random_pixel', phantom_idx=0,
             mape=mape, mae=mae, gain_dB_frozen=gain_dB_frozen),
        open(os.path.join(fig_paper_2, 'showcase_results.json'), 'w'), indent=2)
    print(f"  [stage → figure2_verification] {fig_paper_2} (showcase + filmstrip)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if int(os.environ.get('SUMMARIZE', '0')):
        generate_summary()
        return

    start_idx = int(os.environ.get('START_INDEX', '0'))
    end_idx   = int(os.environ.get('END_INDEX', str(N_PHANTOMS)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Running random_pixel phantoms {start_idx}..{end_idx - 1}")

    os.makedirs(OUT_ROOT, exist_ok=True)
    for idx in range(start_idx, end_idx):
        try:
            run_single_phantom(idx, device)
        except Exception as e:
            print(f"[#{idx}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()
