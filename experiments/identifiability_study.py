#!/usr/bin/env python
"""Identifiability study for the differentiable OCT inverse solver.

Investigates whether the optimizer converges to a unique solution, independent
of initialization, across two experimental phases:

  Phase 1 — Random seed sweep (20 seeds x 10 phantoms)
  Phase 2 — Initialization range sweep (10 conditions x 10 phantoms x 5 seeds)

Both phases use:
  - Single-scatter (Beer-Lambert) forward model only
  - N(z) noise floor subtraction before optimization
  - Per-iteration MSE-optimal gain recomputation (Stage 1)
  - IntMask tissue statistics for mu reporting

Usage (Figure 4 panels a–d):
    python experiments/identifiability_study.py
    PHASE=1 python experiments/identifiability_study.py
    PHASE=1 PHANTOM_INDEX=3 SEED=5 python experiments/identifiability_study.py
    SUMMARIZE=1 PHASE=1 python experiments/identifiability_study.py
"""

import csv
import gc
import math
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.helpers import discover_despeckled_phantoms, load_phantom_bscan, save_results

from src.forward import create_forward_model
from src.inverse import (
    optimize,
    build_noise_floor_mask,
    subtract_noise_floor_linear,
    compute_loss,
)
from src.roi import load_intmask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CF_REFERENCE = {1.0: 4.523, 1.5: 5.914, 2.0: 5.878, 3.0: 7.742, 5.0: 9.346}

# ---- Adjustment parameters (mimic test 11 setup) ----
ADJUSTED_FOCAL_PLANE_PX = 261      # MATLAB-optimized (was 231)

CONC_COLORS = {
    1.0: "tab:blue",
    1.5: "tab:orange",
    2.0: "tab:green",
    3.0: "tab:red",
    5.0: "tab:purple",
}

INIT_CONDITIONS = {
    "0_3": (0.0, 3.0),
    "3_7": (3.0, 7.0),
    "7_15": (7.0, 15.0),
    "0_10": (0.0, 10.0),
    "0_20": (0.0, 20.0),
    "1_1": (1.0, 1.0),
    "5_5": (5.0, 5.0),
    "10_10": (10.0, 10.0),
    "15_15": (15.0, 15.0),
    "informed": None,
}






# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_configs() -> tuple:
    """Load all three YAML configs from the configs directory.

    Returns:
        (config, opt_config, sc_config) as OmegaConf objects.
    """
    cfg_dir = os.path.join(PHANTOM_ROOT, "configs")
    config = OmegaConf.load(os.path.join(cfg_dir, "config.yaml"))
    opt_config = OmegaConf.load(os.path.join(cfg_dir, "optimization_config_fast.yaml"))
    sc_config = OmegaConf.load(os.path.join(cfg_dir, "scatterer_config.yaml"))
    return config, opt_config, sc_config


def _make_ss_forward(config: OmegaConf, device: torch.device) -> nn.Module:
    """Create and return the SS forward model on the given device.

    Applies the adjusted focal plane position (zFP=261) from test 11.
    """
    cfg_ss = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg_ss.forward_model_system.forward_model_type = "ss"
    original_zfp = float(cfg_ss.forward_model_system.focal_plane_position_pixels)
    cfg_ss.forward_model_system.focal_plane_position_pixels = ADJUSTED_FOCAL_PLANE_PX
    print(f"  Focal plane override: {original_zfp} -> {ADJUSTED_FOCAL_PLANE_PX} px")
    model = create_forward_model(cfg_ss, device=str(device))
    model.to(device)
    return model


def _compute_tissue_stats(
    mu_np: np.ndarray,
    intmask_np: np.ndarray,
    nf_mask_np: np.ndarray,
) -> dict:
    """Compute tissue mu statistics filtered by IntMask AND noise floor mask.

    Args:
        mu_np: Full mu map, shape (D, A).
        intmask_np: Binary IntMask, shape (D, A).
        nf_mask_np: Binary noise floor mask (1=tissue), shape (D, A).

    Returns:
        Dict with keys: mu_tissue_mean, mu_tissue_std, mu_tissue_median, n_tissue.
    """
    combined = intmask_np & nf_mask_np
    mu_tissue = mu_np[combined]
    if len(mu_tissue) == 0:
        return dict(
            mu_tissue_mean=float("nan"),
            mu_tissue_std=float("nan"),
            mu_tissue_median=float("nan"),
            n_tissue=0,
        )
    return dict(
        mu_tissue_mean=float(mu_tissue.mean()),
        mu_tissue_std=float(mu_tissue.std()),
        mu_tissue_median=float(np.median(mu_tissue)),
        n_tissue=int(combined.sum()),
    )


def _make_mu_init(
    init_label: str,
    seed: int,
    D: int,
    A: int,
    device: torch.device,
    noise_floor_mask: torch.Tensor,
    cf_ref: float,
) -> torch.Tensor:
    """Create the initial mu tensor for a given init condition and seed.

    Args:
        init_label: Key from INIT_CONDITIONS (or 'informed').
        seed: RNG seed for reproducibility.
        D: Depth dimension.
        A: A-line dimension.
        device: Torch device.
        noise_floor_mask: Binary mask (1=tissue), shape (D, A).
        cf_ref: CF reference mu value for this phantom (used for 'informed').

    Returns:
        mu_init tensor of shape (D, A) on device.
    """
    torch.manual_seed(seed)
    bounds = INIT_CONDITIONS[init_label]

    if init_label == "informed":
        mu_init = torch.zeros(D, A, dtype=torch.float32, device=device)
        tissue = noise_floor_mask > 0.5
        mu_init[tissue] = cf_ref
        return mu_init

    lo, hi = bounds
    if lo == hi:
        mu_init = torch.full((D, A), lo, dtype=torch.float32, device=device)
    else:
        mu_init = torch.empty(D, A, dtype=torch.float32, device=device).uniform_(lo, hi)
    return mu_init


def _cleanup_gpu(*tensors) -> None:
    """Delete tensors, close matplotlib figures, and free GPU memory."""
    for t in tensors:
        if t is not None:
            del t
    plt.close("all")
    gc.collect()
    torch.cuda.empty_cache()


def _write_csv(path: str, rows: list, fieldnames: list) -> None:
    """Write a list of dicts to CSV using stdlib csv.DictWriter."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Stage 2: alpha-only optimization
# ---------------------------------------------------------------------------


def run_optimization(
    phantom: dict,
    phantom_index: int,
    bscan_index: int,
    seed: int,
    init_label: str,
    out_dir: str,
    config: OmegaConf,
    opt_config: OmegaConf,
    device: torch.device,
    mat_dir: str,
) -> dict:
    """Run a single SS optimization for Phases 1-3.

    Loads the phantom B-scan, applies N(z) correction, initializes mu from the
    given init condition, runs optimize(), and saves roi_stats.npz.

    Args:
        phantom: Dict with keys: path, concentration, replicate, basename.
        phantom_index: Tomogram index in [0, 9].
        bscan_index: B-scan index in [0, 511].
        seed: RNG seed for mu_init.
        init_label: Key from INIT_CONDITIONS.
        out_dir: Per-run output directory.
        config: System OmegaConf config.
        opt_config: Optimization OmegaConf config.
        device: Torch device.
        mat_dir: Directory containing IntMask.mat.

    Returns:
        Dict with run results (mu_tissue_mean, mse_final, best_iter, gain_dB, time_sec).
    """
    os.makedirs(out_dir, exist_ok=True)

    noise_floor_dB = float(config.forward_model_system.noise_floor_dB)
    prune_threshold = float(config.patient.prune_threshold_m_inv)
    D = int(config.forward_model_system.depth_samples)
    A = int(config.forward_model_system.a_lines)

    I_target_np = load_phantom_bscan(phantom["path"], bscan_index)
    I_target_orig = torch.tensor(I_target_np, dtype=torch.float32, device=device)
    I_target_dB = subtract_noise_floor_linear(I_target_orig, noise_floor_dB)
    noise_floor_mask = build_noise_floor_mask(I_target_dB, noise_floor_dB, device)

    cf_ref = CF_REFERENCE.get(phantom["concentration"], 5.0)
    mu_init_data = _make_mu_init(
        init_label, seed, D, A, device, noise_floor_mask, cf_ref
    )

    alpha_init_data = torch.ones(D, A, dtype=torch.float32, device=device)

    forward_model = _make_ss_forward(config, device)

    run_opt_config = OmegaConf.create(OmegaConf.to_container(opt_config, resolve=True))
    run_opt_config.optimization.panel_save_interval = 99999

    mu = nn.Parameter(mu_init_data.clone())
    alpha = nn.Parameter(alpha_init_data.clone())
    forward_model.gain_dB.fill_(0.0)

    t0 = time.time()
    mu_best, alpha_best, metrics, seq_steps, seq_imgs, mu_init, alpha_init = optimize(
        forward_model,
        mu,
        alpha,
        I_target_dB,
        run_opt_config,
        prune_threshold,
        device,
        out_dir,
        noise_floor_mask=noise_floor_mask,
        recompute_gain_mse_optimal_each_iter=True,
    )
    elapsed = time.time() - t0

    mu_np = mu_best.detach().cpu().numpy()
    np.save(os.path.join(out_dir, "mu_final_mm.npy"), mu_np)

    intmask_np = load_intmask(phantom_index, bscan_index, mat_dir=mat_dir)
    nf_mask_np = noise_floor_mask.cpu().numpy() > 0.5
    stats = _compute_tissue_stats(mu_np, intmask_np, nf_mask_np)

    mse_arr = np.array(metrics["mse"])
    best_idx = int(np.argmin(mse_arr))
    mse_final = float(mse_arr[best_idx])
    best_iter = int(metrics["iters"][best_idx])
    gain_dB_val = float(forward_model.gain_dB.item())

    np.savez(
        os.path.join(out_dir, "roi_stats.npz"),
        mu_tissue_mean=stats["mu_tissue_mean"],
        mu_tissue_std=stats["mu_tissue_std"],
        mu_tissue_median=stats["mu_tissue_median"],
        mse_final=mse_final,
        best_iter=best_iter,
        gain_dB=gain_dB_val,
        time_sec=elapsed,
        n_tissue=stats["n_tissue"],
    )
    np.savez(
        os.path.join(out_dir, "metrics.npz"),
        iters=np.array(metrics["iters"]),
        mse=np.array(metrics["mse"]),
        mu_mean_mm=np.array(metrics["mu_mean_mm"]),
        mu_std_mm=np.array(metrics["mu_std_mm"]),
        gain_dB=np.array(metrics["gain_dB"]),
        lr=np.array(metrics["lr"]),
        time=np.array(metrics["time"]),
    )

    result = dict(
        phantom_index=phantom_index,
        concentration=phantom["concentration"],
        seed=seed,
        init_label=init_label,
        mse_final=mse_final,
        best_iter=best_iter,
        gain_dB=gain_dB_val,
        time_sec=elapsed,
        **stats,
    )

    _cleanup_gpu(
        mu, alpha, mu_best, alpha_best, mu_init, alpha_init,
        I_target_dB, I_target_orig, noise_floor_mask,
    )
    del forward_model, run_opt_config, metrics, seq_steps, seq_imgs

    return result


# ---------------------------------------------------------------------------

def summarize_phase1(out_base: str, phantoms: list) -> None:
    """Violin plot of mu_tissue_mean across 20 seeds per phantom, plus CSV and extra plots.

    Args:
        out_base: Base output directory for phase1 results.
        phantoms: List of phantom dicts from discover_despeckled_phantoms.
    """
    phase_dir = os.path.join(out_base, "phase1_seed_sweep")
    summary_dir = os.path.join(out_base, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    n_phantoms = len(phantoms)
    n_seeds = 20

    data_per_phantom = []
    csv_rows = []
    labels = []
    for pidx, phantom in enumerate(phantoms):
        vals = []
        for seed in range(n_seeds):
            run_dir = os.path.join(phase_dir, f"phantom_{pidx}_seed_{seed:02d}")
            stats_path = os.path.join(run_dir, "roi_stats.npz")
            if os.path.isfile(stats_path):
                d = np.load(stats_path)
                v = float(d["mu_tissue_mean"])
                mse_val = float(d["mse_final"])
                csv_rows.append(dict(
                    phantom_index=pidx,
                    concentration=phantom["concentration"],
                    replicate=phantom["replicate"],
                    seed=seed,
                    mu_tissue_mean=v,
                    mse_final=mse_val,
                ))
                if not np.isnan(v):
                    vals.append(v)
        data_per_phantom.append(vals)
        conc = phantom["concentration"]
        rep = phantom["replicate"]
        labels.append(f"{conc:.1f}%\nrep{rep}")

    # --- CSV ---
    if csv_rows:
        csv_path = os.path.join(summary_dir, "phase1_seed_sweep_results.csv")
        _write_csv(csv_path, csv_rows, list(csv_rows[0].keys()))

    # --- Existing figure: violin seed consistency ---
    fig, ax = plt.subplots(figsize=(14, 7))
    positions = list(range(1, n_phantoms + 1))

    for pos, vals in zip(positions, data_per_phantom):
        if len(vals) > 1:
            parts = ax.violinplot([vals], positions=[pos], widths=0.7, showmeans=True)
            for pc in parts["bodies"]:
                pc.set_alpha(0.6)
        elif len(vals) == 1:
            ax.plot(pos, vals[0], "ko", ms=8)

    for pos, vals in zip(positions, data_per_phantom):
        if len(vals) > 1:
            cv = float(np.std(vals) / np.mean(vals) * 100)
            ax.annotate(
                f"CV={cv:.1f}%",
                xy=(pos, float(np.mean(vals))),
                xytext=(pos, float(np.max(vals)) + 0.3),
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Phantom (concentration, replicate)")
    ax.set_ylabel("Tissue mu (mm$^{-1}$)")
    ax.set_title("Phase 1: Seed Consistency (20 seeds per phantom, zFP=261)")
    ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    out_path = os.path.join(summary_dir, "phase1_seed_consistency.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # --- New figure: convergence overlay for 3 representative phantoms ---
    rep_phantoms = [0, 4, 8]  # 1.0%, 2.0%, 5.0% (focal 1)
    fig_conv, axes_conv = plt.subplots(1, len(rep_phantoms), figsize=(6 * len(rep_phantoms), 5))
    if len(rep_phantoms) == 1:
        axes_conv = [axes_conv]
    for ax_c, pidx in zip(axes_conv, rep_phantoms):
        if pidx >= n_phantoms:
            continue
        for seed in range(n_seeds):
            metrics_path = os.path.join(
                phase_dir, f"phantom_{pidx}_seed_{seed:02d}", "metrics.npz"
            )
            if os.path.isfile(metrics_path):
                m = np.load(metrics_path)
                ax_c.plot(m["iters"], m["mse"], alpha=0.4, lw=0.8)
        ax_c.set_xlabel("Iteration")
        ax_c.set_ylabel("MSE (dB²)")
        ax_c.set_title(f"Phantom {pidx} ({phantoms[pidx]['concentration']:.1f}%)")
        ax_c.grid(alpha=0.3)
    plt.suptitle("Phase 1: Convergence Overlay (20 seeds)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_conv = os.path.join(summary_dir, "phase1_convergence_overlay.png")
    plt.savefig(out_conv, dpi=150, bbox_inches="tight")
    plt.close(fig_conv)
    print(f"Saved: {out_conv}")

    # --- New figure: box plot per concentration (both reps pooled) ---
    concentrations = sorted(set(p["concentration"] for p in phantoms))
    fig_box, ax_box = plt.subplots(figsize=(10, 7))
    box_data = []
    box_labels = []
    for conc in concentrations:
        pooled = []
        for pidx, phantom in enumerate(phantoms):
            if phantom["concentration"] == conc:
                pooled.extend(data_per_phantom[pidx])
        box_data.append(pooled)
        box_labels.append(f"{conc:.1f}%")
    bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True)
    for i, (conc, patch) in enumerate(zip(concentrations, bp["boxes"])):
        color = CONC_COLORS.get(conc, "gray")
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax_box.set_xlabel("IL Concentration")
    ax_box.set_ylabel("Tissue mu (mm$^{-1}$)")
    ax_box.set_title("Phase 1: Seed Sweep per Concentration (both reps pooled, zFP=261)")
    ax_box.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    out_box = os.path.join(summary_dir, "phase1_box_per_concentration.png")
    plt.savefig(out_box, dpi=150, bbox_inches="tight")
    plt.close(fig_box)
    print(f"Saved: {out_box}")


def summarize_phase2(out_base: str, phantoms: list) -> None:
    """Grouped bar plot of mu_tissue_mean per init condition, plus CSV and extra plots.

    Args:
        out_base: Base output directory for phase2 results.
        phantoms: List of phantom dicts from discover_despeckled_phantoms.
    """
    phase_dir = os.path.join(out_base, "phase2_init_range")
    summary_dir = os.path.join(out_base, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    init_labels = list(INIT_CONDITIONS.keys())
    n_seeds = 5
    n_phantoms = len(phantoms)

    means = np.full((len(init_labels), n_phantoms), np.nan)
    stds = np.full((len(init_labels), n_phantoms), np.nan)
    csv_rows = []

    for ii, init_label in enumerate(init_labels):
        for pidx in range(n_phantoms):
            vals = []
            for seed in range(n_seeds):
                run_dir = os.path.join(
                    phase_dir, init_label, f"phantom_{pidx}_seed_{seed}"
                )
                stats_path = os.path.join(run_dir, "roi_stats.npz")
                if os.path.isfile(stats_path):
                    d = np.load(stats_path)
                    v = float(d["mu_tissue_mean"])
                    mse_val = float(d["mse_final"])
                    csv_rows.append(dict(
                        phantom_index=pidx,
                        concentration=phantoms[pidx]["concentration"],
                        replicate=phantoms[pidx]["replicate"],
                        init_condition=init_label,
                        seed=seed,
                        mu_tissue_mean=v,
                        mse_final=mse_val,
                    ))
                    if not np.isnan(v):
                        vals.append(v)
            if vals:
                means[ii, pidx] = float(np.mean(vals))
                stds[ii, pidx] = float(np.std(vals))

    # --- CSV ---
    if csv_rows:
        csv_path = os.path.join(summary_dir, "phase2_init_range_results.csv")
        _write_csv(csv_path, csv_rows, list(csv_rows[0].keys()))

    # --- Existing figure: grouped bar plot ---
    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(n_phantoms)
    bar_width = 0.8 / len(init_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(init_labels)))

    for ii, (label, color) in enumerate(zip(init_labels, colors)):
        offset = (ii - len(init_labels) / 2 + 0.5) * bar_width
        valid = ~np.isnan(means[ii])
        ax.bar(
            x[valid] + offset,
            means[ii, valid],
            width=bar_width,
            label=label,
            color=color,
            alpha=0.8,
            yerr=stds[ii, valid],
            capsize=2,
        )

    phantom_labels = [f"{p['concentration']:.1f}%\nr{p['replicate']}" for p in phantoms]
    ax.set_xticks(x)
    ax.set_xticklabels(phantom_labels, fontsize=8)
    ax.set_xlabel("Phantom")
    ax.set_ylabel("Tissue mu (mm$^{-1}$)")
    ax.set_title("Phase 2: Initialization Range Sensitivity (mean +/- std across 5 seeds, zFP=261)")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    out_path = os.path.join(summary_dir, "phase2_init_sensitivity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # --- New figure: sensitivity curves (mu vs init condition per phantom) ---
    fig_sc, ax_sc = plt.subplots(figsize=(14, 7))
    for pidx, phantom in enumerate(phantoms):
        color = CONC_COLORS.get(phantom["concentration"], "gray")
        y_vals = means[:, pidx]
        valid = ~np.isnan(y_vals)
        if valid.any():
            ax_sc.plot(
                np.arange(len(init_labels))[valid], y_vals[valid],
                "-o", color=color, lw=1.2, ms=5, alpha=0.7,
                label=f"P{pidx} {phantom['concentration']:.1f}% r{phantom['replicate']}",
            )
    ax_sc.set_xticks(range(len(init_labels)))
    ax_sc.set_xticklabels(init_labels, rotation=40, ha="right", fontsize=8)
    ax_sc.set_xlabel("Init Condition")
    ax_sc.set_ylabel("Tissue mu (mm$^{-1}$)")
    ax_sc.set_title("Phase 2: Sensitivity Curves (mu vs init condition per phantom)")
    ax_sc.legend(fontsize=7, loc="upper left", ncol=2)
    ax_sc.grid(alpha=0.3)
    plt.tight_layout()
    out_sc = os.path.join(summary_dir, "phase2_sensitivity_curves.png")
    plt.savefig(out_sc, dpi=150, bbox_inches="tight")
    plt.close(fig_sc)
    print(f"Saved: {out_sc}")

    # --- New figure: absolute mu heatmap (init x phantom) ---
    fig_hm, ax_hm = plt.subplots(figsize=(14, 7))
    im = ax_hm.imshow(means, aspect="auto", cmap="viridis", origin="upper")
    for ii in range(len(init_labels)):
        for pidx in range(n_phantoms):
            val = means[ii, pidx]
            if not np.isnan(val):
                ax_hm.text(pidx, ii, f"{val:.2f}", ha="center", va="center", fontsize=7,
                           color="white" if val > np.nanmedian(means) else "black")
    ax_hm.set_xticks(range(n_phantoms))
    ax_hm.set_xticklabels(phantom_labels, fontsize=8)
    ax_hm.set_yticks(range(len(init_labels)))
    ax_hm.set_yticklabels(init_labels, fontsize=9)
    ax_hm.set_xlabel("Phantom")
    ax_hm.set_ylabel("Init Condition")
    ax_hm.set_title("Phase 2: Absolute mu Heatmap (init x phantom)")
    plt.colorbar(im, ax=ax_hm, label="mu (mm$^{-1}$)")
    plt.tight_layout()
    out_hm = os.path.join(summary_dir, "phase2_mu_heatmap.png")
    plt.savefig(out_hm, dpi=150, bbox_inches="tight")
    plt.close(fig_hm)
    print(f"Saved: {out_hm}")

    _stage_to_figure_paper_4(out_base, phantoms, summary_dir)


def summarize_phase1_phase2_table(out_base: str, phantoms: list) -> None:
    """Combined table: Phase 1 seed CV + Phase 2 mu means per init condition.

    Rows = phantoms, columns = Phase 1 CV% then Phase 2 mu per init condition.
    Saved as CSV and PNG.
    """
    summary_dir = os.path.join(out_base, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    n_phantoms = len(phantoms)
    init_labels = list(INIT_CONDITIONS.keys())

    # ---- Phase 1: CV per phantom (20 seeds, init 0_10) ----
    phase1_dir = os.path.join(out_base, "phase1_seed_sweep")
    n_seeds_p1 = 20
    p1_mean = np.full(n_phantoms, np.nan)
    p1_cv = np.full(n_phantoms, np.nan)
    for pidx in range(n_phantoms):
        vals = []
        for seed in range(n_seeds_p1):
            sp = os.path.join(phase1_dir, f"phantom_{pidx}_seed_{seed:02d}", "roi_stats.npz")
            if os.path.isfile(sp):
                v = float(np.load(sp)["mu_tissue_mean"])
                if not np.isnan(v):
                    vals.append(v)
        if len(vals) > 1:
            p1_mean[pidx] = float(np.mean(vals))
            p1_cv[pidx] = float(np.std(vals) / np.mean(vals) * 100)

    # ---- Phase 2: mean mu per init condition per phantom (5 seeds each) ----
    phase2_dir = os.path.join(out_base, "phase2_init_range")
    n_seeds_p2 = 5
    p2_means = np.full((len(init_labels), n_phantoms), np.nan)
    for ii, init_label in enumerate(init_labels):
        for pidx in range(n_phantoms):
            vals = []
            for seed in range(n_seeds_p2):
                sp = os.path.join(
                    phase2_dir, init_label, f"phantom_{pidx}_seed_{seed}", "roi_stats.npz")
                if os.path.isfile(sp):
                    v = float(np.load(sp)["mu_tissue_mean"])
                    if not np.isnan(v):
                        vals.append(v)
            if vals:
                p2_means[ii, pidx] = float(np.mean(vals))

    # ---- Build CSV rows ----
    csv_rows = []
    for pidx, phantom in enumerate(phantoms):
        conc = phantom["concentration"]
        rep = phantom["replicate"]
        cf_ref = CF_REFERENCE.get(conc, float("nan"))
        row = dict(
            phantom=pidx,
            concentration=f"{conc:.1f}%",
            replicate=rep,
            cf_reference=round(cf_ref, 2),
            phase1_mu_mean=round(p1_mean[pidx], 2) if not np.isnan(p1_mean[pidx]) else "",
            phase1_cv_pct=round(p1_cv[pidx], 1) if not np.isnan(p1_cv[pidx]) else "",
        )
        for ii, init_label in enumerate(init_labels):
            row[f"p2_{init_label}"] = (
                round(p2_means[ii, pidx], 2) if not np.isnan(p2_means[ii, pidx]) else ""
            )
        csv_rows.append(row)

    csv_path = os.path.join(summary_dir, "phase1_phase2_combined.csv")
    if csv_rows:
        _write_csv(csv_path, csv_rows, list(csv_rows[0].keys()))
        print(f"Saved: {csv_path}")

    # ---- PNG table ----
    col_headers = ["Phantom", "Conc", "CF ref",
                    "P1 mu", "P1 CV%"] + [f"P2 {l}" for l in init_labels]
    cell_text = []
    for pidx, phantom in enumerate(phantoms):
        conc = phantom["concentration"]
        cf_ref = CF_REFERENCE.get(conc, float("nan"))
        row_cells = [
            f"P{pidx} r{phantom['replicate']}",
            f"{conc:.1f}%",
            f"{cf_ref:.2f}",
            f"{p1_mean[pidx]:.2f}" if not np.isnan(p1_mean[pidx]) else "-",
            f"{p1_cv[pidx]:.1f}%" if not np.isnan(p1_cv[pidx]) else "-",
        ]
        for ii in range(len(init_labels)):
            v = p2_means[ii, pidx]
            row_cells.append(f"{v:.2f}" if not np.isnan(v) else "-")
        cell_text.append(row_cells)

    n_cols = len(col_headers)
    fig_w = max(16, 1.2 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, 0.5 * n_phantoms + 2))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Shade CF reference and P1 CV columns
    for i in range(1, n_phantoms + 1):
        table[i, 2].set_facecolor("#E2EFDA")  # CF ref
        table[i, 4].set_facecolor("#FFF2CC")  # CV%

    ax.set_title(
        "Phase 1 Seed Consistency + Phase 2 Init Range Sensitivity",
        fontsize=12, fontweight="bold", pad=20,
    )
    plt.tight_layout()
    out_path = os.path.join(summary_dir, "phase1_phase2_combined_table.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def write_grand_summary_identifiability(out_base: str, phantoms: list) -> None:
    """Write a grand summary CSV with key metrics from each phase."""
    summary_dir = os.path.join(out_base, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    rows = []

    # Phase 1 summary
    phase1_dir = os.path.join(out_base, "phase1_seed_sweep")
    p1_mus = []
    for pidx in range(len(phantoms)):
        for seed in range(20):
            sp = os.path.join(phase1_dir, f"phantom_{pidx}_seed_{seed:02d}", "roi_stats.npz")
            if os.path.isfile(sp):
                d = np.load(sp)
                v = float(d["mu_tissue_mean"])
                if not np.isnan(v):
                    p1_mus.append(v)
    rows.append(dict(
        phase="1_seed_sweep", n_runs=len(p1_mus),
        mu_mean=float(np.mean(p1_mus)) if p1_mus else float("nan"),
        mu_std=float(np.std(p1_mus)) if p1_mus else float("nan"),
        cv_pct=float(np.std(p1_mus) / np.mean(p1_mus) * 100) if len(p1_mus) > 1 else float("nan"),
        notes="20 seeds x 10 phantoms",
    ))

    # Phase 2 summary
    phase2_dir = os.path.join(out_base, "phase2_init_range")
    p2_mus = []
    for init_label in INIT_CONDITIONS.keys():
        for pidx in range(len(phantoms)):
            for seed in range(5):
                sp = os.path.join(phase2_dir, init_label, f"phantom_{pidx}_seed_{seed}", "roi_stats.npz")
                if os.path.isfile(sp):
                    d = np.load(sp)
                    v = float(d["mu_tissue_mean"])
                    if not np.isnan(v):
                        p2_mus.append(v)
    rows.append(dict(
        phase="2_init_range", n_runs=len(p2_mus),
        mu_mean=float(np.mean(p2_mus)) if p2_mus else float("nan"),
        mu_std=float(np.std(p2_mus)) if p2_mus else float("nan"),
        cv_pct=float(np.std(p2_mus) / np.mean(p2_mus) * 100) if len(p2_mus) > 1 else float("nan"),
        notes="10 inits x 10 phantoms x 5 seeds",
    ))

    if rows:
        csv_path = os.path.join(summary_dir, "grand_summary_identifiability.csv")
        _write_csv(csv_path, rows, list(rows[0].keys()))


def run_phase1(
    phantoms: list,
    config: OmegaConf,
    opt_config: OmegaConf,
    device: torch.device,
    out_base: str,
    mat_dir: str,
    phantom_filter: Optional[int],
    seed_filter: Optional[int],
    run_index: Optional[int],
) -> None:
    """Run Phase 1: random seed sweep (20 seeds x 10 phantoms = 200 runs).

    Args:
        phantoms: List of phantom dicts.
        config: System config.
        opt_config: Optimization config.
        device: Torch device.
        out_base: Base output directory.
        mat_dir: Directory containing IntMask.mat.
        phantom_filter: If set, run only this phantom index.
        seed_filter: If set, run only this seed.
        run_index: If set, derive (phantom_idx, seed) from run index.
    """
    phase_dir = os.path.join(out_base, "phase1_seed_sweep")
    os.makedirs(phase_dir, exist_ok=True)

    n_seeds = 20
    bscan_index = int(os.environ.get("BSCAN_INDEX", 256))

    def _iter_runs():
        if run_index is not None:
            phantom_idx = run_index // n_seeds
            seed = run_index % n_seeds
            yield phantom_idx, seed
            return
        for pidx in range(len(phantoms)):
            if phantom_filter is not None and pidx != phantom_filter:
                continue
            for seed in range(n_seeds):
                if seed_filter is not None and seed != seed_filter:
                    continue
                yield pidx, seed

    for pidx, seed in _iter_runs():
        if pidx >= len(phantoms):
            print(f"  [Phase1] phantom_idx={pidx} out of range, skipping.")
            continue
        run_dir = os.path.join(phase_dir, f"phantom_{pidx}_seed_{seed:02d}")
        if os.path.isfile(os.path.join(run_dir, "roi_stats.npz")):
            print(f"  [Phase1] Skipping phantom={pidx} seed={seed} (already done)")
            continue
        print(f"\n[Phase1] phantom={pidx} seed={seed}")
        run_optimization(
            phantoms[pidx],
            pidx,
            bscan_index,
            seed,
            "0_10",
            run_dir,
            config,
            opt_config,
            device,
            mat_dir,
        )


def run_phase2(
    phantoms: list,
    config: OmegaConf,
    opt_config: OmegaConf,
    device: torch.device,
    out_base: str,
    mat_dir: str,
    phantom_filter: Optional[int],
    seed_filter: Optional[int],
    run_index: Optional[int],
) -> None:
    """Run Phase 2: initialization range sweep.

    10 conditions x 10 phantoms x 5 seeds = 500 runs.

    Args:
        phantoms: List of phantom dicts.
        config: System config.
        opt_config: Optimization config.
        device: Torch device.
        out_base: Base output directory.
        mat_dir: Directory containing IntMask.mat.
        phantom_filter: If set, run only this phantom index.
        seed_filter: If set, run only this seed.
        run_index: If set, derive condition from run index.
    """
    phase_dir = os.path.join(out_base, "phase2_init_range")
    os.makedirs(phase_dir, exist_ok=True)

    n_seeds = 5
    init_labels = list(INIT_CONDITIONS.keys())
    n_phantoms = len(phantoms)
    bscan_index = int(os.environ.get("BSCAN_INDEX", 256))

    def _iter_runs():
        if run_index is not None:
            init_idx = run_index // (n_seeds * n_phantoms)
            phantom_idx = (run_index // n_seeds) % n_phantoms
            seed = run_index % n_seeds
            yield init_labels[init_idx], phantom_idx, seed
            return
        for init_label in init_labels:
            for pidx in range(n_phantoms):
                if phantom_filter is not None and pidx != phantom_filter:
                    continue
                for seed in range(n_seeds):
                    if seed_filter is not None and seed != seed_filter:
                        continue
                    yield init_label, pidx, seed

    for init_label, pidx, seed in _iter_runs():
        run_dir = os.path.join(phase_dir, init_label, f"phantom_{pidx}_seed_{seed}")
        if os.path.isfile(os.path.join(run_dir, "roi_stats.npz")):
            print(f"  [Phase2] Skipping {init_label} phantom={pidx} seed={seed} (done)")
            continue
        print(f"\n[Phase2] init={init_label} phantom={pidx} seed={seed}")
        run_optimization(
            phantoms[pidx],
            pidx,
            bscan_index,
            seed,
            init_label,
            run_dir,
            config,
            opt_config,
            device,
            mat_dir,
        )


def main() -> None:
    """Entry point for the identifiability study."""
    print("=== OCT Identifiability Study ===")

    phase = int(os.environ.get("PHASE", 0))
    phantom_filter_env = os.environ.get("PHANTOM_INDEX", None)
    phantom_filter = int(phantom_filter_env) if phantom_filter_env is not None else None
    seed_filter_env = os.environ.get("SEED", None)
    seed_filter = int(seed_filter_env) if seed_filter_env is not None else None
    run_index_env = os.environ.get("RUN_INDEX", None)
    run_index = int(run_index_env) if run_index_env is not None else None
    summarize_only = int(os.environ.get("SUMMARIZE", 0)) == 1

    config, opt_config, sc_config = _load_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    input_dir = os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206")
    phantoms = discover_despeckled_phantoms(input_dir)
    print(f"\nFound {len(phantoms)} despeckled phantoms:")
    for i, p in enumerate(phantoms):
        print(f"  [{i}] {p['concentration']:.1f}% (rep {p['replicate']}): {p['basename']}")

    mat_dir = os.path.join(PHANTOM_ROOT, "matlab_CF_scripts_results")
    out_base = os.path.join(PHANTOM_ROOT, "outputs", "identifiability_study")
    os.makedirs(out_base, exist_ok=True)
    summary_dir = os.path.join(out_base, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    if summarize_only:
        phases_to_summarize = [phase] if phase != 0 else [1, 2]
        for p in phases_to_summarize:
            if p == 1:
                summarize_phase1(out_base, phantoms)
            elif p == 2:
                summarize_phase2(out_base, phantoms)
        summarize_phase1_phase2_table(out_base, phantoms)
        write_grand_summary_identifiability(out_base, phantoms)
        return

    phases_to_run = [phase] if phase != 0 else [1, 2]

    for p in phases_to_run:
        print(f"\n{'='*70}")
        print(f"=== Running Phase {p} ===")
        print(f"{'='*70}")

        if p == 1:
            run_phase1(
                phantoms, config, opt_config, device, out_base, mat_dir,
                phantom_filter, seed_filter, run_index,
            )
            if phantom_filter is None and seed_filter is None and run_index is None:
                summarize_phase1(out_base, phantoms)
                summarize_phase1_phase2_table(out_base, phantoms)
        elif p == 2:
            run_phase2(
                phantoms, config, opt_config, device, out_base, mat_dir,
                phantom_filter, seed_filter, run_index,
            )
            if phantom_filter is None and seed_filter is None and run_index is None:
                summarize_phase2(out_base, phantoms)
                summarize_phase1_phase2_table(out_base, phantoms)

    # Grand summary is only safe to write from the SUMMARIZE=1 path; calling
    # it from every per-RUN_INDEX subprocess causes a write race when 8 array
    # tasks invoke this script concurrently.
    print(f"\nDone. Outputs under: {out_base}")



def _stage_to_figure_paper_4(out_base, phantoms, summary_dir):
    """Stage the identifiability-study subset the Figure 4 notebook reads.

    Copies ``phase2_init_range_results.csv`` and the IL_5.0pct_rep1 cartoon μ
    pair (used by the panel-(a) seed-sweep diagram) into
    ``figures/data/figure4_identifiability_robustness/``. Closes the experiments → outputs → figures
    loop for the identifiability inputs.
    """
    import shutil
    fig_paper_4 = os.path.join(PHANTOM_ROOT, "figures", "data", "figure4_identifiability_robustness")
    pmu_dir     = os.path.join(fig_paper_4, "phantom_mu")
    os.makedirs(pmu_dir, exist_ok=True)

    csv_src = os.path.join(summary_dir, "phase2_init_range_results.csv")
    if os.path.isfile(csv_src):
        shutil.copy2(csv_src, os.path.join(fig_paper_4, "phase2_init_range_results.csv"))

    # Cartoon phantom = first 5.0% rep-1 acquisition. Find it by concentration/replicate.
    cartoon = None
    for pidx, p in enumerate(phantoms):
        if abs(p.get("concentration", -1) - 5.0) < 1e-6 and int(p.get("replicate", -1)) == 1:
            cartoon = pidx
            break
    if cartoon is None:
        print(f"  [stage] phantom_mu cartoon (5.0% rep1) not found; skipping")
        return

    # Phase-1 seed 0 run dir holds the recovered/init μ for that phantom
    run_dir = os.path.join(out_base, "phase1_seed_sweep", f"phantom_{cartoon}_seed_00")
    init_src  = os.path.join(run_dir, "mu_init_mm.npy")
    final_src = os.path.join(run_dir, "mu_final_mm.npy")
    if os.path.isfile(init_src):
        shutil.copy2(init_src,
                     os.path.join(pmu_dir, "IL_5.0pct_rep1_mu_init_mm.npy"))
    if os.path.isfile(final_src):
        shutil.copy2(final_src,
                     os.path.join(pmu_dir, "IL_5.0pct_rep1_mu_final_mm.npy"))
    print(f"  [stage \u2192 figure4_identifiability_robustness] phase2_init_range CSV + IL_5.0pct_rep1 phantom_mu")


if __name__ == "__main__":
    main()
