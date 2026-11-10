#!/usr/bin/env python
"""Noise robustness study (Figure 4 panels e–g): physical phantom under Gaussian noise.

Runs single-scattering optimization on every pre-noised slice of the 10
intralipid acquisitions and reports recovery vs SNR. Noise was injected
externally and stored as 29-slice TIFFs in
``inputs/IntralipidPhantoms_20250206_Noised/`` — slice indices 0..27 correspond
to integer SNR levels -3..24 dB; slice 28 is the noiseless baseline.

Each run uses:
    - Single-scatter (Beer-Lambert) forward model
    - N(z) noise floor subtraction
    - Per-iteration MSE-optimal gain recomputation
    - IntMask tissue statistics for μ reporting

Outputs (under ``outputs/robustness_study/summary/``):
    - ``phase2_all_results.csv``                — per-(phantom, slice) metrics
    - ``tableB_physical_by_snr.csv|.tex``       — cohort × SNR aggregates
    - ``per_acquisition_noise_curves.csv``      — per-(phantom, SNR) δμ / r² / σ
                                                  (consumed by Figure 4 panels f, g)
    - ``fig4_physical_degradation.png`` etc.    — diagnostic plots

Usage:
    python experiments/robustness_study.py                  # full sweep
    RUN_INDEX=12 python experiments/robustness_study.py     # single (phantom, slice)
    SUMMARIZE=1 python experiments/robustness_study.py      # rebuild summaries
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.helpers import discover_despeckled_phantoms, load_phantom_bscan

from src.forward import create_forward_model
from src.inverse import (
    optimize,
    build_noise_floor_mask,
    subtract_noise_floor_linear,
)
from src.roi import load_intmask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CF_REFERENCE = {1.0: 4.523, 1.5: 5.914, 2.0: 5.878, 3.0: 7.742, 5.0: 9.346}


# SNR levels — match the 29-slice layout of inputs/IntralipidPhantoms_20250206_Noised/
# slice indices 0..27 correspond to integer SNR levels -3..24 dB
# slice index 28 is the noiseless baseline (snr_dB = inf)
SNR_LEVELS_DB = list(range(-3, 25))  # 28 integer dB steps
BASELINE_SLICE = 28
# Combined list including the baseline used by per-SNR aggregation loops
SNR_LEVELS = list(SNR_LEVELS_DB) + [float("inf")]

CONC_COLORS = {
    1.0: "tab:blue",
    1.5: "tab:orange",
    2.0: "tab:green",
    3.0: "tab:red",
    5.0: "tab:purple",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pre-noised phantom loader
# ---------------------------------------------------------------------------

NOISED_PHANTOM_DIR = os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206_Noised")


def load_noised_phantom_slice(phantom_basename: str, slice_idx: int) -> np.ndarray:
    """Load one slice from a pre-noised phantom TIFF (29 slices per file).

    Args:
        phantom_basename: Phantom filename or stem matching the noiseless
            acquisition. Both forms are accepted:
              "[OACProject][AgarInIL_1..0][02-06-2026_12-21-26]"
              "[OACProject][AgarInIL_1..0][02-06-2026_12-21-26]_DespeckledLogInt.tif"
            The "_DespeckledLogInt.tif" suffix is stripped if present, then
            "_DespeckledNoisyLogInt.tif" is appended to form the noised path.
        slice_idx: 0..27 for SNR -3..24 dB, 28 for the noiseless baseline.

    Returns:
        B-scan in dB scale, shape (500, 500), float32.
    """
    from PIL import Image
    stem = phantom_basename
    if stem.endswith("_DespeckledLogInt.tif"):
        stem = stem[:-len("_DespeckledLogInt.tif")]
    tiff_path = os.path.join(NOISED_PHANTOM_DIR, f"{stem}_DespeckledNoisyLogInt.tif")
    img = Image.open(tiff_path)
    img.seek(slice_idx)
    raw = np.array(img, dtype=np.float32)
    return raw * (130.0 / 65535.0)


def slice_to_snr_dB(slice_idx: int) -> float:
    """Map noised-TIFF slice index to the SNR label used downstream."""
    if slice_idx == BASELINE_SLICE:
        return float("inf")
    return float(SNR_LEVELS_DB[slice_idx])

NOISED_NOISEFLOOR_XLSX = os.path.join(NOISED_PHANTOM_DIR, "Noisefloor_For_SNR_Experiment.xlsx")
BASELINE_LABEL = "baseline"          # sentinel for the un-noised frame
_BASELINE_NOISEFLOOR_COL = 30        # AE: baseline noise floor (linear units)
_BASELINE_SNR_COL = 32               # AG: baseline SNR (dB)
_NOISEFLOOR_LOOKUP: dict = {}
_BASELINE_SNR_DB_LOOKUP: dict = {}


def _load_noisefloor_lookup() -> dict:
    """Parse Noisefloor_For_SNR_Experiment.xlsx once and cache.

    Populates:
        _NOISEFLOOR_LOOKUP[(phantom_idx_0based, snr_entry)] -> linear noise floor
        _BASELINE_SNR_DB_LOOKUP[phantom_idx_0based]         -> baseline SNR (dB)

    `snr_entry` is either an integer dB level in SNR_LEVELS_DB or BASELINE_LABEL.

    Excel layout (1-indexed volumes, blanks where SNR exceeds baseline):
        Row 0: header ("SNR (dB)")
        Row 1: ['Volume index', 'Filename', -3, ..., 24,
                <gap>, 'Baseline SNR' (col 30; actually noise floor linear),
                <gap>, 'True SNR' (col 32; baseline SNR dB)]
        Rows 2..11: one row per volume.
    """
    global _NOISEFLOOR_LOOKUP, _BASELINE_SNR_DB_LOOKUP
    if _NOISEFLOOR_LOOKUP:
        return _NOISEFLOOR_LOOKUP
    import openpyxl
    wb = openpyxl.load_workbook(NOISED_NOISEFLOOR_XLSX, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    header_row = rows[1]
    snr_cols = {}
    for ci, v in enumerate(header_row):
        if v is None or ci < 2 or ci >= _BASELINE_NOISEFLOOR_COL:
            continue
        try:
            snr_cols[int(round(float(v)))] = ci
        except (TypeError, ValueError):
            continue
    lookup: dict = {}
    baseline_snr: dict = {}
    for r in rows[2:]:
        if r[0] is None:
            continue
        try:
            vol_index = int(r[0])  # 1-indexed in spreadsheet
        except (TypeError, ValueError):
            continue
        p_idx = vol_index - 1
        for snr_dB, ci in snr_cols.items():
            val = r[ci]
            if val is None:
                continue
            lookup[(p_idx, int(snr_dB))] = float(val)
        if (len(r) > _BASELINE_NOISEFLOOR_COL
                and r[_BASELINE_NOISEFLOOR_COL] is not None):
            lookup[(p_idx, BASELINE_LABEL)] = float(r[_BASELINE_NOISEFLOOR_COL])
        if len(r) > _BASELINE_SNR_COL and r[_BASELINE_SNR_COL] is not None:
            baseline_snr[p_idx] = float(r[_BASELINE_SNR_COL])
    _NOISEFLOOR_LOOKUP = lookup
    _BASELINE_SNR_DB_LOOKUP = baseline_snr
    return lookup


def get_noised_phantom_noisefloor_dB(phantom_index: int, snr_entry) -> float:
    """Per-(phantom, SNR) noise floor in dB. xlsx stores linear; convert via 10·log10."""
    lookup = _load_noisefloor_lookup()
    key = (phantom_index,
           BASELINE_LABEL if snr_entry == BASELINE_LABEL else int(snr_entry))
    if key not in lookup:
        raise KeyError(
            f"Missing noise-floor entry for phantom {phantom_index}, "
            f"SNR {snr_entry!r}")
    return 10.0 * np.log10(float(lookup[key]))


def is_valid_noised_entry(phantom_index: int, snr_entry) -> bool:
    """True if (phantom, snr_entry) has a noise-floor entry in the xlsx.

    Combinations whose nominal SNR exceeds the phantom's baseline (native) SNR
    are blank in the xlsx and should be skipped. The baseline entry is always
    valid (it's the un-noised frame).
    """
    if snr_entry == BASELINE_LABEL:
        return True
    return (phantom_index, int(snr_entry)) in _load_noisefloor_lookup()



def _load_configs() -> tuple:
    """Load all three YAML configs from the configs directory."""
    cfg_dir = os.path.join(PHANTOM_ROOT, "configs")
    config = OmegaConf.load(os.path.join(cfg_dir, "config.yaml"))
    opt_config = OmegaConf.load(os.path.join(cfg_dir, "optimization_config_fast.yaml"))
    sc_config = OmegaConf.load(os.path.join(cfg_dir, "scatterer_config.yaml"))
    return config, opt_config, sc_config


def _make_ss_forward(config: OmegaConf, device: torch.device) -> nn.Module:
    """Create and return the SS forward model on the given device."""
    cfg_ss = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg_ss.forward_model_system.forward_model_type = "ss"
    model = create_forward_model(cfg_ss, device=str(device))
    model.to(device)
    return model


def _cleanup_gpu(*tensors) -> None:
    """Delete tensors, close matplotlib figures, and free GPU memory."""
    for t in tensors:
        if t is not None:
            del t
    plt.close("all")
    gc.collect()
    torch.cuda.empty_cache()


def _write_csv(path: str, rows: list, fieldnames: list) -> None:
    """Write a list of dicts to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def _write_latex_table(path: str, headers: list, rows: list, caption: str, label: str) -> None:
    """Write a LaTeX table from headers and rows (list of lists)."""
    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(v) for v in row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


def _compute_psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """PSNR between predicted and target B-scans within masked region."""
    if mask.sum() == 0:
        return float("nan")
    diff = (pred[mask > 0.5] - target[mask > 0.5]) ** 2
    mse = float(diff.mean())
    if mse < 1e-12:
        return float("inf")
    max_val = float(target[mask > 0.5].max() - target[mask > 0.5].min())
    return 10.0 * np.log10(max_val**2 / mse)


def _compute_ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Scalar SSIM between predicted and target within masked region."""
    if mask.sum() < 2:
        return float("nan")
    x = pred[mask > 0.5].astype(np.float64)
    y = target[mask > 0.5].astype(np.float64)
    L = float(y.max() - y.min())
    if L < 1e-12:
        return float("nan")
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x_sq, sigma_y_sq = x.var(), y.var()
    sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return float(ssim)


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------


# =========================================================================
# Phase 2: Physical phantom (noiseless baseline as reference)
# =========================================================================


def run_physical_optimization(
    phantom: dict,
    phantom_index: int,
    slice_idx: int,
    out_dir: str,
    config: OmegaConf,
    opt_config: OmegaConf,
    device: torch.device,
    mat_dir: str,
) -> dict:
    """Run one physical phantom noise robustness trial.

    Args:
        phantom: Dict with keys: path, concentration, replicate, basename.
        phantom_index: Tomogram index in [0, 9].
        bscan_index: B-scan index in [0, 511].
        snr_dB: Target SNR in dB (inf = no added noise).
        noise_seed: RNG seed.
        out_dir: Per-run output directory.
        config: System OmegaConf config.
        opt_config: Optimization OmegaConf config.
        device: Torch device.
        mat_dir: Directory containing IntMask.mat.

    Returns:
        Dict with run results.
    """
    os.makedirs(out_dir, exist_ok=True)

    prune_threshold = float(config.patient.prune_threshold_m_inv)
    D = int(config.forward_model_system.depth_samples)
    A = int(config.forward_model_system.a_lines)
    snr_dB = slice_to_snr_dB(slice_idx)
    bscan_index = int(os.environ.get("BSCAN_INDEX", 256))

    snr_entry = BASELINE_LABEL if slice_idx == BASELINE_SLICE else int(round(snr_dB))
    try:
        noise_floor_dB = get_noised_phantom_noisefloor_dB(phantom_index, snr_entry)
    except KeyError:
        noise_floor_dB = float(config.forward_model_system.noise_floor_dB)
        print(f"  [warn] no xlsx entry for phantom={phantom_index} snr={snr_entry}; "
              f"falling back to global noise_floor={noise_floor_dB} dB")
    print(f"  Per-condition noise floor: {noise_floor_dB:.2f} dB")

    # IntMask for tissue stats — depends on phantom + B-scan (independent of SNR)
    intmask_np = load_intmask(phantom_index, bscan_index, mat_dir=mat_dir)

    # Load the pre-noised B-scan slice. Noise was injected externally and saved
    # into inputs/IntralipidPhantoms_20250206_Noised/; slice 28 is the baseline.
    I_noisy_np = load_noised_phantom_slice(phantom["basename"], slice_idx)

    # N(z) subtract on the loaded image so the optimizer sees the same correction
    # the noiseless physical-validation pipeline applies (matches 7b conventions).
    I_corrected_dB = subtract_noise_floor_linear(
        torch.tensor(I_noisy_np, dtype=torch.float32, device=device), noise_floor_dB
    )

    I_target_dB = I_corrected_dB
    noise_floor_mask = build_noise_floor_mask(I_target_dB, noise_floor_dB, device)

    # Deterministic seed matching 7b: 42 + phantom_index * 512 + bscan_index
    seed = 42 + phantom_index * 512 + bscan_index
    torch.manual_seed(seed)
    mu_init_data = torch.empty(D, A, dtype=torch.float32, device=device).uniform_(0.0, 10.0)
    alpha_init_data = torch.ones(D, A, dtype=torch.float32, device=device)

    forward_model = _make_ss_forward(config, device)

    run_opt_config = OmegaConf.create(OmegaConf.to_container(opt_config, resolve=True))
    run_opt_config.optimization.panel_save_interval = 99999

    mu = nn.Parameter(mu_init_data.clone())
    alpha = nn.Parameter(alpha_init_data.clone())
    forward_model.gain_dB.fill_(0.0)

    t0 = time.time()
    mu_best, alpha_best, metrics, seq_steps, seq_imgs, mu_init, alpha_init = optimize(
        forward_model, mu, alpha, I_target_dB, run_opt_config,
        prune_threshold, device, out_dir,
        noise_floor_mask=noise_floor_mask,
        recompute_gain_mse_optimal_each_iter=True,
    )
    elapsed = time.time() - t0

    mu_np = mu_best.detach().cpu().numpy()
    np.save(os.path.join(out_dir, "mu_final_mm.npy"), mu_np)

    # Save predicted B-scan for later PSNR/SSIM comparison vs noiseless baseline
    with torch.no_grad():
        pred_dB_tensor = forward_model(mu_best, alpha_best, noise_mask=noise_floor_mask)
        pred_dB_np = pred_dB_tensor.cpu().numpy()
    np.save(os.path.join(out_dir, "pred_dB.npy"), pred_dB_np)

    # IntMask-filtered tissue mu stats
    nf_mask_np = noise_floor_mask.cpu().numpy() > 0.5
    combined = intmask_np & nf_mask_np
    mu_tissue = mu_np[combined]

    mu_tissue_mean = float(mu_tissue.mean()) if len(mu_tissue) > 0 else float("nan")
    mu_tissue_std = float(mu_tissue.std()) if len(mu_tissue) > 0 else float("nan")

    mse_arr = np.array(metrics["mse"])
    best_idx = int(np.argmin(mse_arr))
    mse_final = float(mse_arr[best_idx])

    # NOTE: PSNR/SSIM vs noiseless baseline computed during summarization
    # (baseline might be on a different GPU / not yet available)
    np.savez(
        os.path.join(out_dir, "roi_stats.npz"),
        mu_tissue_mean=mu_tissue_mean,
        mu_tissue_std=mu_tissue_std,
        mse_final=mse_final,
        time_sec=elapsed,
        snr_dB=snr_dB if np.isfinite(snr_dB) else np.inf,
        phantom_index=phantom_index,
        concentration=phantom["concentration"],
    )

    result = dict(
        phantom_index=phantom_index,
        concentration=phantom["concentration"],
        replicate=phantom["replicate"],
        snr_dB=snr_dB,
        mu_tissue_mean=mu_tissue_mean,
        mu_tissue_std=mu_tissue_std,
        mse_final=mse_final,
        time_sec=elapsed,
    )

    _cleanup_gpu(
        mu, alpha, mu_best, alpha_best, mu_init, alpha_init,
        I_target_dB, noise_floor_mask, pred_dB_tensor,
    )
    del forward_model, run_opt_config, metrics, seq_steps, seq_imgs

    return result


def run_phase2(
    phantoms: list,
    config: OmegaConf,
    opt_config: OmegaConf,
    device: torch.device,
    out_base: str,
    mat_dir: str,
    run_index: Optional[int],
) -> None:
    """Run optimization on each pre-noised slice (10 phantoms × 29 slices = 290 runs)."""
    phase_dir = os.path.join(out_base, "physical")
    os.makedirs(phase_dir, exist_ok=True)

    n_phantoms = len(phantoms)
    n_slices = len(SNR_LEVELS_DB) + 1  # 28 SNR levels + 1 baseline

    def _decode(ri):
        slice_idx = ri % n_slices
        phantom_idx = ri // n_slices
        return phantom_idx, slice_idx

    def _iter_runs():
        if run_index is not None:
            yield _decode(run_index)
            return
        for pidx in range(n_phantoms):
            for slice_idx in range(n_slices):
                yield pidx, slice_idx

    for pidx, slice_idx in _iter_runs():
        if pidx >= n_phantoms or slice_idx >= n_slices:
            continue

        snr_dB = slice_to_snr_dB(slice_idx)
        snr_label = "baseline" if not np.isfinite(snr_dB) else f"{int(snr_dB)}dB"
        run_dir = os.path.join(phase_dir, f"phantom_{pidx}", f"snr_{snr_label}")

        snr_entry = (BASELINE_LABEL if slice_idx == BASELINE_SLICE
                     else int(round(snr_dB)))
        if not is_valid_noised_entry(pidx, snr_entry):
            print(f"  [Phase2] Skip p={pidx} snr={snr_label} "
                  f"(no xlsx entry — nominal SNR exceeds phantom baseline)")
            continue

        if os.path.isfile(os.path.join(run_dir, "roi_stats.npz")):
            print(f"  [Phase2] Skip p={pidx} snr={snr_label} (done)")
            continue

        print(f"\n[Phase2] phantom={pidx} snr={snr_label} slice={slice_idx}")
        run_physical_optimization(
            phantoms[pidx], pidx, slice_idx,
            run_dir, config, opt_config, device, mat_dir,
        )


# =========================================================================
# Summary: Phase 2 (Physical)
# =========================================================================


def _collect_physical_results(out_base: str, phantoms: list) -> list:
    """Scan physical output directories (one run per (phantom, slice))."""
    phase_dir = os.path.join(out_base, "physical")
    rows = []
    n_phantoms = len(phantoms)
    n_slices = len(SNR_LEVELS_DB) + 1
    for pidx in range(n_phantoms):
        for slice_idx in range(n_slices):
            snr_dB = slice_to_snr_dB(slice_idx)
            snr_label = "baseline" if not np.isfinite(snr_dB) else f"{int(snr_dB)}dB"
            stats_path = os.path.join(
                phase_dir, f"phantom_{pidx}", f"snr_{snr_label}", "roi_stats.npz",
            )
            if not os.path.isfile(stats_path):
                continue
            d = np.load(stats_path, allow_pickle=True)
            rows.append(dict(
                phantom_index=pidx,
                concentration=phantoms[pidx]["concentration"],
                replicate=phantoms[pidx]["replicate"],
                slice_idx=slice_idx,
                snr_dB=snr_dB,
                snr_label=snr_label,
                mu_tissue_mean=float(d["mu_tissue_mean"]),
                mu_tissue_std=float(d["mu_tissue_std"]),
                mse_final=float(d["mse_final"]),
                time_sec=float(d["time_sec"]),
            ))
    return rows


def _load_baseline_mu_and_pred(out_base: str, pidx: int) -> tuple:
    """Load noiseless (SNR=inf, seed=0) baseline mu and pred_dB for a phantom.

    Returns:
        (baseline_mu_mean, baseline_pred_dB_np) or (nan, None) if not found.
    """
    phase_dir = os.path.join(out_base, "physical")
    baseline_dir = os.path.join(phase_dir, f"phantom_{pidx}", "snr_baseline")
    stats_path = os.path.join(baseline_dir, "roi_stats.npz")
    pred_path = os.path.join(baseline_dir, "pred_dB.npy")

    if not os.path.isfile(stats_path):
        return float("nan"), None
    d = np.load(stats_path)
    mu_baseline = float(d["mu_tissue_mean"])

    pred_dB = np.load(pred_path) if os.path.isfile(pred_path) else None
    return mu_baseline, pred_dB



def write_per_acquisition_curves(out_base: str, phantoms: list, mat_dir: str) -> None:
    """Build per-phantom × per-SNR curves and write per_acquisition_noise_curves.csv.

    For each (phantom, slice_idx) pair, compute three structural-fidelity metrics
    against that phantom's noiseless baseline μ map (slice 28):

      - delta_pct      = |mu_tissue_mean - mu_baseline| / mu_baseline × 100
      - r2             = pixel-level R² of recovered μ map vs baseline μ map,
                         restricted to the IntMask tissue region
      - sigma_mm_inv   = spatial std of recovered μ over the IntMask region

    The CSV layout matches what the Figure 4 panel-(g) and panel-(f) cells expect:
      phantom_index, label, concentration_pct, focal_plane, line_color,
      snr_dB, baseline_snr_dB, delta_pct, r2, sigma_mm_inv
    """
    summary_dir = os.path.join(out_base, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    phase_dir = os.path.join(out_base, "physical")
    bscan_index = int(os.environ.get("BSCAN_INDEX", 256))
    n_slices = len(SNR_LEVELS_DB) + 1

    # Per-phantom baseline SNRs (acquisition properties, not optimizer outputs).
    BASELINE_SNR_LOOKUP = {0: 19.75, 1: 17.17, 2: 21.31, 3: 19.53, 4: 21.35,
                           5: 18.70, 6: 22.63, 7: 21.52, 8: 24.68, 9: 23.16}
    LINE_COLOR = {1.0: "tab:blue", 1.5: "tab:orange", 2.0: "tab:green",
                  3.0: "tab:red", 5.0: "tab:purple"}

    rows = []
    for pidx, phantom in enumerate(phantoms):
        baseline_dir = os.path.join(phase_dir, f"phantom_{pidx}", "snr_baseline")
        baseline_mu_path    = os.path.join(baseline_dir, "mu_final_mm.npy")
        baseline_stats_path = os.path.join(baseline_dir, "roi_stats.npz")
        if not (os.path.isfile(baseline_mu_path) and os.path.isfile(baseline_stats_path)):
            print(f"  [per-acq] Skip phantom {pidx}: missing baseline files")
            continue
        mu_base_map = np.load(baseline_mu_path)
        mu_base_tissue_mean = float(np.load(baseline_stats_path)["mu_tissue_mean"])
        intmask_np = load_intmask(pidx, bscan_index, mat_dir=mat_dir)
        if intmask_np.sum() < 10 or not (mu_base_tissue_mean > 0):
            print(f"  [per-acq] Skip phantom {pidx}: degenerate mask or baseline")
            continue

        conc = phantom["concentration"]
        rep  = int(phantom["replicate"])
        label = f"{conc:.1f}%_F{rep}"
        line_color = LINE_COLOR.get(conc, "black")
        baseline_snr = BASELINE_SNR_LOOKUP.get(pidx, float("nan"))

        for slice_idx in range(n_slices - 1):  # finite-SNR slices only
            snr_dB = slice_to_snr_dB(slice_idx)
            if not np.isfinite(snr_dB):
                continue
            snr_label = f"{int(snr_dB)}dB"
            run_dir = os.path.join(phase_dir, f"phantom_{pidx}", f"snr_{snr_label}")
            mu_path    = os.path.join(run_dir, "mu_final_mm.npy")
            stats_path = os.path.join(run_dir, "roi_stats.npz")
            if not (os.path.isfile(mu_path) and os.path.isfile(stats_path)):
                continue
            mu_noisy_map = np.load(mu_path)
            mu_noisy_tissue_mean = float(np.load(stats_path)["mu_tissue_mean"])

            # Δμ vs noiseless baseline — uses the per-run mu_tissue_mean (matches
            # what the cohort table B reports for consistency)
            delta_pct = abs(mu_noisy_tissue_mean - mu_base_tissue_mean) / mu_base_tissue_mean * 100.0

            # R² between recovered and baseline μ maps on the IntMask tissue region
            # — Pearson r² (clamped to [0, 1]) so curves are monotone-shaped.
            t_base  = mu_base_map[intmask_np]
            t_noisy = mu_noisy_map[intmask_np]
            if t_base.size > 1 and t_base.std() > 0 and t_noisy.std() > 0:
                r = np.corrcoef(t_base, t_noisy)[0, 1]
                r2 = float(r * r) if np.isfinite(r) else float("nan")
            else:
                r2 = float("nan")

            # σ_μ = spatial std of recovery deviation from noiseless baseline (→0 as SNR rises)
            sigma = float((mu_noisy_map[intmask_np] - mu_base_map[intmask_np]).std())

            rows.append(dict(
                phantom_index=pidx,
                label=label,
                concentration_pct=conc,
                focal_plane=rep,
                line_color=line_color,
                snr_dB=snr_dB,
                baseline_snr_dB=baseline_snr,
                delta_pct=delta_pct,
                r2=r2,
                sigma_mm_inv=sigma,
            ))

    if rows:
        csv_path = os.path.join(summary_dir, "per_acquisition_noise_curves.csv")
        _write_csv(csv_path, rows, list(rows[0].keys()))
        print(f"  Wrote {csv_path} ({len(rows)} rows)")

def summarize_phase2(out_base: str, phantoms: list) -> None:
    """Generate summary tables and figures for Phase 2 (physical phantom)."""
    summary_dir = os.path.join(out_base, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    phase_dir = os.path.join(out_base, "physical")

    # Per-acquisition (phantom × SNR) curves used by Figure 4 panels (f) and (g)
    mat_dir = os.path.join(PHANTOM_ROOT, "matlab_CF_scripts_results")
    write_per_acquisition_curves(out_base, phantoms, mat_dir)

    all_rows = _collect_physical_results(out_base, phantoms)
    print(f"  Phase 2: {len(all_rows)} completed runs")
    if not all_rows:
        print("  No physical results found.")
        return

    n_phantoms = len(phantoms)
    concentrations = sorted(set(p["concentration"] for p in phantoms))

    # Load noiseless baselines (SNR=inf, seed=0) for each phantom
    baselines = {}  # pidx -> mu_baseline
    baseline_preds = {}  # pidx -> pred_dB_np
    for pidx in range(n_phantoms):
        mu_base, pred_base = _load_baseline_mu_and_pred(out_base, pidx)
        baselines[pidx] = mu_base
        baseline_preds[pidx] = pred_base

    n_baselines = sum(1 for v in baselines.values() if not np.isnan(v))
    print(f"  Loaded {n_baselines}/{n_phantoms} noiseless baselines")

    # Compute metrics relative to noiseless baseline
    enriched_rows = []
    for r in all_rows:
        pidx = r["phantom_index"]
        mu_base = baselines.get(pidx, float("nan"))
        mu_noisy = r["mu_tissue_mean"]

        # Delta mu relative to noiseless baseline
        if not np.isnan(mu_base) and not np.isnan(mu_noisy) and mu_base > 0:
            delta_mu = mu_noisy - mu_base
            delta_pct = abs(delta_mu) / mu_base * 100.0
        else:
            delta_mu = float("nan")
            delta_pct = float("nan")

        # PSNR/SSIM vs baseline pred_dB
        pred_path = os.path.join(
            phase_dir, f"phantom_{pidx}", f"snr_{r['snr_label']}", "pred_dB.npy",
        )
        psnr_vs_base = float("nan")
        ssim_vs_base = float("nan")
        if baseline_preds[pidx] is not None and os.path.isfile(pred_path):
            pred_noisy = np.load(pred_path)
            pred_base = baseline_preds[pidx]
            # Use all pixels with nonzero signal in baseline as mask
            mask = (pred_base > 0).astype(np.float32)
            psnr_vs_base = _compute_psnr(pred_noisy, pred_base, mask)
            ssim_vs_base = _compute_ssim(pred_noisy, pred_base, mask)

        enriched_rows.append(dict(
            **r,
            mu_baseline=mu_base,
            delta_mu=delta_mu,
            delta_pct=delta_pct,
            psnr_vs_baseline=psnr_vs_base,
            ssim_vs_baseline=ssim_vs_base,
        ))

    # --- Raw CSV ---
    _write_csv(
        os.path.join(summary_dir, "phase2_all_results.csv"),
        enriched_rows, list(enriched_rows[0].keys()),
    )

    # --- Table B: aggregated by SNR ---
    table_csv = []
    latex_data = []
    for snr_dB in SNR_LEVELS:
        snr_str = "baseline" if not np.isfinite(snr_dB) else str(int(snr_dB))
        subset = [r for r in enriched_rows if r["snr_dB"] == snr_dB]
        if not subset:
            continue

        deltas = np.array([r["delta_pct"] for r in subset])
        psnrs = np.array([r["psnr_vs_baseline"] for r in subset])
        ssims = np.array([r["ssim_vs_baseline"] for r in subset])
        mus = np.array([r["mu_tissue_mean"] for r in subset])
        mses = np.array([r["mse_final"] for r in subset])
        deltas = deltas[~np.isnan(deltas)]
        psnrs = psnrs[~np.isnan(psnrs)]
        ssims = ssims[~np.isnan(ssims)]
        mus = mus[~np.isnan(mus)]

        is_baseline = not np.isfinite(snr_dB)

        row = dict(
            snr_dB=snr_str,
            n_runs=len(subset),
            delta_pct_mean=0.0 if is_baseline else (
                round(float(np.mean(deltas)), 2) if len(deltas) > 0 else "nan"),
            delta_pct_std="--" if is_baseline else (
                round(float(np.std(deltas)), 2) if len(deltas) > 0 else "nan"),
            psnr_vs_baseline="ref" if is_baseline else (
                round(float(np.mean(psnrs)), 2) if len(psnrs) > 0 else "nan"),
            ssim_vs_baseline="ref" if is_baseline else (
                round(float(np.mean(ssims)), 4) if len(ssims) > 0 else "nan"),
            mu_mean=round(float(np.mean(mus)), 3) if len(mus) > 0 else "nan",
            mu_std_seed=round(float(np.std(mus)), 3) if len(mus) > 0 else "nan",
            mse_mean=f"{float(np.mean(mses)):.4e}",
        )
        table_csv.append(row)

        latex_data.append([
            snr_str,
            "0 (ref)" if is_baseline else (
                f"{float(np.mean(deltas)):.1f}$\\pm${float(np.std(deltas)):.1f}" if len(deltas) > 0 else "--"),
            "ref" if is_baseline else (
                f"{float(np.mean(psnrs)):.1f}" if len(psnrs) > 0 else "--"),
            "ref" if is_baseline else (
                f"{float(np.mean(ssims)):.3f}" if len(ssims) > 0 else "--"),
            f"{float(np.mean(mus)):.2f}$\\pm${float(np.std(mus)):.2f}" if len(mus) > 0 else "--",
        ])

    if table_csv:
        _write_csv(os.path.join(summary_dir, "tableB_physical_by_snr.csv"),
                   table_csv, list(table_csv[0].keys()))
    if latex_data:
        _write_latex_table(
            os.path.join(summary_dir, "tableB_physical_by_snr.tex"),
            ["SNR (dB)", r"$\Delta\mu$ vs baseline (\%)", "PSNR (dB)", "SSIM",
             r"$\bar{\mu} \pm \sigma$ (mm$^{-1}$)"],
            latex_data,
            caption="Physical phantom noise robustness: degradation vs noiseless baseline. "
                    "10 phantoms, 5 noise seeds, SS (Beer-Lambert) model.",
            label="tab:physical_noise",
        )

    # --- Table B2: per-concentration ---
    table_conc_csv = []
    latex_conc_data = []
    for snr_dB in SNR_LEVELS:
        snr_str = "baseline" if not np.isfinite(snr_dB) else str(int(snr_dB))
        is_baseline = not np.isfinite(snr_dB)
        row_csv = dict(snr_dB=snr_str)
        latex_row = [snr_str]
        for conc in concentrations:
            subset = [r for r in enriched_rows
                      if r["snr_dB"] == snr_dB and r["concentration"] == conc]
            mus = [r["mu_tissue_mean"] for r in subset if not np.isnan(r["mu_tissue_mean"])]
            deltas = [r["delta_pct"] for r in subset if not np.isnan(r["delta_pct"])]
            if mus:
                mu_m = float(np.mean(mus))
                delta_m = 0.0 if is_baseline else (float(np.mean(deltas)) if deltas else float("nan"))
                row_csv[f"{conc:.1f}pct_mu"] = round(mu_m, 2)
                row_csv[f"{conc:.1f}pct_delta"] = round(delta_m, 1)
                if is_baseline:
                    latex_row.append(f"{mu_m:.2f} (ref)")
                else:
                    latex_row.append(f"{mu_m:.2f} ($\\Delta${delta_m:.1f}\\%)")
            else:
                row_csv[f"{conc:.1f}pct_mu"] = "nan"
                row_csv[f"{conc:.1f}pct_delta"] = "nan"
                latex_row.append("--")
        table_conc_csv.append(row_csv)
        latex_conc_data.append(latex_row)

    if table_conc_csv:
        _write_csv(os.path.join(summary_dir, "tableB2_physical_per_conc.csv"),
                   table_conc_csv, list(table_conc_csv[0].keys()))
    if latex_conc_data:
        conc_headers = [f"{c:.1f}\\% IL" for c in concentrations]
        _write_latex_table(
            os.path.join(summary_dir, "tableB2_physical_per_conc.tex"),
            ["SNR (dB)"] + conc_headers, latex_conc_data,
            caption="Per-concentration attenuation recovery under noise "
                    "(physical phantoms). $\\Delta$ = degradation vs noiseless baseline.",
            label="tab:physical_per_conc",
        )

    # --- Figure 4: degradation vs SNR ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x_snr = np.array([45.0 if not np.isfinite(s) else float(s) for s in SNR_LEVELS])

    delta_means, delta_stds = [], []
    for snr_dB in SNR_LEVELS:
        ds = [r["delta_pct"] for r in enriched_rows
              if r["snr_dB"] == snr_dB and not np.isnan(r["delta_pct"])]
        delta_means.append(float(np.mean(ds)) if ds else float("nan"))
        delta_stds.append(float(np.std(ds)) if ds else float("nan"))
    delta_means = np.array(delta_means)
    delta_stds = np.array(delta_stds)
    v = ~np.isnan(delta_means)
    ax.fill_between(x_snr[v], delta_means[v] - delta_stds[v],
                     delta_means[v] + delta_stds[v], alpha=0.15, color="gray")
    ax.plot(x_snr[v], delta_means[v], "-ko", lw=2, ms=6, label="All phantoms")

    for conc in concentrations:
        dm = []
        for snr_dB in SNR_LEVELS:
            ds = [r["delta_pct"] for r in enriched_rows
                  if r["snr_dB"] == snr_dB and r["concentration"] == conc
                  and not np.isnan(r["delta_pct"])]
            dm.append(float(np.mean(ds)) if ds else float("nan"))
        dm = np.array(dm)
        vc = ~np.isnan(dm)
        ax.plot(x_snr[vc], dm[vc], "-o", color=CONC_COLORS.get(conc, "gray"),
                lw=1.2, ms=4, label=f"{conc:.1f}% IL")

    ax.set_xlabel("SNR (dB)  [45 = no noise]", fontsize=12)
    ax.set_ylabel(r"$|\Delta\mu|$ vs baseline (%)", fontsize=12)
    ax.set_title("Physical Phantom: Degradation vs Noiseless Baseline",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "fig4_physical_degradation.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: fig4_physical_degradation.png")

    # --- Figure 5: PSNR/SSIM vs SNR (physical) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    psnr_m, ssim_m = [], []
    for snr_dB in SNR_LEVELS:
        if not np.isfinite(snr_dB):
            psnr_m.append(float("nan"))  # baseline = self
            ssim_m.append(float("nan"))
            continue
        s = [r for r in enriched_rows if r["snr_dB"] == snr_dB]
        p = [r["psnr_vs_baseline"] for r in s if not np.isnan(r["psnr_vs_baseline"])]
        ss = [r["ssim_vs_baseline"] for r in s if not np.isnan(r["ssim_vs_baseline"])]
        psnr_m.append(float(np.mean(p)) if p else float("nan"))
        ssim_m.append(float(np.mean(ss)) if ss else float("nan"))
    psnr_m = np.array(psnr_m)
    ssim_m = np.array(ssim_m)
    v1 = ~np.isnan(psnr_m)
    v2 = ~np.isnan(ssim_m)
    ax1.plot(x_snr[v1], psnr_m[v1], "-o", color="#2171b5", lw=2, ms=6, label="PSNR")
    ax2.plot(x_snr[v2], ssim_m[v2], "--s", color="#d94701", lw=1.5, ms=5, label="SSIM")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("PSNR vs baseline (dB)", color="#2171b5")
    ax2.set_ylabel("SSIM vs baseline", color="#d94701")
    ax1.set_title("Physical Phantom: Reconstruction Quality vs Noise",
                  fontsize=13, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "fig5_physical_psnr_ssim.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: fig5_physical_psnr_ssim.png")

    # --- Print compact summary ---
    print("\n" + "=" * 70)
    print("PHYSICAL PHANTOM: DEGRADATION vs NOISELESS BASELINE")
    print("=" * 70)
    for snr_dB in SNR_LEVELS:
        snr_str = "baseline" if not np.isfinite(snr_dB) else f"{int(snr_dB):3d}"
        subset = [r for r in enriched_rows if r["snr_dB"] == snr_dB]
        if not subset:
            continue
        deltas = [r["delta_pct"] for r in subset if not np.isnan(r["delta_pct"])]
        mus = [r["mu_tissue_mean"] for r in subset if not np.isnan(r["mu_tissue_mean"])]
        if mus:
            d_str = f"delta={np.mean(deltas):.1f}±{np.std(deltas):.1f}%" if deltas else "ref"
            print(f"  SNR={snr_str} dB | mu={np.mean(mus):.2f}±{np.std(mus):.2f} | "
                  f"{d_str} | n={len(subset)}")


# =========================================================================
# Noise level visualization
# =========================================================================


# =========================================================================
# Main
# =========================================================================

    _stage_to_figure_paper_4(summary_dir)


def main() -> None:
    """Entry point for the noise robustness study (physical phantoms only)."""
    print("=== Noise Robustness Study: Physical Phantoms (pre-noised) ===")

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
    out_base = os.path.join(PHANTOM_ROOT, "outputs", "robustness_study")
    os.makedirs(out_base, exist_ok=True)

    n_runs = len(phantoms) * (len(SNR_LEVELS_DB) + 1)
    print(f"\nPhysical phantoms: {len(phantoms)} phantoms × {len(SNR_LEVELS_DB) + 1} slices "
          f"(28 SNR + 1 baseline) = {n_runs} runs")
    print(f"Output: {out_base}")

    if summarize_only:
        summarize_phase2(out_base, phantoms)
        return

    run_phase2(phantoms, config, opt_config, device, out_base, mat_dir, run_index)
    if run_index is None:
        summarize_phase2(out_base, phantoms)

    print(f"\nDone. Outputs under: {out_base}")



def _stage_to_figure_paper_4(summary_dir):
    """Stage the robustness-study subset the Figure 4 notebook reads.

    Copies ``tableB_physical_by_snr.csv`` (cohort × SNR table — feeds panel f's
    Δμ column) and ``per_acquisition_noise_curves.csv`` (per-(phantom, SNR)
    curves — feeds panel g and panel f's R²/σ aggregates) into
    ``figures/data/figure4_identifiability_robustness/``.
    """
    import shutil
    fig_paper_4 = os.path.join(PHANTOM_ROOT, "figures", "data", "figure4_identifiability_robustness")
    os.makedirs(fig_paper_4, exist_ok=True)
    n = 0
    for fname in ("tableB_physical_by_snr.csv", "per_acquisition_noise_curves.csv"):
        src = os.path.join(summary_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(fig_paper_4, fname))
            n += 1
    print(f"  [stage \u2192 figure4_identifiability_robustness] {n} robustness CSVs into {fig_paper_4}")


if __name__ == "__main__":
    main()
