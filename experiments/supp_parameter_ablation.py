#!/usr/bin/env python
"""Parameter ablation study (Figures S1 + S2).

One-parameter-at-a-time sensitivity analysis on phantom validation. Each phase
sweeps one parameter while holding all others at their baseline; for every
condition, the standard phantom-validation pipeline is run on B-scan 256 of all
10 acquisitions (5 IL concentrations × 2 focal positions) and per-phantom μ
recovery metrics are aggregated into a per-phase CSV.

Phases:
    1  focal_plane_position_pixels    \\
    2  rayleigh_range_pixels           |  System / physics  → Figure S1
    3  noise_floor_dB                  |
    4  axial_pixel_size_mm            /

    5  num_iterations                  \\
    6  learning_rate                    |  Optimizer        → Figure S2
    7  mu_tv_weight                     |
    8  loss_scale                      /

Outputs:
    outputs/parameter_ablation/phase_NN_<axis>/
        cond_<value>/<acquisition_label>/bscan_256/{ss}/...
        phase_results.csv      (per-phantom × per-condition aggregate)
        phase_results.npz      (numpy mirror of the CSV)
        sensitivity_curve.png  (diagnostic plot)

After completion, _stage_to_figure_paper_S12() copies each phase_results.csv
into the appropriate figures/data/figureS3{a_parameter,b_optimizer}_sensitivity/parameter_ablation/
subdirectory so the figure notebooks render without re-running the sweep.

Usage:
    python experiments/supp_parameter_ablation.py                 # run all 8 phases
    PHASE=4 python experiments/supp_parameter_ablation.py         # one phase only
    SUMMARIZE=1 python experiments/supp_parameter_ablation.py     # rebuild CSVs + stage
    STAGE_ONLY=1 python experiments/supp_parameter_ablation.py    # just stage existing CSVs
"""

import csv
import gc
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

# Phases producing data for Figure S1 (system / physics parameters)
S1_PHASE_IDS = (1, 2, 3, 4)
# Phases producing data for Figure S2 (optimizer hyperparameters)
S2_PHASE_IDS = (5, 6, 7, 8)

PHASES = {
    # System parameters (FigureS1)
    1: dict(name="focal_plane_position_pixels",
            config="forward_model_system.focal_plane_position_pixels",
            values=[200.0, 215.0, 231.0, 250.0, 261.0, 275.0, 300.0],
            baseline=231.0),
    2: dict(name="rayleigh_range_pixels",
            config="forward_model_system.rayleigh_range_pixels",
            values=[80.0, 100.0, 115.0, 131.5, 150.0, 170.0, 200.0],
            baseline=131.5),
    3: dict(name="noise_floor_dB",
            config="forward_model_system.noise_floor_dB",
            values=[70.0, 73.0, 76.0, 79.0, 82.0, 85.0, 88.0, 90.0],
            baseline=79.0),
    4: dict(name="axial_pixel_size_mm",
            config="forward_model_system.axial_pixel_size_mm",
            values=[0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010],
            baseline=0.004),
    # Optimizer parameters (FigureS2)
    5: dict(name="num_iterations",
            config="optimization.num_iterations",
            values=[1000, 2500, 5000, 10000, 15000, 20000, 30000],
            baseline=15000),
    6: dict(name="learning_rate",
            config="optimization.learning_rate",
            values=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            baseline=1e-3),
    7: dict(name="mu_tv_weight",
            config="optimization.mu_tv_weight",
            values=[0.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0],
            baseline=1000.0),
    8: dict(name="loss_scale",
            config="optimization.loss_scale",
            values=[100.0, 300.0, 1000.0, 3000.0, 10000.0],
            baseline=1000.0),
}

OUT_BASE = os.path.join(PHANTOM_ROOT, "outputs", "parameter_ablation")
BSCAN_INDEX = int(os.environ.get("BSCAN_INDEX", 256))


# ---------------------------------------------------------------------------
# Run a single phase
# ---------------------------------------------------------------------------

def run_phase(phase_id: int) -> None:
    """Run one ablation phase by invoking phantom_validation_ss_or_ms.py with
    the parameter override applied to a temporary config file."""
    phase = PHASES[phase_id]
    phase_dir = os.path.join(OUT_BASE, f"phase_{phase_id:02d}_{phase['name']}")
    os.makedirs(phase_dir, exist_ok=True)
    base_cfg = OmegaConf.load(os.path.join(PHANTOM_ROOT, "configs", "config.yaml"))
    base_opt = OmegaConf.load(os.path.join(PHANTOM_ROOT, "configs", "optimization_config_fast.yaml"))

    cfg_root, cfg_key = phase["config"].split(".", 1)

    for v in phase["values"]:
        cond_tag = _format_cond_tag(v)
        cond_dir = os.path.join(phase_dir, f"cond_{cond_tag}")
        os.makedirs(cond_dir, exist_ok=True)
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        opt = OmegaConf.create(OmegaConf.to_container(base_opt, resolve=True))
        target = cfg if cfg_root == "forward_model_system" else opt
        OmegaConf.update(target, phase["config"], v)

        # Persist the per-condition overrides so the sub-process picks them up
        cfg_path = os.path.join(cond_dir, "config.yaml")
        opt_path = os.path.join(cond_dir, "optimization_config.yaml")
        OmegaConf.save(cfg, cfg_path)
        OmegaConf.save(opt, opt_path)

        env = os.environ.copy()
        env["OUT_SUBDIR"]      = os.path.relpath(cond_dir, os.path.join(PHANTOM_ROOT, "outputs"))
        env["BSCAN_INDEX"]     = str(BSCAN_INDEX)
        env["START_FRAME"]     = str(BSCAN_INDEX)
        env["END_FRAME"]       = str(BSCAN_INDEX + 1)
        env["CONFIG_PATH"]     = cfg_path
        env["OPT_CONFIG_PATH"] = opt_path

        t0 = time.time()
        print(f"\n[phase {phase_id} {phase['name']}] cond={cond_tag} (param={v})")
        subprocess.run(
            [sys.executable, os.path.join(PHANTOM_ROOT, "experiments",
                                           "phantom_validation_ss_or_ms.py")],
            env=env, check=True,
        )
        print(f"  cond {cond_tag}: {time.time() - t0:.1f}s")

    summarize_phase(phase_id)


def summarize_phase(phase_id: int) -> None:
    """Aggregate per-frame intmask μ across (phantom × condition) into a CSV."""
    phase = PHASES[phase_id]
    phase_dir = os.path.join(OUT_BASE, f"phase_{phase_id:02d}_{phase['name']}")

    rows = []
    for v in phase["values"]:
        cond_tag = _format_cond_tag(v)
        cond_dir = os.path.join(phase_dir, f"cond_{cond_tag}")
        if not os.path.isdir(cond_dir):
            continue
        for label_dir in sorted(_list_phantom_dirs(cond_dir)):
            bscan_dir = os.path.join(cond_dir, label_dir,
                                      f"bscan_{BSCAN_INDEX:03d}")
            mu_npy    = os.path.join(bscan_dir, "intmask_mu.npy")
            stats_npz = os.path.join(bscan_dir, "intmask_stats.npz")
            if not os.path.isfile(mu_npy):
                continue
            tissue = np.load(mu_npy)
            if tissue.size == 0:
                continue
            # Per-frame optimization stats (populated by phantom_validation_ss_or_ms.py).
            best_mse = frame_time = gain_dB = float("nan")
            if os.path.isfile(stats_npz):
                with np.load(stats_npz) as st:
                    best_mse   = float(st["best_mse"])
                    frame_time = float(st["frame_time"])
                    gain_dB    = float(st["gain_dB"])
            conc, rep = _parse_label(label_dir)
            rows.append(dict(
                phase_id=phase_id,
                axis_name=phase["name"],
                param_value=float(v),
                condition_tag=cond_tag,
                phantom_index=_phantom_index(conc, rep),
                concentration=conc,
                replicate=rep,
                mu_mean=float(tissue.mean()),
                mu_std=float(tissue.std()),
                mu_median=float(np.median(tissue)),
                mu_err_pct=_err_pct(conc, float(tissue.mean())),
                best_mse=best_mse,
                frame_time=frame_time,
                gain_dB=gain_dB,
                n_tissue_pixels=int(tissue.size),
            ))

    if not rows:
        print(f"[phase {phase_id}] no condition results yet")
        return
    csv_path = os.path.join(phase_dir, "phase_results.csv")
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[phase {phase_id}] wrote {csv_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Stage to figureS3{a_parameter,b_optimizer}_sensitivity/
# ---------------------------------------------------------------------------

def _stage_to_figure_paper_S12() -> None:
    """Mirror per-phase phase_results.csv into figureS3{a_parameter,b_optimizer}_sensitivity/parameter_ablation/."""
    for phase_id, phase in PHASES.items():
        src = os.path.join(OUT_BASE,
                           f"phase_{phase_id:02d}_{phase['name']}",
                           "phase_results.csv")
        if not os.path.isfile(src):
            continue
        if phase_id in S1_PHASE_IDS:
            sub = "figureS3a_parameter_sensitivity"
        elif phase_id in S2_PHASE_IDS:
            sub = "figureS3b_optimizer_sensitivity"
        else:
            continue  # unmapped phase — skip staging
        dst = os.path.join(PHANTOM_ROOT, "figures", "data", sub,
                            "parameter_ablation",
                            f"phase_{phase_id:02d}_{phase['name']}",
                            "phase_results.csv")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  [stage → {sub}] phase_{phase_id:02d}_{phase['name']}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CF_REFERENCE = {1.0: 4.523, 1.5: 5.914, 2.0: 5.878, 3.0: 7.742, 5.0: 9.346}
_PHANTOM_ORDER = [(1.0, 1), (1.0, 2), (1.5, 1), (1.5, 2),
                   (2.0, 1), (2.0, 2), (3.0, 1), (3.0, 2),
                   (5.0, 1), (5.0, 2)]


def _format_cond_tag(v) -> str:
    """Match the cond_<tag> naming convention used under outputs/parameter_ablation/."""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:g}"
    return str(v)


def _list_phantom_dirs(cond_dir: str) -> list:
    return [d for d in os.listdir(cond_dir)
            if d.startswith("IL_") and "pct_rep" in d
            and os.path.isdir(os.path.join(cond_dir, d))]


def _parse_label(label: str) -> tuple:
    """Parse 'IL_5.0pct_rep1_bscan_256' or 'IL_5.0pct_rep1' into (5.0, 1)."""
    parts = label.split("_")
    conc = float(parts[1].replace("pct", ""))
    rep = int(parts[2].replace("rep", ""))
    return conc, rep


def _phantom_index(conc: float, rep: int) -> int:
    return _PHANTOM_ORDER.index((conc, rep))


def _err_pct(conc: float, mu_mean: float) -> float:
    cf = CF_REFERENCE.get(conc, float("nan"))
    if not (math.isfinite(cf) and cf > 0):
        return float("nan")
    return 100.0 * abs(mu_mean - cf) / cf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_BASE, exist_ok=True)
    summarize_only = int(os.environ.get("SUMMARIZE", "0")) == 1
    stage_only     = int(os.environ.get("STAGE_ONLY", "0")) == 1
    phase_env      = os.environ.get("PHASE", "0")
    phase = int(phase_env) if phase_env else 0

    if stage_only:
        _stage_to_figure_paper_S12()
        return

    phases_to_run = [phase] if phase in PHASES else list(PHASES.keys())
    for p in phases_to_run:
        if summarize_only:
            summarize_phase(p)
        else:
            run_phase(p)

    _stage_to_figure_paper_S12()
    print(f"\nDone. Outputs under: {OUT_BASE}")


if __name__ == "__main__":
    main()
