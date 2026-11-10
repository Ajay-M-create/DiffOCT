#!/usr/bin/env python3
"""
CPU & GPU Benchmarking via Modal
=================================
Runs the DiffOCT SS (Beer-Lambert) optimization routine on one phantom B-scan
across different Modal GPU types and CPU configurations.  Measures per-iteration
timing (avg + std), total optimization wall-clock time (excluding panel I/O),
and peak GPU memory.  Generates a 2x2 final panel PNG per config.

The B-scan data is loaded locally from the phantom TIFF stack, serialized, and
sent to each Modal container.  Source code (src/) and config YAML files are
shipped via image layers.

Usage:
    modal run experiments/hardware_benchmark.py

Environment variables:
    NUM_ITERATIONS  -- iterations per benchmark run (default: 15000)
    PHANTOM_INDEX   -- which despeckled phantom to use, 0-9 (default: 0)
    BSCAN_INDEX     -- B-scan frame index within the TIFF (default: 256)

Outputs:
    outputs/15_benchmarking/benchmark_results.csv
    outputs/15_benchmarking/benchmark_summary.txt
    outputs/15_benchmarking/{config_name}_final_panel.png   (one per config)
"""

import os
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Modal app & container image
# ---------------------------------------------------------------------------
app = modal.App("edelmanlab-diffoct-benchmark")

bench_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install("omegaconf==2.3.0", "numpy==1.26.4", "matplotlib", "Pillow", "pyyaml")
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/root/project/src")
    .add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/root/project/configs")
)

# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------
BENCHMARK_CONFIGS = [
    # ---- GPU type comparison (2 CPUs each) ----
    {"name": "T4_2cpu",         "gpu": "T4",         "cpu": 2},
    {"name": "L4_2cpu",         "gpu": "L4",         "cpu": 2},
    {"name": "A10_2cpu",        "gpu": "A10",        "cpu": 2},
    {"name": "L40S_2cpu",       "gpu": "L40S",       "cpu": 2},
    {"name": "A100_40GB_2cpu",  "gpu": "A100-40GB",  "cpu": 2},
    {"name": "A100_80GB_2cpu",  "gpu": "A100-80GB",  "cpu": 2},
    {"name": "H100_2cpu",       "gpu": "H100",       "cpu": 2},
    {"name": "H200_2cpu",       "gpu": "H200",       "cpu": 2},
    # ---- CPU-only (varied core counts) ----
    {"name": "CPU_1core",       "gpu": None,          "cpu": 1},
    {"name": "CPU_2core",       "gpu": None,          "cpu": 2},
    {"name": "CPU_4core",       "gpu": None,          "cpu": 4},
    {"name": "CPU_8core",       "gpu": None,          "cpu": 8},
    {"name": "CPU_16core",      "gpu": None,          "cpu": 16},
    # ---- CPU variation with a fixed GPU (A100-40GB) ----
    {"name": "A100_1cpu",       "gpu": "A100-40GB",   "cpu": 1},
    {"name": "A100_4cpu",       "gpu": "A100-40GB",   "cpu": 4},
    {"name": "A100_8cpu",       "gpu": "A100-40GB",   "cpu": 8},
]


# ---------------------------------------------------------------------------
# Core benchmark implementation (executed inside Modal container)
# ---------------------------------------------------------------------------
def _run_benchmark(
    bscan_bytes: bytes,
    num_iters: int,
    config_yaml: str,
    opt_config_yaml: str,
    sc_config_yaml: str,
) -> dict:
    """Run one full optimization and return timing metrics + final panel PNG."""
    import sys, os, time, tempfile, io       # noqa: E401

    sys.path.insert(0, "/root/project")

    import numpy as np                        # noqa: E402
    import torch                              # noqa: E402
    from torch import nn                      # noqa: E402
    from omegaconf import OmegaConf           # noqa: E402
    import matplotlib                         # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt           # noqa: E402

    from src.forward import create_forward_model                      # noqa: E402
    from src.inverse import (optimize, build_noise_floor_mask,        # noqa: E402
                             subtract_noise_floor_linear)

    # ── device info ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        gpu_name = "CPU"
        gpu_mem_gb = 0.0
    cpu_count = os.cpu_count()

    # ── load configs ──────────────────────────────────────────────
    config = OmegaConf.create(config_yaml)
    opt_config = OmegaConf.create(opt_config_yaml)
    sc_config = OmegaConf.create(sc_config_yaml)

    config.forward_model_system.forward_model_type = "ss"
    opt_config.optimization.num_iterations = num_iters
    opt_config.optimization.panel_save_interval = num_iters * 10  # effectively disable

    # ── prepare target B-scan ─────────────────────────────────────
    D = int(config.forward_model_system.depth_samples)
    A = int(config.forward_model_system.a_lines)
    bscan_np = np.frombuffer(bscan_bytes, dtype=np.float32).reshape(D, A).copy()
    I_target_dB = torch.tensor(bscan_np, dtype=torch.float32, device=device)

    noise_floor_dB = float(config.forward_model_system.noise_floor_dB)
    I_target_dB = subtract_noise_floor_linear(I_target_dB, noise_floor_dB)
    noise_floor_mask = build_noise_floor_mask(I_target_dB, noise_floor_dB, device)

    # ── forward model ─────────────────────────────────────────────
    forward_model = create_forward_model(config, device=str(device))
    forward_model.to(device)

    # ── init parameters (deterministic seed) ──────────────────────
    torch.manual_seed(42)
    mu_min = float(sc_config.scatterers.random_values_min)
    mu_max = float(sc_config.scatterers.random_values_max)
    mu_data = torch.empty(D, A, dtype=torch.float32, device=device).uniform_(mu_min, mu_max)
    alpha_data = torch.ones(D, A, dtype=torch.float32, device=device)

    mu = nn.Parameter(mu_data.clone())
    alpha = nn.Parameter(alpha_data.clone())

    prune_threshold = float(config.patient.prune_threshold_m_inv)

    # ── run optimization ──────────────────────────────────────────
    out_dir = tempfile.mkdtemp()
    t_wall_start = time.perf_counter()

    mu_best, alpha_best, metrics, _, _, mu_init, _ = optimize(
        forward_model, mu, alpha, I_target_dB, opt_config,
        prune_threshold, device, out_dir,
        noise_floor_mask=noise_floor_mask,
        recompute_gain_mse_optimal_each_iter=True,
    )

    t_wall_total = time.perf_counter() - t_wall_start

    # ── extract timing & metrics ──────────────────────────────────
    iter_times = np.array(metrics["time"])
    mse_vals = np.array(metrics["mse"])
    mu_means = np.array(metrics["mu_mean_mm"])

    warmup = min(10, len(iter_times) // 2)
    steady = iter_times[warmup:]

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # ── generate synthetic OCT with final mu ──────────────────────
    with torch.no_grad():
        I_pred_dB = forward_model(mu_best, alpha_best, noise_mask=noise_floor_mask)

    mu_init_np = mu_init.cpu().numpy()
    mu_best_np = mu_best.detach().cpu().numpy()
    I_pred_np = I_pred_dB.cpu().numpy()
    I_target_np = I_target_dB.cpu().numpy()

    # ── 2x2 final panel ──────────────────────────────────────────
    mu_vmax = float(opt_config.optimization.mu_max_mm_inv)
    tissue = noise_floor_mask.cpu().numpy() > 0.5
    tgt_tissue = I_target_np[tissue]
    dB_vmin = float(tgt_tissue.min()) if tissue.any() else float(I_target_np.min())
    dB_vmax = float(tgt_tissue.max()) if tissue.any() else float(I_target_np.max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im0 = axes[0, 0].imshow(mu_init_np, aspect="auto", cmap="plasma", vmin=0, vmax=mu_vmax)
    axes[0, 0].set_title("Initial mu [mm-1]")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(mu_best_np, aspect="auto", cmap="plasma", vmin=0, vmax=mu_vmax)
    axes[0, 1].set_title("Final mu [mm-1]")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(I_pred_np, aspect="auto", cmap="gray", vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 0].set_title("Synthetic OCT (forward model)")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(I_target_np, aspect="auto", cmap="gray", vmin=dB_vmin, vmax=dB_vmax)
    axes[1, 1].set_title("Target OCT")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    best_mse = float(np.min(mse_vals))
    fig.suptitle(
        f"{gpu_name}  |  {cpu_count} CPUs  |  {num_iters} iters  |  "
        f"MSE={best_mse:.4f}  |  mu_mean={mu_means[-1]:.4f} mm-1",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    panel_png = buf.getvalue()

    # ── return results ────────────────────────────────────────────
    return {
        "gpu_name":             gpu_name,
        "gpu_mem_gb":           round(gpu_mem_gb, 1),
        "cpu_count":            cpu_count,
        "device":               str(device),
        "num_iterations":       num_iters,
        "wall_time_s":          round(t_wall_total, 2),
        "opt_time_s":           round(float(np.sum(iter_times)), 2),
        "avg_iter_time_ms":     round(float(np.mean(steady)) * 1000, 3),
        "std_iter_time_ms":     round(float(np.std(steady)) * 1000, 3),
        "median_iter_time_ms":  round(float(np.median(steady)) * 1000, 3),
        "p5_iter_time_ms":      round(float(np.percentile(steady, 5)) * 1000, 3),
        "p95_iter_time_ms":     round(float(np.percentile(steady, 95)) * 1000, 3),
        "best_mse":             round(best_mse, 6),
        "final_mu_mean_mm":     round(float(mu_means[-1]), 4),
        "peak_memory_mb":       round(peak_mem_mb, 1),
        "panel_png":            panel_png,
    }


# ---------------------------------------------------------------------------
# Register one Modal function per benchmark config
# ---------------------------------------------------------------------------
def _make_bench_fn(cfg_name):
    """Factory: returns a uniquely-named function that calls _run_benchmark."""
    def _fn(
        bscan_bytes: bytes,
        num_iters: int,
        config_yaml: str,
        opt_config_yaml: str,
        sc_config_yaml: str,
    ) -> dict:
        return _run_benchmark(
            bscan_bytes, num_iters, config_yaml, opt_config_yaml, sc_config_yaml,
        )
    _fn.__name__ = f"bench_{cfg_name}"
    _fn.__qualname__ = f"bench_{cfg_name}"
    return _fn


_benchmarks = {}

for _cfg in BENCHMARK_CONFIGS:
    _fn = _make_bench_fn(_cfg["name"])
    _kw = {
        "image":   bench_image,
        "cpu":     float(_cfg["cpu"]),
        "memory":  32768,
        "timeout": 3600,
    }
    if _cfg.get("gpu"):
        _kw["gpu"] = _cfg["gpu"]
    _decorated = app.function(**_kw)(_fn)
    _benchmarks[_cfg["name"]] = _decorated
    # Modal resolves functions via getattr(module, qualname) -- must be a module global
    globals()[f"bench_{_cfg['name']}"] = _decorated


# ---------------------------------------------------------------------------
# Local entrypoint -- loads data, dispatches, collects & saves results
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    import glob
    import re
    import csv
    import numpy as np
    from PIL import Image

    num_iters   = int(os.environ.get("NUM_ITERATIONS", 15000))
    phantom_idx = int(os.environ.get("PHANTOM_INDEX", 0))
    bscan_idx   = int(os.environ.get("BSCAN_INDEX", 256))

    print("=" * 60)
    print("DiffOCT Modal Benchmark")
    print("=" * 60)
    print(f"  Phantom index : {phantom_idx}")
    print(f"  B-scan index  : {bscan_idx}")
    print(f"  Iterations    : {num_iters}")
    print(f"  Configs       : {len(BENCHMARK_CONFIGS)}")

    # ── discover & load phantom B-scan ────────────────────────────
    input_dir = PROJECT_ROOT / "inputs" / "IntralipidPhantoms_20250206"
    tiff_pattern = str(input_dir / "*_DespeckledLogInt.tif")
    files = sorted(glob.glob(tiff_pattern))

    phantoms = []
    for f in files:
        match = re.search(r"AgarInIL_(\d+)\.\.(\d+)", os.path.basename(f))
        if match:
            conc = float(f"{match.group(1)}.{match.group(2)}")
            same = [p for p in phantoms if p["concentration"] == conc]
            phantoms.append({
                "concentration": conc,
                "path": f,
                "replicate": len(same) + 1,
            })
    phantoms.sort(key=lambda x: (x["concentration"], x["replicate"]))

    if phantom_idx >= len(phantoms):
        print(f"ERROR: PHANTOM_INDEX={phantom_idx} out of range (found {len(phantoms)} phantoms)")
        sys.exit(1)

    p = phantoms[phantom_idx]
    print(f"  Phantom       : {p['concentration']}% IL (replicate {p['replicate']})")
    print(f"  TIFF          : {os.path.basename(p['path'])}")

    img = Image.open(p["path"])
    img.seek(bscan_idx)
    raw = np.array(img, dtype=np.float32)
    bscan_dB = raw * (130.0 / 65535.0)
    bscan_bytes = bscan_dB.tobytes()
    print(f"  B-scan shape  : {bscan_dB.shape}  range=[{bscan_dB.min():.1f}, {bscan_dB.max():.1f}] dB")

    # ── read config YAML strings ──────────────────────────────────
    config_yaml     = (PROJECT_ROOT / "configs" / "config.yaml").read_text()
    opt_config_yaml = (PROJECT_ROOT / "configs" / "optimization_config_fast.yaml").read_text()
    sc_config_yaml  = (PROJECT_ROOT / "configs" / "scatterer_config.yaml").read_text()

    # ── output directory ──────────────────────────────────────────
    out_dir = PROJECT_ROOT / "outputs" / "15_benchmarking"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── dispatch all benchmarks in parallel via .spawn() ──────────
    print(f"\nDispatching {len(BENCHMARK_CONFIGS)} benchmarks to Modal...")
    handles = {}
    for cfg in BENCHMARK_CONFIGS:
        name = cfg["name"]
        gpu_label = cfg["gpu"] or "CPU-only"
        print(f"  -> {name:24s}  (gpu={gpu_label}, cpu={cfg['cpu']})")
        handles[name] = _benchmarks[name].spawn(
            bscan_bytes, num_iters, config_yaml, opt_config_yaml, sc_config_yaml,
        )

    # ── collect results ───────────────────────────────────────────
    print(f"\nCollecting results...")
    results = []
    for name, handle in handles.items():
        cfg = next(c for c in BENCHMARK_CONFIGS if c["name"] == name)
        print(f"  {name:24s} ... ", end="", flush=True)
        try:
            r = handle.get()

            # Save panel PNG locally
            panel_png = r.pop("panel_png", None)
            if panel_png:
                png_path = out_dir / f"{name}_final_panel.png"
                with open(png_path, "wb") as fout:
                    fout.write(panel_png)

            r["config_name"] = name
            r["modal_gpu_spec"] = cfg["gpu"] or "none"
            r["requested_cpu"] = cfg["cpu"]
            results.append(r)
            print(f"done  {r['wall_time_s']:>7.1f}s  "
                  f"{r['avg_iter_time_ms']:>8.2f} ms/iter  "
                  f"({r['gpu_name']})")
        except Exception as e:
            print(f"FAILED  ({e})")

    if not results:
        print("\nNo results collected!")
        sys.exit(1)

    # ── save CSV ──────────────────────────────────────────────────
    csv_path = out_dir / "benchmark_results.csv"
    fieldnames = [
        "config_name", "modal_gpu_spec", "gpu_name", "gpu_mem_gb",
        "requested_cpu", "cpu_count", "device",
        "num_iterations", "wall_time_s", "opt_time_s",
        "avg_iter_time_ms", "std_iter_time_ms", "median_iter_time_ms",
        "p5_iter_time_ms", "p95_iter_time_ms",
        "best_mse", "final_mu_mean_mm", "peak_memory_mb",
    ]
    sorted_results = sorted(results, key=lambda x: x["avg_iter_time_ms"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nCSV saved: {csv_path}")

    # ── formatted summary table ───────────────────────────────────
    hdr = (f"{'Config':<22s}  {'GPU':<26s}  {'CPU':>3s}  "
           f"{'Avg(ms)':>8s}  {'Std(ms)':>8s}  {'Med(ms)':>8s}  "
           f"{'Total(s)':>8s}  {'Mem(MB)':>8s}  {'BestMSE':>12s}")
    sep = "-" * len(hdr)

    lines = [
        f"DiffOCT Modal Benchmark Results",
        f"Phantom: {p['concentration']}% IL rep{p['replicate']}, "
        f"B-scan {bscan_idx}, {num_iters} iters",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timing: avg/std/median exclude first 10 warmup iterations",
        "",
        sep,
        hdr,
        sep,
    ]
    for r in sorted_results:
        lines.append(
            f"{r['config_name']:<22s}  {r['gpu_name']:<26s}  {r['requested_cpu']:>3d}  "
            f"{r['avg_iter_time_ms']:>8.3f}  {r['std_iter_time_ms']:>8.3f}  "
            f"{r['median_iter_time_ms']:>8.3f}  "
            f"{r['wall_time_s']:>8.1f}  {r['peak_memory_mb']:>8.1f}  "
            f"{r['best_mse']:>12.6f}"
        )
    lines.append(sep)

    summary_text = "\n".join(lines)
    print(f"\n{summary_text}")

    txt_path = out_dir / "benchmark_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary saved: {txt_path}")
    print(f"Panels saved: {out_dir}/*_final_panel.png ({len(results)} files)")
