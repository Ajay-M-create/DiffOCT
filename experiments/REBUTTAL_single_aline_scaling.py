#!/usr/bin/env python3
"""
rebuttal_single_aline_scaling.py — Figure 1 rerun on the axial axis

Reviewer question on Figure 1: the scaling axis there grows the number of
A-lines optimized jointly while each A-line keeps a fixed 40-pixel tissue
band, so the number of *coupled* parameters within an A-line never changes.
Would increasing the axial sampling not be more representative of increasing
dimensionality?

This script reruns Figure 1's experiment on that axis instead. A single
A-line (N_A = 1) is optimized throughout, and the problem size N is the
number of attenuation parameters along depth:

    N = 25, 50, 100, 200, 500, 1000, 2000, 5000

Every parameter in a given run is coupled to every deeper one through the
cumulative path integral, so N here is the coupled dimensionality rather than
a count of independent subproblems.

Axial refinement keeps the phantom physically identical. The underlying
continuous mu(z, x) field is the one used by phantom_0000; refining by
f = N / 40 divides Delta z by f and multiplies z_FP and z_R by f, so physical
tissue thickness (0.16 mm), optical thickness and focus geometry are all
unchanged and only the sampling density changes. Increasing N therefore adds
coupled parameters without altering the physical measurement.

Optimizer implementations, hyperparameters (LR, MU_MIN/MAX, LOSS_SCALE,
NOISE_FLOOR_DB, iteration budget) and the timing protocol are imported
unchanged from experiments/optimizer_scaling.py, and the same 8 seeds are
used, so numbers are directly comparable to Figure 1.

Usage:
    python experiments/REBUTTAL_single_aline_scaling.py              # all seeds
    SEED_INDEX=3 python experiments/REBUTTAL_single_aline_scaling.py # one shard
    MERGE=1 python experiments/REBUTTAL_single_aline_scaling.py      # plot

Output: outputs/rebuttal_single_aline_scaling/
  - results_pixels{N:05d}_seed{S}.json     per-seed shard
  - results_single_aline.json              merged across seeds
  - rebuttal_single_aline_scaling.pdf/.jpeg/.png/.svg
Staged for the paper under figures/data/figure1_single_aline_scaling/.
"""

import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import zoom
from scipy.interpolate import PchipInterpolator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedLocator, FixedFormatter, SymmetricalLogLocator,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.forward import create_forward_model
from src.inverse import build_noise_floor_mask

# Reuse Figure 1's optimizers, hyperparameters and timing protocol verbatim.
from experiments.optimizer_scaling import (  # noqa: E402
    OPTIMIZERS, OPTIMIZER_ORDER,
    MU_MIN, MU_AIR, NOISE_FLOOR_DB, NUM_ITERS, NUM_SEEDS,
    compute_mape, collect_provenance,
)

# ── the sweep: attenuation parameters along a single A-line ───────────────
PIXEL_COUNTS = [25, 50, 100, 200, 500, 1000, 2000, 5000]
N_ALINES = 1

# ── phantom geometry (matches inverse_model_verification.py phantom_0000) ──
BASE_DEPTH = 500          # depth samples at f = 1
BASE_A_LINES = 500        # A-lines in the source field
BASE_TISSUE_PX = 40       # tissue band thickness at f = 1 (0.16 mm)
TISSUE_START = 230        # tissue band start pixel at f = 1
PHANTOM_SEED = 30000      # phantom_0000's seed family
INIT_SEED_BASE = 42       # matches optimizer_scaling.main()
INIT_MU_HIGH = 10.0

_OUT_SUFFIX = os.environ.get('OUT_SUFFIX', '')
OUT_DIR = os.path.join(PHANTOM_ROOT, "outputs",
                       f"rebuttal_single_aline_scaling{_OUT_SUFFIX}")
STAGE_DIR = os.path.join(PHANTOM_ROOT, "figures", "data",
                         f"figure1_single_aline_scaling{_OUT_SUFFIX}")

SEED_INDEX = os.environ.get('SEED_INDEX', '')
MERGE = os.environ.get('MERGE', '0') == '1'


# ═══════════════════════════════════════════════════════════════════════════
# Axially refined phantom
# ═══════════════════════════════════════════════════════════════════════════

def build_phantom(n_pixels, device):
    """Build the phantom_0000 field with `n_pixels` parameters along depth.

    f = n_pixels / BASE_TISSUE_PX. The continuous mu(z, x) is unchanged;
    Delta z is divided by f while z_FP and z_R (held in pixels by the forward
    model) are multiplied by f, so the physical phantom is identical at every
    N and only the axial sampling density changes.

    Returns (fwd, mu_gt, alpha_gt, target_dB, noise_mask, tissue_bool, meta).
    """
    f = n_pixels / BASE_TISSUE_PX
    depth = int(round(BASE_DEPTH * f))
    t0 = int(round(TISSUE_START * f))
    t1 = t0 + n_pixels
    assert t1 <= depth, f"tissue band {t0}:{t1} exceeds depth {depth}"

    rng = np.random.default_rng(seed=PHANTOM_SEED)
    low_res = rng.uniform(0.0, 10.0, size=(20, 50)).astype(np.float32)
    # Draw order preserved so alpha matches phantom_0000 exactly.
    alpha_val = float(rng.uniform(1e-10, 1e-2))

    # Same continuous field, resampled to n_pixels along depth.
    upsampled = zoom(low_res, (n_pixels / 20.0, BASE_A_LINES / 50.0), order=1)

    # Center A-line, same slice convention as Figure 1 at N = 1 A-line.
    a0 = BASE_A_LINES // 2
    a1 = a0 + N_ALINES

    mu_np = np.full((depth, N_ALINES), MU_AIR, dtype=np.float32)
    mu_np[t0:t1, :] = upsampled[:n_pixels, a0:a1]
    alpha_np = np.full((depth, N_ALINES), alpha_val, dtype=np.float32)

    mu_gt = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_gt = torch.tensor(alpha_np, dtype=torch.float32, device=device)

    config = OmegaConf.load(os.path.join(PHANTOM_ROOT, 'configs', 'config.yaml'))
    fms = config.forward_model_system
    fms.forward_model_type = 'ss'
    fms.depth_samples = depth
    fms.a_lines = N_ALINES
    fms.axial_pixel_size_mm = float(fms.axial_pixel_size_mm) / f
    fms.focal_plane_position_pixels = float(fms.focal_plane_position_pixels) * f
    fms.rayleigh_range_pixels = float(fms.rayleigh_range_pixels) * f

    fwd = create_forward_model(config, device=device)

    # Gain auto-init exactly as in inverse_model_verification.py: tissue
    # median sits 15 dB above the noise floor, then frozen.
    tissue_bool = mu_gt > MU_AIR * 10
    with torch.no_grad():
        fwd.gain_dB.fill_(0.0)
        I_raw = fwd(mu_gt, alpha_gt)
        raw_median = (I_raw[tissue_bool].median().item() if tissue_bool.any()
                      else I_raw.median().item())
        gain_dB_val = (NOISE_FLOOR_DB + 15.0) - raw_median
        fwd.gain_dB.fill_(gain_dB_val)
        target_dB = fwd(mu_gt, alpha_gt)
    fwd.eval()

    noise_mask = build_noise_floor_mask(target_dB, NOISE_FLOOR_DB, device)

    meta = {
        'n_pixels': n_pixels,
        'n_alines': N_ALINES,
        'f': f,
        'depth_samples': depth,
        'axial_pixel_size_mm': float(fms.axial_pixel_size_mm),
        'n_params': int(tissue_bool.sum().item()),
        'n_params_above_floor': int(
            (noise_mask * tissue_bool.float()).sum().item()),
        'n_array_pixels': int(mu_gt.numel()),
        'gain_dB': gain_dB_val,
        'alpha': alpha_val,
        'optical_thickness': float(
            mu_np[t0:t1, :].mean() * n_pixels * float(fms.axial_pixel_size_mm)),
    }
    return fwd, mu_gt, alpha_gt, target_dB, noise_mask, tissue_bool, meta


def make_init(depth, seed_idx, tissue_bool, device):
    """Scrambled initial mu, uniform in [MU_MIN, 10] on tissue pixels.

    Same distribution and seed family as optimizer_scaling.main(). The draw
    cannot be sliced from one shared array here because the depth grid itself
    changes with N, so each N draws from the same generator family.
    """
    rng = np.random.default_rng(seed=INIT_SEED_BASE + seed_idx)
    mu_scr = rng.uniform(MU_MIN, INIT_MU_HIGH,
                         size=(depth, N_ALINES)).astype(np.float32)
    tb = tissue_bool.float()
    return torch.tensor(mu_scr, dtype=torch.float32, device=device) * tb \
        + MU_AIR * (1.0 - tb)


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_size(n_pixels, seeds_to_run, device):
    """Run every optimizer at one problem size."""
    print(f"\n{'#' * 70}")
    print(f"  N = {n_pixels} pixels on a single A-line")
    print(f"{'#' * 70}")

    (fwd, mu_gt, alpha_gt, target_dB, noise_mask,
     tissue_bool, meta) = build_phantom(n_pixels, device)
    print(f"  depth={meta['depth_samples']:,}  "
          f"dz={meta['axial_pixel_size_mm']*1e3:.4f} um  "
          f"n={meta['n_params']:,} params "
          f"({meta['n_params_above_floor']:,} above floor)  "
          f"gain={meta['gain_dB']:.1f} dB  "
          f"OT={meta['optical_thickness']:.4f}")

    inits = {s: make_init(meta['depth_samples'], s, tissue_bool, device)
             for s in seeds_to_run}

    out = {}
    for opt_name in OPTIMIZER_ORDER:
        opt_fn = OPTIMIZERS[opt_name]
        rec = {k: [] for k in ('time_ms', 'mape', 'fwd_evals',
                               'wall_sec', 'ms_per_fwd', 'peak_mem_mb')}
        for seed_idx in seeds_to_run:
            (mu_final, avg_time_ms, _hist, fwd_evals,
             total_wall_sec, peak_mem_mb) = opt_fn(
                fwd, inits[seed_idx], alpha_gt, target_dB,
                noise_mask, device, NUM_ITERS, seed_idx=seed_idx)
            rec['time_ms'].append(avg_time_ms)
            rec['mape'].append(compute_mape(mu_final, mu_gt, tissue_bool))
            rec['fwd_evals'].append(fwd_evals)
            rec['wall_sec'].append(total_wall_sec)
            rec['ms_per_fwd'].append(
                (total_wall_sec / max(fwd_evals, 1)) * 1000.0)
            rec['peak_mem_mb'].append(peak_mem_mb)
            del mu_final

        s = {f'seed_{k}': v for k, v in rec.items()}
        s.update({f'avg_{k}': float(np.mean(v)) for k, v in rec.items()})
        s.update({f'std_{k}': float(np.std(v)) for k, v in rec.items()})
        s['seeds_present'] = list(seeds_to_run)
        out[opt_name] = s
        print(f"  {opt_name:15s}  ms/eval={s['avg_ms_per_fwd']:9.3f}"
              f"+-{s['std_ms_per_fwd']:.3f}  "
              f"wall={s['avg_wall_sec']:9.2f} s  "
              f"MAPE={s['avg_mape']:8.3f}+-{s['std_mape']:.3f}%  "
              f"peak={s['avg_peak_mem_mb']:.0f}MB")
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out['_meta'] = meta
    del fwd, mu_gt, alpha_gt, target_dB, noise_mask, tissue_bool, inits
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out


def main():
    if SEED_INDEX != '':
        try:
            seed = int(SEED_INDEX)
            assert 0 <= seed < NUM_SEEDS
        except (ValueError, AssertionError):
            raise SystemExit(f"Invalid SEED_INDEX={SEED_INDEX!r}; "
                             f"must be integer in [0, {NUM_SEEDS - 1}]")
        seeds_to_run = [seed]
    else:
        seeds_to_run = list(range(NUM_SEEDS))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Single-A-line axial scaling (Figure 1 reviewer response)")
    print(f"  N (pixels) : {PIXEL_COUNTS}")
    print(f"  seeds={seeds_to_run}  iters={NUM_ITERS:,}")
    print(f"  Device: {device}"
          + (f"  GPU: {torch.cuda.get_device_name(0)}"
             if device.type == 'cuda' else ""))
    print("=" * 70)

    t0 = time.perf_counter()
    suffix = f"_seed{seeds_to_run[0]}" if len(seeds_to_run) == 1 else ""
    for n_pixels in PIXEL_COUNTS:
        res = run_size(n_pixels, seeds_to_run, device)
        path = os.path.join(
            OUT_DIR, f'results_pixels{n_pixels:05d}{suffix}.json')
        with open(path, 'w') as fh:
            json.dump({str(n_pixels): res, '_provenance': collect_provenance()},
                      fh, indent=2, default=str)
        print(f"  -> {path}")

    print(f"\n  {len(PIXEL_COUNTS)} sizes in "
          f"{(time.perf_counter() - t0) / 60:.1f} min")
    print("  Plot with MERGE=1 python "
          "experiments/REBUTTAL_single_aline_scaling.py")


# ═══════════════════════════════════════════════════════════════════════════
# Merge + figure (Figure 1 style)
# ═══════════════════════════════════════════════════════════════════════════

def merge_shards():
    """Aggregate per-seed shards, write results_single_aline.json, plot, stage."""
    collected, metas, prov = {}, {}, None
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'results_pixels*.json'))):
        with open(path) as fh:
            d = json.load(fh)
        prov = prov or d.get('_provenance')
        for key, res in d.items():
            if key.startswith('_'):
                continue
            n = int(key)
            for opt_name, stats in res.items():
                if opt_name == '_meta':
                    metas[n] = stats
                    continue
                collected.setdefault(n, {}).setdefault(
                    opt_name, []).append(stats)
    if not collected:
        raise SystemExit(f"No shards found in {OUT_DIR}")

    merged = {}
    for n in sorted(collected):
        entry = {}
        for opt_name, entries in collected[n].items():
            out = {}
            for key in ('time_ms', 'mape', 'fwd_evals', 'wall_sec',
                        'ms_per_fwd', 'peak_mem_mb'):
                vals = [v for e in entries for v in e.get(f'seed_{key}', [])]
                if not vals:
                    continue
                out[f'avg_{key}'] = float(np.mean(vals))
                out[f'std_{key}'] = float(np.std(vals))
                out[f'seed_{key}'] = vals
            out['seeds_present'] = sorted(
                {s for e in entries for s in e.get('seeds_present', [])})
            entry[opt_name] = out
        entry['_meta'] = metas.get(n, {})
        merged[n] = entry

    payload = {str(n): v for n, v in merged.items()}
    payload['_provenance_merge'] = prov
    out_path = os.path.join(OUT_DIR, 'results_single_aline.json')
    with open(out_path, 'w') as fh:
        json.dump(payload, fh, indent=2, default=str)
    n_seeds = len(merged[next(iter(merged))][OPTIMIZER_ORDER[0]]
                  ['seeds_present'])
    print(f"  {len(merged)} sizes x {n_seeds} seeds -> {out_path}")

    _plot(merged)
    _stage()


# Figure 1's rc block, colors and markers, so the two panels sit side by side.
METHODS = [
    ('Random Search', 'tab:orange', '^'),
    ('CMA-ES',        'tab:red',    'D'),
    ('Adam',          'tab:blue',   'o'),
    ('L-BFGS',        'tab:green',  's'),
]
OPTIMIZER_FAMILIES = {
    'Random Search': 'GF', 'CMA-ES': 'GF', 'Adam': 'GB', 'L-BFGS': 'GB',
}
PAPER_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 120,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'lines.linewidth': 1.5,
}
TICK_FS = 13
LABEL_FS = 13


def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.tick_params(axis='both', labelsize=TICK_FS, width=0.8, length=3.5)


def _style_legend(legend, text_color='black'):
    frame = legend.get_frame()
    frame.set_boxstyle('round,pad=0.20')
    frame.set_facecolor('black')
    frame.set_edgecolor('none')
    frame.set_alpha(0.25)
    if legend.get_title().get_text():
        legend.get_title().set_color(text_color)
        legend.get_title().set_fontweight('bold')
    for txt in legend.get_texts():
        txt.set_color(text_color)
        txt.set_fontweight('bold')


def _pchip(xs, ys, log_xs_dense):
    """Monotone interpolation through the measured points, in log-log space.

    Figure 1 additionally fits parametric forms (logistic / Richards /
    offset power law) whose breakpoints were tuned to the A-line sweep and do
    not transfer to this axis, so every curve here is a PCHIP through the
    measured points. No extrapolation beyond the sampled range.
    """
    ys = np.maximum(np.asarray(ys, dtype=float), 1e-12)
    p = PchipInterpolator(np.log10(np.asarray(xs, dtype=float)), np.log10(ys))
    return 10 ** p(log_xs_dense)


def _plot(merged):
    plt.rcParams.update(PAPER_RC)
    sizes = sorted(merged)
    xs = np.array(sizes, dtype=float)
    log_xs_dense = np.linspace(np.log10(xs.min()), np.log10(xs.max()), 600)
    xs_dense = 10 ** log_xs_dense

    fig, ax_l = plt.subplots(figsize=(7.0, 4.6))
    _style_ax(ax_l)
    ax_r = ax_l.twinx()
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['right'].set_visible(True)
    ax_r.spines['right'].set_linewidth(0.9)
    ax_r.tick_params(axis='y', labelsize=TICK_FS, width=0.8, length=3.5)

    POINT_ALPHA = 0.5
    CURVE_ALPHA = 0.95
    for name, color, marker in METHODS:
        mape = np.array([merged[n][name]['avg_mape'] for n in sizes])
        msev = np.array([merged[n][name]['avg_ms_per_fwd'] for n in sizes])
        mape_std = np.array([merged[n][name]['std_mape'] for n in sizes])
        msev_std = np.array([merged[n][name]['std_ms_per_fwd'] for n in sizes])

        ax_l.plot(xs_dense, _pchip(xs, mape, log_xs_dense), color=color,
                  linestyle='-', linewidth=2.2, alpha=CURVE_ALPHA, zorder=3)
        ax_r.plot(xs_dense, _pchip(xs, msev, log_xs_dense), color=color,
                  linestyle='--', linewidth=1.5, alpha=CURVE_ALPHA * 0.95,
                  zorder=2)
        ax_l.errorbar(xs, mape, yerr=mape_std,
                      color=color, marker=marker, markersize=6,
                      linestyle='', capsize=3, elinewidth=1.0,
                      alpha=POINT_ALPHA, markerfacecolor=color,
                      markeredgecolor='white', markeredgewidth=0.6, zorder=4)
        ax_r.errorbar(xs, msev, yerr=msev_std,
                      color=color, marker=marker, markersize=5,
                      linestyle='', capsize=3, elinewidth=1.0,
                      alpha=POINT_ALPHA, markerfacecolor='none',
                      markeredgecolor=color, markeredgewidth=1.2, zorder=3)

    ax_l.set_xscale('log')
    ax_l.set_yscale('symlog', linthresh=0.1, linscale=0.15)
    ax_l.set_xlabel('Problem size $K$ (pixels in a single A-line)',
                    fontsize=LABEL_FS)
    ax_l.set_ylabel('MAPE (%)', fontsize=LABEL_FS)
    ax_r.set_yscale('symlog', linthresh=0.1, linscale=0.15)
    ax_r.set_ylabel('Time per evaluation (ms)', fontsize=LABEL_FS)

    ax_l.set_xticks(sizes)
    ax_l.set_xticklabels([str(n) for n in sizes])
    ax_l.tick_params(axis='x', which='minor', bottom=False, top=False)
    ax_l.set_xlim(xs.min() * 0.8, xs.max() * 1.25)

    yticks = [0, 1e-1, 1e0, 1e1, 1e2, 2e2]
    ylabels = ['0', '1e-1', '1e0', '1e1', '1e2', '2e2']
    for ax in (ax_l, ax_r):
        ax.set_ylim(-0.005, 2e2)
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_major_formatter(FixedFormatter(ylabels))
        ax.yaxis.set_minor_locator(SymmetricalLogLocator(
            base=10.0, linthresh=0.1, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
        ax.tick_params(axis='y', which='minor', length=2, labelsize=0)

    ax_l.grid(True, which='major', axis='both', alpha=0.40, linestyle=':',
              linewidth=0.7, zorder=0)
    ax_l.grid(True, which='minor', axis='y', alpha=0.18, linestyle=':',
              linewidth=0.4, zorder=0)
    ax_r.grid(False)

    LEGEND_X, LEGEND_W = 0.020, 0.355
    H_TOP, H_BOT, GAP = 0.245, 0.125, 0.045
    TOP_TOP = 0.965
    TOP_Y0 = TOP_TOP - H_TOP
    BOT_Y0 = TOP_TOP - H_TOP - GAP - H_BOT

    opt_handles = [
        Line2D([0], [0], color=c, marker=m, linestyle='-',
               lw=2.2, markersize=6, markerfacecolor=c,
               markeredgecolor='white', markeredgewidth=0.7,
               label=f'{OPTIMIZER_FAMILIES[name]}: {name}')
        for name, c, m in METHODS
    ]
    opt_legend = ax_l.legend(
        handles=opt_handles, loc='upper left',
        bbox_to_anchor=(LEGEND_X, TOP_Y0, LEGEND_W, H_TOP),
        bbox_transform=ax_l.transAxes,
        mode='expand', borderaxespad=0,
        fontsize=9, frameon=True, fancybox=True,
        title='Optimizer', title_fontsize=9, handlelength=1.8)
    ax_l.add_artist(opt_legend)
    _style_legend(opt_legend)

    style_handles = [
        Line2D([0], [0], color='#333333', lw=2.2, linestyle='-', label='MAPE'),
        Line2D([0], [0], color='#333333', lw=1.5, linestyle='--',
               label='ms/eval'),
        Line2D([], [], ls='',
               label=f'N = {len(merged[sizes[0]][METHODS[0][0]]["seed_mape"])} initializations'),
    ]
    metric_legend = ax_l.legend(
        handles=style_handles, loc='upper left',
        bbox_to_anchor=(LEGEND_X, BOT_Y0, LEGEND_W, H_BOT),
        bbox_transform=ax_l.transAxes,
        mode='expand', borderaxespad=0,
        fontsize=9, frameon=True, fancybox=True, handlelength=2.2)
    _style_legend(metric_legend)
    # The N entry has no line: drop its handle space so the text starts at the left edge.
    n_row = metric_legend._legend_handle_box.get_children()[0].get_children()[-1]
    n_row.get_children()[0].width = 0
    n_row.sep = 0

    plt.tight_layout()
    base = os.path.join(OUT_DIR, 'rebuttal_single_aline_scaling')
    fig.savefig(base + '.pdf', bbox_inches='tight', pad_inches=0.1)
    fig.savefig(base + '.jpeg', dpi=600, bbox_inches='tight', pad_inches=0.1,
                format='jpeg')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(base + '.svg', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  Figure saved to {base}.{{pdf,jpeg,png,svg}}")


def _stage():
    os.makedirs(STAGE_DIR, exist_ok=True)
    n = 0
    for src in glob.glob(os.path.join(OUT_DIR, 'results_pixels[0-9]*.json')):
        shutil.copy2(src, os.path.join(STAGE_DIR, os.path.basename(src)))
        n += 1
    for name in ('results_single_aline.json',
                 'rebuttal_single_aline_scaling.pdf',
                 'rebuttal_single_aline_scaling.jpeg',
                 'rebuttal_single_aline_scaling.png',
                 'rebuttal_single_aline_scaling.svg'):
        src = os.path.join(OUT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(STAGE_DIR, name))
            n += 1
    print(f"  [stage] copied {n} files -> {STAGE_DIR}")


if __name__ == '__main__':
    if MERGE:
        merge_shards()
    else:
        main()
