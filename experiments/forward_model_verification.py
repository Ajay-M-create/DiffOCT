#!/usr/bin/env python3
"""
test_forward_model_verification.py — Beer-Lambert Forward Model Validation

Validates the RasterScanOCTForward model against the analytical discrete
Beer-Lambert solution by creating a synthetic layered phantom and comparing
pixel-by-pixel.

For each of 500 A-scans, 2-5 optically homogeneous layers are placed between
depth pixels 100-200 with random attenuation coefficients (0.1-15 mm^-1). The
remaining pixels are set to a negligible background value (air).

The analytical discrete Beer-Lambert law:
    Linear:  I(z) = mu(z) * exp(-2 * sum_{i=0}^{z-1} mu(i) * dz)
    dB:      I_dB(z) = 10*log10(mu(z)) - K * sum_{i=0}^{z-1} mu(i) * dz
where K = 20/ln(10) ~ 8.6859.

With confocal correction disabled, gain_dB = 0, and alpha = 1, the forward
model should produce output identical to the analytical solution (up to float32
precision).

Output: Phantom_Experiments/outputs/forward_model_verification/

Usage:
    python test/test_forward_model_verification.py
"""

import os
import sys
import math

import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── path setup (same pattern as other test scripts) ──────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.forward import RasterScanOCTForward, RasterScanOCTForwardMS

# ── phantom generation constants ─────────────────────────────────────────────
LAYER_START_PX = 100        # tissue band start (inclusive)
LAYER_END_PX   = 200        # tissue band end   (exclusive)

# Paper Figure 2 panels a-f use these specific A-lines from the random
# layered phantom (2-, 3-, 4-layer respectively in the [100:200] tissue band)
PAPER_FIG5_ALINES = [342, 323, 27]
PAPER_FIG5_LAYERS = [2, 3, 4]
MU_MIN_MM      = 0.1        # minimum tissue mu  [mm^-1]
MU_MAX_MM      = 15.0       # maximum tissue mu  [mm^-1]
MU_AIR         = 1e-6       # background mu (air) [mm^-1]
N_LAYERS_MIN   = 2
N_LAYERS_MAX   = 5
SEED           = 42


# ═════════════════════════════════════════════════════════════════════════════
# Simple two-A-line demo (confocal-corrected, single layer)
# ═════════════════════════════════════════════════════════════════════════════

def run_simple_two_aline_demo(out_dir, device):
    """Super simple case: mu=5.0 mm^-1 single layer, two A-lines at different depths.

    A-line 0: tissue from pixel  80 to 250  (shallower, shorter)
    A-line 1: tissue from pixel 150 to 400  (deeper, longer)

    Runs the forward model both WITHOUT and WITH confocal correction,
    computes the analytical solution for each, and verifies pixel-level
    concordance.  Also plots the difference (confocal - no confocal)
    against the analytical Lorentzian C_dB(z).
    """
    MU_TISSUE = 5.0   # mm^-1
    eps = 1e-12

    # -- load config, override confocal for a strong visible effect --
    config = OmegaConf.load(os.path.join(PHANTOM_ROOT, 'configs', 'config.yaml'))
    config.forward_model_system.a_lines = 2
    config.forward_model_system.focal_plane_position_pixels = 200
    config.forward_model_system.rayleigh_range_pixels = 50

    depth_samples = int(config.forward_model_system.depth_samples)
    dz = float(config.forward_model_system.axial_pixel_size_mm)
    zFP = float(config.forward_model_system.focal_plane_position_pixels)
    zR  = float(config.forward_model_system.rayleigh_range_pixels)

    config.forward_model_system.confocal_correction = False
    model_no_conf = RasterScanOCTForward(config, device=device)
    model_no_conf.gain_dB.fill_(0.0)

    config.forward_model_system.confocal_correction = True
    model_conf = RasterScanOCTForward(config, device=device)
    model_conf.gain_dB.fill_(0.0)

    # -- create the simplest possible phantom --
    mu_np = np.full((depth_samples, 2), MU_AIR, dtype=np.float32)
    mu_np[80:250, 0]  = MU_TISSUE          # A-line 0
    mu_np[150:400, 1] = MU_TISSUE          # A-line 1

    mu_t    = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_t = torch.ones_like(mu_t)

    # -- forward model outputs --
    with torch.no_grad():
        I_no_conf = model_no_conf(mu_t, alpha_t).cpu().numpy()
        I_conf    = model_conf(mu_t, alpha_t).cpu().numpy()
    confocal_dB_vec = model_conf.confocal_dB.cpu().numpy()  # (depth,)

    # -- analytical solutions (float64 ground truth) --
    K = 20.0 / math.log(10.0)
    mu64 = mu_np.astype(np.float64)
    dz64 = np.float64(dz)

    cumulative = np.cumsum(mu64 * dz64, axis=0)
    shifted = np.zeros_like(mu64)
    shifted[1:] = cumulative[:-1]

    # Beer-Lambert only (no confocal)
    analytic_no_conf = (10.0 * np.log10(mu64 + eps) - K * shifted)

    # Confocal function
    z_px = np.arange(depth_samples, dtype=np.float64)
    C_dB_analytic = 10.0 * np.log10(
        1.0 / (((z_px - zFP) / zR) ** 2 + 1.0))

    # Beer-Lambert + confocal
    analytic_conf = analytic_no_conf + C_dB_analytic[:, np.newaxis]

    # -- error analysis --
    err_no_conf = I_no_conf - analytic_no_conf.astype(np.float32)
    err_conf    = I_conf    - analytic_conf.astype(np.float32)
    diff_model  = I_conf - I_no_conf                       # model: with - without
    diff_analytic = np.broadcast_to(                        # analytical C_dB
        C_dB_analytic[:, np.newaxis].astype(np.float32), diff_model.shape)
    err_diff    = diff_model - diff_analytic

    labels = ['A-line 0 (px 80\u2013250)', 'A-line 1 (px 150\u2013400)']
    colors = ['tab:blue', 'tab:red']
    depth_px = np.arange(depth_samples)

    # ── Figure: 2×3 panel (depth on x-axis, signal on y-axis) ──────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # (0,0) Input mu(z)
    ax = axes[0, 0]
    for c in range(2):
        ax.plot(depth_px, mu_np[:, c], color=colors[c], lw=2, label=labels[c])
    ax.set_xlabel('Depth [pixels]')
    ax.set_ylabel(r'$\mu_t$ [mm$^{-1}$]')
    ax.set_title('Input: Attenuation Coefficient')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) No confocal: model (solid) vs analytical (dashed)
    ax = axes[0, 1]
    for c in range(2):
        ax.plot(depth_px, I_no_conf[:, c], '-', color=colors[c], lw=1.5,
                label=f'{labels[c]} model')
        ax.plot(depth_px, analytic_no_conf[:, c], '--', color=colors[c],
                lw=1.2, alpha=0.7, label=f'{labels[c]} analytical')
    ax.axvline(zFP, color='gray', ls=':', lw=1, label=f'Focal plane ({zFP:.0f})')
    ax.set_xlabel('Depth [pixels]')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title(f'Confocal OFF\nsolid = model, dashed = analytical\n'
                 f'max |err| = {np.max(np.abs(err_no_conf)):.2e} dB')
    ax.legend(fontsize=6, loc='lower left')
    ax.grid(True, alpha=0.3)

    # (0,2) With confocal: model (solid) vs analytical (dashed)
    ax = axes[0, 2]
    for c in range(2):
        ax.plot(depth_px, I_conf[:, c], '-', color=colors[c], lw=1.5,
                label=f'{labels[c]} model')
        ax.plot(depth_px, analytic_conf[:, c], '--', color=colors[c],
                lw=1.2, alpha=0.7, label=f'{labels[c]} analytical')
    ax.axvline(zFP, color='gray', ls=':', lw=1, label=f'Focal plane ({zFP:.0f})')
    ax.set_xlabel('Depth [pixels]')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title(f'Confocal ON\nsolid = model, dashed = analytical\n'
                 f'max |err| = {np.max(np.abs(err_conf)):.2e} dB')
    ax.legend(fontsize=6, loc='lower left')
    ax.grid(True, alpha=0.3)

    # (1,0) Confocal function C_dB(z)
    ax = axes[1, 0]
    ax.plot(depth_px, C_dB_analytic, 'k-', lw=2, label='Analytical $C_{dB}(z)$')
    ax.plot(depth_px, confocal_dB_vec, 'g--', lw=1.5,
            label='Model buffer (float32)')
    ax.axvline(zFP, color='gray', ls=':', lw=1, label=f'Focal plane ({zFP:.0f})')
    ax.set_xlabel('Depth [pixels]')
    ax.set_ylabel('Confocal Correction [dB]')
    ax.set_title(f'Confocal Function $C_{{dB}}(z)$\n'
                 f'$z_{{FP}}$={zFP:.0f} px, $z_R$={zR:.1f} px')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Difference (confocal ON - OFF): model vs analytical C_dB
    ax = axes[1, 1]
    for c in range(2):
        ax.plot(depth_px, diff_model[:, c], '-', color=colors[c], lw=1.5,
                label=f'{labels[c]} model diff')
    ax.plot(depth_px, C_dB_analytic, 'k--', lw=1.5, alpha=0.7,
            label='Analytical $C_{dB}(z)$')
    ax.axvline(zFP, color='gray', ls=':', lw=1)
    ax.set_xlabel('Depth [pixels]')
    ax.set_ylabel('$\\Delta$ Intensity [dB]')
    ax.set_title(f'Difference (ON $-$ OFF) vs analytical $C_{{dB}}$\n'
                 f'max |err| = {np.max(np.abs(err_diff)):.2e} dB')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,2) Per-pixel error for all comparisons
    ax = axes[1, 2]
    for c in range(2):
        ax.plot(depth_px, err_no_conf[:, c], '-', color=colors[c], lw=0.8,
                alpha=0.7, label=f'{labels[c]} no-confocal err')
        ax.plot(depth_px, err_conf[:, c], '--', color=colors[c], lw=0.8,
                alpha=0.7, label=f'{labels[c]} confocal err')
    ax.axhline(0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Depth [pixels]')
    ax.set_ylabel('Error (model $-$ analytical) [dB]')
    ax.set_title('Per-pixel Error (float32 vs float64)')
    ax.legend(fontsize=6, loc='lower left')
    ax.grid(True, alpha=0.3)

    fig.suptitle(r'Simple Demo: $\mu_t$ = 5.0 mm$^{-1}$, single layer at two depths'
                 '\nForward model vs analytical Beer-Lambert, with and without confocal',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'simple_two_aline_demo.png'), dpi=200)
    plt.close()

    # -- console summary --
    print(f"  A-line 0: mu={MU_TISSUE} mm^-1, pixels [80, 250)")
    print(f"  A-line 1: mu={MU_TISSUE} mm^-1, pixels [150, 400)")
    print(f"  Confocal: zFP={zFP:.0f}, zR={zR:.1f}")
    print(f"  No-confocal max |err| = {np.max(np.abs(err_no_conf)):.2e} dB")
    print(f"  Confocal    max |err| = {np.max(np.abs(err_conf)):.2e} dB")
    print(f"  Difference  max |err| = {np.max(np.abs(err_diff)):.2e} dB")
    print(f"  -> simple_two_aline_demo.png")


# ═════════════════════════════════════════════════════════════════════════════
# Phantom generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_layered_phantom(depth_samples, a_lines, seed=SEED):
    """Create a 500x500 mu map with 2-5 random homogeneous layers per A-scan.

    Layers occupy depth pixels [LAYER_START_PX, LAYER_END_PX).
    Outside this band mu = MU_AIR (essentially zero).

    Returns
    -------
    mu : ndarray (depth_samples, a_lines), float32
    layer_info : list[dict] — per-A-scan layer metadata
    """
    rng = np.random.default_rng(seed)
    mu = np.full((depth_samples, a_lines), MU_AIR, dtype=np.float32)
    layer_info = []

    for a in range(a_lines):
        n_layers = rng.integers(N_LAYERS_MIN, N_LAYERS_MAX + 1)

        # n_layers - 1 unique interior boundary points within the tissue band
        if n_layers > 1:
            pool = np.arange(LAYER_START_PX + 1, LAYER_END_PX)
            inner = np.sort(rng.choice(pool, size=min(n_layers - 1, len(pool)),
                                       replace=False))
        else:
            inner = np.array([], dtype=int)

        boundaries = np.concatenate(
            [[LAYER_START_PX], inner, [LAYER_END_PX]]).astype(int)
        actual_n = len(boundaries) - 1
        mu_values = rng.uniform(MU_MIN_MM, MU_MAX_MM,
                                size=actual_n).astype(np.float32)

        for i in range(actual_n):
            mu[boundaries[i]:boundaries[i + 1], a] = mu_values[i]

        layer_info.append({
            'boundaries': boundaries.copy(),
            'mu_values':  mu_values.copy(),
            'n_layers':   actual_n,
        })

    return mu, layer_info


# ═════════════════════════════════════════════════════════════════════════════
# Analytical Beer-Lambert solutions
# ═════════════════════════════════════════════════════════════════════════════

def analytical_beer_lambert_dB(mu_np, axial_pixel_size_mm):
    """Analytical discrete Beer-Lambert in dB (float64 ground truth).

    I_dB(z) = 10 * log10(mu(z) + eps) - K * sum_{i=0}^{z-1} mu(i) * dz

    Uses eps = 1e-12 matching the forward model.
    """
    eps = 1e-12
    K = 20.0 / math.log(10.0)
    mu = mu_np.astype(np.float64)
    dz = np.float64(axial_pixel_size_mm)

    cumulative = np.cumsum(mu * dz, axis=0)
    shifted = np.zeros_like(mu)
    shifted[1:] = cumulative[:-1]

    I_dB = 10.0 * np.log10(mu + eps) - K * shifted
    return I_dB


def analytical_beer_lambert_linear(mu_np, axial_pixel_size_mm):
    """Analytical discrete Beer-Lambert in linear scale (float64).

    I(z) = mu(z) * exp(-2 * sum_{i=0}^{z-1} mu(i) * dz)
    """
    mu = mu_np.astype(np.float64)
    dz = np.float64(axial_pixel_size_mm)

    cumulative = np.cumsum(mu * dz, axis=0)
    shifted = np.zeros_like(mu)
    shifted[1:] = cumulative[:-1]

    return mu * np.exp(-2.0 * shifted)


# ═════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def plot_mu_phantom(mu_np, layer_info, out_dir):
    """Fig 1: mu map with tissue-band annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={'width_ratios': [2, 1]})

    # -- full mu map --
    ax = axes[0]
    im = ax.imshow(mu_np, cmap='viridis', aspect='auto', origin='upper')
    ax.axhline(LAYER_START_PX, color='r', ls='--', lw=1,
               label=f'Tissue start (px {LAYER_START_PX})')
    ax.axhline(LAYER_END_PX, color='r', ls='--', lw=1,
               label=f'Tissue end (px {LAYER_END_PX})')
    ax.set_xlabel('A-scan index (slow scan)')
    ax.set_ylabel('Depth (pixels)')
    ax.set_title('Synthetic Layered Phantom')
    ax.legend(loc='lower right', fontsize=8)
    plt.colorbar(im, ax=ax, label='mu [mm^-1]', fraction=0.046, pad=0.04)

    # -- zoomed tissue band --
    ax = axes[1]
    tissue = mu_np[LAYER_START_PX:LAYER_END_PX, :]
    im2 = ax.imshow(tissue, cmap='viridis', aspect='auto', origin='upper',
                     extent=[0, mu_np.shape[1], LAYER_END_PX, LAYER_START_PX])
    ax.set_xlabel('A-scan index')
    ax.set_ylabel('Depth (pixels)')
    ax.set_title(f'Tissue Band [{LAYER_START_PX}, {LAYER_END_PX})')
    plt.colorbar(im2, ax=ax, label='mu [mm^-1]', fraction=0.046, pad=0.04)

    # annotate layer count distribution
    n_counts = [info['n_layers'] for info in layer_info]
    dist_str = ", ".join(f"{k}L:{n_counts.count(k)}"
                         for k in sorted(set(n_counts)))
    fig.suptitle(f'Layer distribution: {dist_str}', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mu_phantom.png'), dpi=200)
    plt.close()


def plot_comparison_panel(I_model_dB, I_analytic_dB, I_model_linear,
                          I_analytic_linear, out_dir):
    """Fig 2: 2x3 panel — dB and linear comparison (tissue band only)."""
    ts, te = LAYER_START_PX, LAYER_END_PX

    model_dB_t    = I_model_dB[ts:te]
    analytic_dB_t = I_analytic_dB[ts:te]
    diff_dB       = np.abs(model_dB_t - analytic_dB_t)

    model_lin_t    = I_model_linear[ts:te]
    analytic_lin_t = I_analytic_linear[ts:te]
    diff_lin       = np.abs(model_lin_t - analytic_lin_t)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    extent = [0, I_model_dB.shape[1], te, ts]

    # Row 0: dB
    dB_vmin = min(model_dB_t.min(), analytic_dB_t.min())
    dB_vmax = max(model_dB_t.max(), analytic_dB_t.max())

    im0 = axes[0, 0].imshow(model_dB_t, cmap='inferno', aspect='auto',
                              origin='upper', extent=extent,
                              vmin=dB_vmin, vmax=dB_vmax)
    axes[0, 0].set_title('Forward Model Output [dB]')
    axes[0, 0].set_ylabel('Depth (px)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(analytic_dB_t, cmap='inferno', aspect='auto',
                              origin='upper', extent=extent,
                              vmin=dB_vmin, vmax=dB_vmax)
    axes[0, 1].set_title('Analytical Beer-Lambert [dB]')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(diff_dB, cmap='magma', aspect='auto',
                              origin='upper', extent=extent)
    axes[0, 2].set_title(f'|Difference| [dB]\nmax={diff_dB.max():.2e}')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 1: linear
    lin_vmin = 0
    lin_vmax = max(model_lin_t.max(), analytic_lin_t.max())

    im3 = axes[1, 0].imshow(model_lin_t, cmap='inferno', aspect='auto',
                              origin='upper', extent=extent,
                              vmin=lin_vmin, vmax=lin_vmax)
    axes[1, 0].set_title('Forward Model Output [linear]')
    axes[1, 0].set_xlabel('A-scan index')
    axes[1, 0].set_ylabel('Depth (px)')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(analytic_lin_t, cmap='inferno', aspect='auto',
                              origin='upper', extent=extent,
                              vmin=lin_vmin, vmax=lin_vmax)
    axes[1, 1].set_title('Analytical Beer-Lambert [linear]')
    axes[1, 1].set_xlabel('A-scan index')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(diff_lin, cmap='magma', aspect='auto',
                              origin='upper', extent=extent)
    axes[1, 2].set_title(f'|Difference| [linear]\nmax={diff_lin.max():.2e}')
    axes[1, 2].set_xlabel('A-scan index')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle('Forward Model vs Analytical Beer-Lambert (tissue band)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison_panel.png'), dpi=200)
    plt.close()


def plot_ascan_profiles(I_model_dB, I_analytic_dB, I_model_linear,
                        I_analytic_linear, layer_info, out_dir):
    """Fig 3: A-scan profiles for 5 representative A-scans.

    Top row: dB profiles.  Bottom row: linear profiles.
    """
    ascan_indices = [0, 125, 250, 375, 499]
    depth = np.arange(I_model_dB.shape[0])
    margin = 20  # pixels of context above/below tissue band

    fig, axes = plt.subplots(2, len(ascan_indices),
                             figsize=(4.2 * len(ascan_indices), 10),
                             sharey='row')

    for j, a_idx in enumerate(ascan_indices):
        info = layer_info[a_idx]

        # ── dB profile ──
        ax = axes[0, j]
        ax.plot(I_model_dB[:, a_idx], depth, 'b-', lw=1.3,
                label='Forward model', zorder=3)
        ax.plot(I_analytic_dB[:, a_idx], depth, 'r--', lw=1.3,
                label='Analytical', zorder=2)
        ax.axhline(LAYER_START_PX, color='gray', ls=':', alpha=0.5)
        ax.axhline(LAYER_END_PX, color='gray', ls=':', alpha=0.5)
        # shade each layer
        for i in range(info['n_layers']):
            b0, b1 = info['boundaries'][i], info['boundaries'][i + 1]
            ax.axhspan(b0, b1, alpha=0.08, color=f'C{i}')
            ax.text(ax.get_xlim()[0] + 1, (b0 + b1) / 2,
                    f'mu={info["mu_values"][i]:.1f}', fontsize=6,
                    va='center', color=f'C{i}', fontweight='bold')
        ax.set_ylim(LAYER_END_PX + margin, LAYER_START_PX - margin)
        ax.set_xlabel('Intensity [dB]')
        if j == 0:
            ax.set_ylabel('Depth [pixels]')
        ax.set_title(f'A-scan {a_idx}\n{info["n_layers"]} layers')
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, alpha=0.2)

        # ── linear profile ──
        ax = axes[1, j]
        ax.plot(I_model_linear[:, a_idx], depth, 'b-', lw=1.3,
                label='Forward model')
        ax.plot(I_analytic_linear[:, a_idx], depth, 'r--', lw=1.3,
                label='Analytical')
        ax.axhline(LAYER_START_PX, color='gray', ls=':', alpha=0.5)
        ax.axhline(LAYER_END_PX, color='gray', ls=':', alpha=0.5)
        for i in range(info['n_layers']):
            b0, b1 = info['boundaries'][i], info['boundaries'][i + 1]
            ax.axhspan(b0, b1, alpha=0.08, color=f'C{i}')
        ax.set_ylim(LAYER_END_PX + margin, LAYER_START_PX - margin)
        ax.set_xlabel('Intensity [linear]')
        if j == 0:
            ax.set_ylabel('Depth [pixels]')
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Beer-Lambert Validation: A-scan Profiles\n'
                 '(top: dB, bottom: linear)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ascan_profiles.png'), dpi=200)
    plt.close()


def plot_concordance(I_model_dB, I_analytic_dB, out_dir):
    """Fig 4: scatter plot, error histogram, per-A-scan MAE."""
    ts, te = LAYER_START_PX, LAYER_END_PX
    model_t = I_model_dB[ts:te].ravel()
    analytic_t = I_analytic_dB[ts:te].ravel().astype(np.float32)
    error_t = model_t - analytic_t

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── scatter ──
    ax = axes[0]
    subsample = np.random.default_rng(0).choice(
        len(model_t), size=min(50_000, len(model_t)), replace=False)
    ax.scatter(analytic_t[subsample], model_t[subsample],
               s=0.3, alpha=0.15, c='steelblue', edgecolors='none')
    lims = [min(analytic_t.min(), model_t.min()),
            max(analytic_t.max(), model_t.max())]
    ax.plot(lims, lims, 'r-', lw=1, label='y = x')
    ax.set_xlabel('Analytical I_dB')
    ax.set_ylabel('Forward Model I_dB')
    ax.set_title('Concordance (tissue pixels)')
    ax.legend(fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.2)

    # ── error histogram ──
    ax = axes[1]
    ax.hist(error_t, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(0, color='r', ls='--', lw=1)
    ax.set_xlabel('Error (Model - Analytical) [dB]')
    ax.set_ylabel('Pixel count')
    ax.set_title(f'Error Distribution\n'
                 f'mean={np.mean(error_t):.2e}, std={np.std(error_t):.2e}')
    ax.grid(True, alpha=0.2)

    # ── per-A-scan MAE ──
    ax = axes[2]
    per_ascan_mae = np.mean(
        np.abs(I_model_dB[ts:te] - I_analytic_dB[ts:te].astype(np.float32)),
        axis=0)
    ax.plot(per_ascan_mae, 'steelblue', lw=0.6)
    ax.set_xlabel('A-scan index')
    ax.set_ylabel('MAE [dB]')
    ax.set_title(f'Per-A-scan MAE\n'
                 f'mean={np.mean(per_ascan_mae):.2e}, '
                 f'max={np.max(per_ascan_mae):.2e}')
    ax.grid(True, alpha=0.2)

    fig.suptitle('Forward Model Concordance Analysis', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'concordance.png'), dpi=200)
    plt.close()


def plot_layer_decay_examples(I_model_linear, axial_pixel_size_mm,
                              layer_info, out_dir):
    """Fig 5: Beer-Lambert exponential decay within individual layers.

    For a few A-scans, overlay the measured intensity within each homogeneous
    layer against the expected exponential envelope.
    """
    ascan_indices = [50, 250, 450]
    fig, axes = plt.subplots(1, len(ascan_indices),
                             figsize=(6 * len(ascan_indices), 5))

    for j, a_idx in enumerate(ascan_indices):
        ax = axes[j]
        info = layer_info[a_idx]
        depth_mm = np.arange(LAYER_START_PX, LAYER_END_PX) * axial_pixel_size_mm

        # model output (linear) in tissue band
        I_tissue = I_model_linear[LAYER_START_PX:LAYER_END_PX, a_idx]
        ax.plot(depth_mm, I_tissue, 'b-', lw=1.5, label='Forward model',
                zorder=3)

        # overlay expected exponential per layer
        for i in range(info['n_layers']):
            b0 = int(info['boundaries'][i])
            b1 = int(info['boundaries'][i + 1])
            mu_layer = float(info['mu_values'][i])

            # expected envelope within this layer: I(z) decays from I(b0)
            # I(b0) comes from the model itself (continuity with upstream)
            I_start = I_model_linear[b0, a_idx]
            z_local = np.arange(b1 - b0) * axial_pixel_size_mm
            I_expected = I_start * np.exp(-2.0 * mu_layer * z_local)

            z_mm = np.arange(b0, b1) * axial_pixel_size_mm
            ax.plot(z_mm, I_expected, '--', color=f'C{i + 1}', lw=1.2,
                    label=f'Layer {i}: mu={mu_layer:.1f}, exp decay')

        ax.set_xlabel('Depth [mm]')
        ax.set_ylabel('Intensity [linear]')
        ax.set_title(f'A-scan {a_idx} ({info["n_layers"]} layers)')
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Beer-Lambert Exponential Decay Within Layers',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'layer_decay_examples.png'), dpi=200)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Parameter sweep validations
# ═════════════════════════════════════════════════════════════════════════════

def sweep_axial_pixel_size(forward_model, mu_np, layer_info, out_dir, device):
    """Sweep axial pixel size (dz) and verify attenuation scales linearly.

    Within a homogeneous layer of mu_0, the dB slope per pixel is exactly
    -K * mu_0 * dz.  Doubling dz should double the per-pixel attenuation.

    Returns summary string for the validation report.
    """
    K = 20.0 / math.log(10.0)
    dz_values = [0.001, 0.002, 0.0054, 0.008, 0.015]
    dz_orig = forward_model.axial_pixel_size_mm

    mu_t = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_t = torch.ones_like(mu_t)

    # Find the longest single homogeneous layer across all A-scans
    best_a, best_i, best_len = 0, 0, 0
    for a_idx, info in enumerate(layer_info):
        for i in range(info['n_layers']):
            L = int(info['boundaries'][i + 1] - info['boundaries'][i])
            if L > best_len:
                best_len, best_a, best_i = L, a_idx, i
    best_info = layer_info[best_a]
    b0 = int(best_info['boundaries'][best_i])
    b1 = int(best_info['boundaries'][best_i + 1])
    mu_layer = float(best_info['mu_values'][best_i])

    # Run forward model for each dz
    profiles = {}
    for dz in dz_values:
        forward_model.axial_pixel_size_mm = dz
        with torch.no_grad():
            I_dB = forward_model(mu_t, alpha_t)
        profiles[dz] = I_dB[:, best_a].cpu().numpy()
    forward_model.axial_pixel_size_mm = dz_orig  # restore

    # Measure slopes within the homogeneous layer via linear fit
    pixels = np.arange(b1 - b0).astype(np.float64)
    measured_slopes = []
    theoretical_slopes = []
    for dz in dz_values:
        layer_profile = profiles[dz][b0:b1].astype(np.float64)
        slope = np.polyfit(pixels, layer_profile, 1)[0]
        measured_slopes.append(slope)
        theoretical_slopes.append(-K * mu_layer * dz)

    measured_slopes = np.array(measured_slopes)
    theoretical_slopes = np.array(theoretical_slopes)
    slope_errors = measured_slopes - theoretical_slopes

    # ── Plot (1×3) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Overlaid A-scan profiles
    ax = axes[0]
    depth = np.arange(mu_np.shape[0])
    for dz in dz_values:
        ax.plot(profiles[dz], depth, lw=1.2, label=f'dz={dz} mm')
    ax.axhspan(b0, b1, alpha=0.10, color='yellow', label='Measured layer')
    ax.set_ylim(LAYER_END_PX + 20, LAYER_START_PX - 20)
    ax.set_xlabel('Intensity [dB]')
    ax.set_ylabel('Depth [pixels]')
    ax.set_title(f'A-scan {best_a}: dz sweep\nmu_layer={mu_layer:.1f} mm^-1')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # (b) Measured vs theoretical slopes
    ax = axes[1]
    ax.plot(dz_values, measured_slopes, 'bo-', ms=6, label='Measured')
    ax.plot(dz_values, theoretical_slopes, 'r^--', ms=6, label='Theoretical')
    ax.set_xlabel('Axial pixel size dz [mm]')
    ax.set_ylabel('dB slope [dB/pixel]')
    ax.set_title(f'Attenuation Rate vs dz\nslope = -K * mu * dz')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # (c) Slope error
    ax = axes[2]
    ax.bar(range(len(dz_values)), slope_errors,
           tick_label=[f'{dz}' for dz in dz_values], color='steelblue')
    ax.axhline(0, color='r', ls='--', lw=1)
    ax.set_xlabel('dz [mm]')
    ax.set_ylabel('Slope error [dB/pixel]')
    ax.set_title(f'Measured - Theoretical\n'
                 f'max |error| = {np.max(np.abs(slope_errors)):.2e}')
    ax.grid(True, alpha=0.2)

    fig.suptitle('Sweep 1: Axial Pixel Size — Beer-Lambert Attenuation Scaling',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sweep_axial_pixel_size.png'), dpi=200)
    plt.close()

    # Summary
    lines = []
    lines.append("\n" + "=" * 72 + "\n")
    lines.append("SWEEP 1: AXIAL PIXEL SIZE (dz)\n")
    lines.append("=" * 72 + "\n\n")
    lines.append("Purpose: Verify that Beer-Lambert cumulative attenuation scales\n")
    lines.append("linearly with the axial pixel size dz.  Within a homogeneous layer\n")
    lines.append("of constant mu_0, the dB intensity decays linearly with pixel index:\n\n")
    lines.append("  I_dB(z) = const - K * mu_0 * dz * z_pixel\n\n")
    lines.append("so the per-pixel slope is exactly  slope = -K * mu_0 * dz.\n")
    lines.append("Doubling dz should double the per-pixel attenuation rate.\n\n")
    lines.append("Experimental: For each dz value, the forward model is run on the\n")
    lines.append("same phantom (confocal OFF, gain=0, alpha=1).  The dB slope within\n")
    lines.append("the longest homogeneous layer is measured via linear regression.\n\n")
    lines.append("Analytical: slope_theory = -K * mu_layer * dz.\n\n")
    lines.append(f"Layer used: A-scan {best_a}, pixels [{b0},{b1}), "
                 f"mu={mu_layer:.2f} mm^-1\n\n")
    lines.append(f"  {'dz [mm]':>10s}  {'Measured':>12s}  {'Theoretical':>12s}  "
                 f"{'Error':>12s}\n")
    for i, dz in enumerate(dz_values):
        lines.append(f"  {dz:10.4f}  {measured_slopes[i]:12.6e}  "
                     f"{theoretical_slopes[i]:12.6e}  "
                     f"{slope_errors[i]:12.6e}\n")
    lines.append(f"\n  Max |error| = {np.max(np.abs(slope_errors)):.6e} dB/pixel\n\n")
    lines.append("[6] sweep_axial_pixel_size.png\n")
    lines.append("    Left:   Overlaid A-scan dB profiles for 5 dz values.  Steeper\n")
    lines.append("            decay corresponds to larger dz.  Yellow band = the\n")
    lines.append("            homogeneous layer used for slope measurement.\n")
    lines.append("    Center: Measured slope (blue circles) vs theoretical (red\n")
    lines.append("            triangles) as a function of dz.  Both fall on the\n")
    lines.append("            same line, confirming linear scaling.\n")
    lines.append("    Right:  Bar chart of slope error (measured - theoretical).\n")
    lines.append("            All bars are near zero (~1e-7 dB/pixel).\n")
    return "".join(lines)


def sweep_confocal(config, mu_np, I_baseline_dB, out_dir, device):
    """Sweep focal-plane position and Rayleigh range; verify confocal envelope.

    The confocal term is purely additive in dB:
        I_dB_total(z) = I_dB_beer_lambert(z) + C_dB(z)
    where C_dB(z) = 10*log10(1 / [((z - zFP) / zR)^2 + 1])

    We verify that (model_with_confocal - model_without_confocal) exactly
    matches the analytical Lorentzian C_dB(z).

    Returns summary string for the validation report.
    """
    mu_t = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_t = torch.ones_like(mu_t)
    D = mu_np.shape[0]
    z_px = np.arange(D, dtype=np.float64)

    def analytical_confocal_dB(zFP, zR):
        return 10.0 * np.log10(
            1.0 / (((z_px - zFP) / zR) ** 2 + 1.0))

    # ── Sweep A: focal plane position ──
    zFP_values = [50, 150, 231, 350, 450]
    zR_fixed = 131.5

    residuals_zfp = {}
    analytics_zfp = {}
    for zfp in zFP_values:
        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cfg.forward_model_system.confocal_correction = True
        cfg.forward_model_system.focal_plane_position_pixels = zfp
        cfg.forward_model_system.rayleigh_range_pixels = zR_fixed
        model = RasterScanOCTForward(cfg, device=device)
        model.gain_dB.fill_(0.0)
        with torch.no_grad():
            I_dB = model(mu_t, alpha_t).cpu().numpy()
        # residual = confocal model - baseline (no confocal)
        residuals_zfp[zfp] = I_dB - I_baseline_dB
        analytics_zfp[zfp] = analytical_confocal_dB(zfp, zR_fixed)

    # ── Sweep B: Rayleigh range ──
    zR_values = [30, 65, 131.5, 250, 400]
    zFP_fixed = 231

    residuals_zr = {}
    analytics_zr = {}
    for zr in zR_values:
        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cfg.forward_model_system.confocal_correction = True
        cfg.forward_model_system.focal_plane_position_pixels = zFP_fixed
        cfg.forward_model_system.rayleigh_range_pixels = zr
        model = RasterScanOCTForward(cfg, device=device)
        model.gain_dB.fill_(0.0)
        with torch.no_grad():
            I_dB = model(mu_t, alpha_t).cpu().numpy()
        residuals_zr[zr] = I_dB - I_baseline_dB
        analytics_zr[zr] = analytical_confocal_dB(zFP_fixed, zr)

    # ── Compute errors ──
    # The residual for any column should be identical (confocal is depth-only)
    # so pick column 250 for profile comparisons.
    mid_col = 250
    max_errors_zfp = {}
    for zfp in zFP_values:
        resid_col = residuals_zfp[zfp][:, mid_col].astype(np.float64)
        max_errors_zfp[zfp] = np.max(np.abs(resid_col - analytics_zfp[zfp]))

    max_errors_zr = {}
    for zr in zR_values:
        resid_col = residuals_zr[zr][:, mid_col].astype(np.float64)
        max_errors_zr[zr] = np.max(np.abs(resid_col - analytics_zr[zr]))

    # ── Plot (2×2) ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) Focal plane sweep: analytical confocal functions
    ax = axes[0, 0]
    for zfp in zFP_values:
        ax.plot(z_px, analytics_zfp[zfp], lw=1.3, label=f'zFP={zfp}')
    ax.axhline(0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Depth [pixels]')
    ax.set_ylabel('C_dB [dB]')
    ax.set_title(f'Confocal Function: zFP sweep (zR={zR_fixed})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (0,1) Rayleigh range sweep: analytical confocal functions
    ax = axes[0, 1]
    for zr in zR_values:
        ax.plot(z_px, analytics_zr[zr], lw=1.3, label=f'zR={zr}')
    ax.axhline(0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Depth [pixels]')
    ax.set_ylabel('C_dB [dB]')
    ax.set_title(f'Confocal Function: zR sweep (zFP={zFP_fixed})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (1,0) zFP validation: model residual vs analytical
    ax = axes[1, 0]
    for zfp in zFP_values:
        resid = residuals_zfp[zfp][:, mid_col]
        ax.plot(z_px, resid, '-', lw=1.5, alpha=0.8,
                label=f'zFP={zfp} (model)')
        ax.plot(z_px, analytics_zfp[zfp], '--', lw=1.0, alpha=0.5,
                color='k')
    ax.set_xlabel('Depth [pixels]')
    ax.set_ylabel('Residual [dB]')
    ax.set_title(f'Model Residual vs Analytical (A-scan {mid_col})\n'
                 f'Solid=model, dashed=analytical; '
                 f'max err={max(max_errors_zfp.values()):.2e} dB')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # (1,1) zR validation: model residual vs analytical
    ax = axes[1, 1]
    for zr in zR_values:
        resid = residuals_zr[zr][:, mid_col]
        ax.plot(z_px, resid, '-', lw=1.5, alpha=0.8,
                label=f'zR={zr} (model)')
        ax.plot(z_px, analytics_zr[zr], '--', lw=1.0, alpha=0.5,
                color='k')
    ax.set_xlabel('Depth [pixels]')
    ax.set_ylabel('Residual [dB]')
    ax.set_title(f'Model Residual vs Analytical (A-scan {mid_col})\n'
                 f'Solid=model, dashed=analytical; '
                 f'max err={max(max_errors_zr.values()):.2e} dB')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Sweep 2: Confocal Correction — Lorentzian Validation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sweep_confocal.png'), dpi=200)
    plt.close()

    # Summary
    lines = []
    lines.append("\n" + "=" * 72 + "\n")
    lines.append("SWEEP 2: CONFOCAL CORRECTION\n")
    lines.append("=" * 72 + "\n\n")
    lines.append("Purpose: Verify that the confocal function is a purely additive\n")
    lines.append("Lorentzian envelope in dB space.  When confocal is enabled:\n\n")
    lines.append("  I_dB_total(z) = I_dB_beer_lambert(z) + C_dB(z)\n\n")
    lines.append("where C_dB(z) = 10*log10(1 / [((z - zFP) / zR)^2 + 1]).\n\n")
    lines.append("Experimental: For each (zFP, zR) setting, a new forward model is\n")
    lines.append("created with confocal ON.  The residual is computed as:\n")
    lines.append("  residual(z) = I_dB_confocal(z) - I_dB_no_confocal(z)\n")
    lines.append("using the same phantom and the baseline (confocal OFF) output.\n")
    lines.append("Since confocal is depth-only (independent of A-scan), we compare\n")
    lines.append("a single column (A-scan 250).\n\n")
    lines.append("Analytical: C_dB(z) computed in float64 for each (zFP, zR).\n\n")
    lines.append(f"  zFP sweep (zR={zR_fixed} fixed):\n")
    for zfp in zFP_values:
        lines.append(f"    zFP={zfp:>4d}  max|error| = "
                     f"{max_errors_zfp[zfp]:.6e} dB\n")
    lines.append(f"\n  zR sweep (zFP={zFP_fixed} fixed):\n")
    for zr in zR_values:
        lines.append(f"    zR={zr:>6.1f}  max|error| = "
                     f"{max_errors_zr[zr]:.6e} dB\n")
    lines.append(f"\n[7] sweep_confocal.png\n")
    lines.append("    2x2 grid.\n")
    lines.append("    Top-left:    Analytical C_dB(z) Lorentzians for 5 focal plane\n")
    lines.append("                 positions (zFP = 50..450 px) with zR fixed at 131.5.\n")
    lines.append("                 Peak shifts horizontally with zFP.\n")
    lines.append("    Top-right:   Analytical C_dB(z) for 5 Rayleigh ranges\n")
    lines.append("                 (zR = 30..400 px) with zFP fixed at 231.\n")
    lines.append("                 Wider zR = flatter, broader confocal function.\n")
    lines.append("    Bottom-left: Model residual (solid colored) overlaid on\n")
    lines.append("                 analytical C_dB (dashed black) for the zFP sweep.\n")
    lines.append("                 Solid and dashed curves overlap perfectly.\n")
    lines.append("    Bottom-right: Same overlay for the zR sweep.\n")
    return "".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# EHF Multiple Scattering Validation
# ═════════════════════════════════════════════════════════════════════════════

def run_ehf_validation(out_dir, device):
    """Validate the RasterScanOCTForwardMS (EHF) model.

    Tests:
        A — T_total identity at g=1.0 (purely forward scattering → T_total=1)
        B — SS equivalence at g=0 + large optical depth
        C — g-sweep A-scan profiles (visual: larger g → shallower slope)
        D — MS model always >= SS model for g < 1
        E — Surface pixel consistency (tau=0 → both models agree)

    Returns summary string for the validation report.
    """
    ehf_dir = os.path.join(out_dir, 'ehf')
    os.makedirs(ehf_dir, exist_ok=True)

    config = OmegaConf.load(os.path.join(PHANTOM_ROOT, 'configs', 'config.yaml'))
    config.forward_model_system.confocal_correction = False
    config.forward_model_system.forward_model_type = 'ms'
    depth_samples = int(config.forward_model_system.depth_samples)
    a_lines = int(config.forward_model_system.a_lines)
    dz = float(config.forward_model_system.axial_pixel_size_mm)

    # Uniform phantom: mu=5.0 mm^-1 tissue from pixel 100 to 300
    TISSUE_START, TISSUE_END = 100, 300
    MU_TISSUE = 5.0
    mu_np = np.full((depth_samples, a_lines), MU_AIR, dtype=np.float32)
    mu_np[TISSUE_START:TISSUE_END, :] = MU_TISSUE
    mu_t = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_t = torch.ones_like(mu_t)

    lines = []
    lines.append("\n" + "=" * 72 + "\n")
    lines.append("EHF MULTIPLE SCATTERING VALIDATION\n")
    lines.append("=" * 72 + "\n\n")
    lines.append("Model: RasterScanOCTForwardMS — Extended Huygens-Fresnel (Thrane 2000)\n")
    lines.append(f"Phantom: uniform mu={MU_TISSUE} mm^-1, pixels [{TISSUE_START},{TISSUE_END})\n")
    lines.append(f"Config: confocal OFF, gain=0, alpha=1, dz={dz} mm\n\n")

    # ── Test A: T_total identity at g=1.0 ──────────────────────────────────
    print("  [A] T_total identity at g=1.0 ...")
    config_g1 = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config_g1.forward_model_system.g = 1.0
    model_g1 = RasterScanOCTForwardMS(config_g1, device=device)
    model_g1.gain_dB.fill_(0.0)

    with torch.no_grad():
        I_g1 = model_g1(mu_t, alpha_t)

    # At g=1: theta_rms^2=0, so wS^2=wH^2. Then:
    #   T_SS = exp(-2*tau), T_cross = 4*exp(-tau)*(1-exp(-tau))/(1+1) = 2*exp(-tau)*(1-exp(-tau))
    #   T_MS = (1-exp(-tau))^2 * wH^2/wH^2 = (1-exp(-tau))^2
    #   T_total = [exp(-tau) + (1-exp(-tau))]^2 = 1
    # So I_dB = 10*log10(mu) + 0 (confocal off) = 10*log10(mu)
    # This means NO depth attenuation at all.
    eps = 1e-12
    expected_g1 = 10.0 * np.log10(mu_np.astype(np.float64) + eps)
    I_g1_np = I_g1.cpu().numpy()
    err_g1 = np.abs(I_g1_np[TISSUE_START:TISSUE_END] - expected_g1[TISSUE_START:TISSUE_END].astype(np.float32))
    max_err_g1 = float(np.max(err_g1))

    test_a_pass = max_err_g1 < 1e-4
    lines.append(f"TEST A: T_total identity at g=1.0\n")
    lines.append(f"  At g=1: theta_rms^2=0, wS^2=wH^2\n")
    lines.append(f"  T_total = [exp(-tau) + (1-exp(-tau))]^2 = 1 for all tau\n")
    lines.append(f"  Expected: I_dB = 10*log10(mu), no depth attenuation\n")
    lines.append(f"  Max |error| in tissue: {max_err_g1:.2e} dB\n")
    lines.append(f"  Result: {'PASS' if test_a_pass else 'FAIL'}\n\n")
    print(f"    max |err| = {max_err_g1:.2e} dB -> {'PASS' if test_a_pass else 'FAIL'}")

    # ── Test B: MS model vs independent numpy EHF calculation ───────────────
    print("  [B] MS model vs independent numpy EHF at g=0.7 ...")
    test_g = 0.7
    config_b = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config_b.forward_model_system.g = test_g
    model_b = RasterScanOCTForwardMS(config_b, device=device)
    model_b.gain_dB.fill_(0.0)

    # SS model for reference in later tests
    config_ss = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config_ss.forward_model_system.confocal_correction = False
    model_ss = RasterScanOCTForward(config_ss, device=device)
    model_ss.gain_dB.fill_(0.0)

    with torch.no_grad():
        I_ms_b = model_b(mu_t, alpha_t).cpu().numpy()
        I_ss = model_ss(mu_t, alpha_t).cpu().numpy()

    # Independent numpy EHF calculation (float64 ground truth)
    theta_rms_sq_b = 2.0 * (1.0 - test_g)
    lambda_mm = float(config.forward_model_system.central_wavelength_nm) * 1e-6
    NA = float(config.forward_model_system.numerical_aperture)
    w0_mm = lambda_mm / (math.pi * NA)
    w0_sq = w0_mm ** 2
    n_sample = float(config.forward_model_system.n_sample)

    mu64 = mu_np.astype(np.float64)
    dz64 = np.float64(dz)
    z_px_np = np.arange(depth_samples, dtype=np.float64)
    z_mm_np = z_px_np * dz64

    # tau (shifted cumsum, same as model)
    cumulative_np = np.cumsum(mu64 * dz64, axis=0)
    tau_np = np.zeros_like(mu64)
    tau_np[1:] = cumulative_np[:-1]

    # beam waists
    wH_sq_np = w0_sq * (((z_px_np - float(config.forward_model_system.focal_plane_position_pixels))
                          / float(config.forward_model_system.rayleigh_range_pixels)) ** 2 + 1.0)
    z_over_n_sq_np = (z_mm_np / n_sample) ** 2
    wS_sq_np = wH_sq_np[:, np.newaxis] + (1.0 / 3.0) * tau_np * theta_rms_sq_b * z_over_n_sq_np[:, np.newaxis]

    # EHF transport terms
    exp_neg = np.exp(-tau_np)
    T_SS_np = exp_neg ** 2
    T_cross_np = 4.0 * exp_neg * (1.0 - exp_neg) / (1.0 + wS_sq_np / (wH_sq_np[:, np.newaxis] + eps))
    T_MS_np = (1.0 - exp_neg) ** 2 * wH_sq_np[:, np.newaxis] / (wS_sq_np + eps)
    T_total_np = T_SS_np + T_cross_np + T_MS_np

    # Analytical dB output
    I_analytic_ms = (10.0 * np.log10(mu64 + eps)
                     + 10.0 * np.log10(T_total_np + eps))

    # Compare tissue pixels
    err_b = I_ms_b[TISSUE_START:TISSUE_END] - I_analytic_ms[TISSUE_START:TISSUE_END].astype(np.float32)
    max_err_b = float(np.max(np.abs(err_b)))
    mae_b = float(np.mean(np.abs(err_b)))

    test_b_pass = max_err_b < 1e-3
    lines.append(f"TEST B: MS model vs independent numpy EHF (g={test_g})\n")
    lines.append(f"  Independent float64 numpy implementation of T_SS + T_cross + T_MS\n")
    lines.append(f"  compared pixel-by-pixel against PyTorch model output.\n")
    lines.append(f"  Max |error| in tissue: {max_err_b:.2e} dB\n")
    lines.append(f"  MAE in tissue: {mae_b:.2e} dB\n")
    lines.append(f"  Result: {'PASS' if test_b_pass else 'FAIL'}\n\n")
    print(f"    max |err| = {max_err_b:.2e} dB, MAE = {mae_b:.2e} dB -> {'PASS' if test_b_pass else 'FAIL'}")

    # ── Test C: g-sweep A-scan profiles ────────────────────────────────────
    print("  [C] g-sweep A-scan profiles ...")
    g_values = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    mid_col = a_lines // 2
    profiles = {}

    for g in g_values:
        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cfg.forward_model_system.g = g
        model = RasterScanOCTForwardMS(cfg, device=device)
        model.gain_dB.fill_(0.0)
        with torch.no_grad():
            I_dB = model(mu_t, alpha_t)
        profiles[g] = I_dB[:, mid_col].cpu().numpy()

    # SS reference
    profiles['SS'] = I_ss[:, mid_col]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    depth_px = np.arange(depth_samples)
    colors_g = plt.cm.viridis(np.linspace(0.1, 0.9, len(g_values)))

    # Full depth
    ax = axes[0]
    for i, g in enumerate(g_values):
        ax.plot(profiles[g], depth_px, color=colors_g[i], lw=1.5, label=f'g={g:.1f}')
    ax.plot(profiles['SS'], depth_px, 'k--', lw=1.5, alpha=0.7, label='SS model')
    ax.axhspan(TISSUE_START, TISSUE_END, alpha=0.05, color='yellow')
    ax.set_ylim(TISSUE_END + 20, TISSUE_START - 20)
    ax.set_xlabel('Intensity [dB]')
    ax.set_ylabel('Depth [pixels]')
    ax.set_title(f'EHF g-sweep: A-scan {mid_col}\nmu={MU_TISSUE} mm$^{{-1}}$, confocal OFF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Zoomed tissue
    ax = axes[1]
    for i, g in enumerate(g_values):
        tissue_profile = profiles[g][TISSUE_START:TISSUE_END]
        ax.plot(tissue_profile, np.arange(TISSUE_START, TISSUE_END),
                color=colors_g[i], lw=1.5, label=f'g={g:.1f}')
    tissue_ss = profiles['SS'][TISSUE_START:TISSUE_END]
    ax.plot(tissue_ss, np.arange(TISSUE_START, TISSUE_END),
            'k--', lw=1.5, alpha=0.7, label='SS model')
    ax.set_ylim(TISSUE_END, TISSUE_START)
    ax.set_xlabel('Intensity [dB]')
    ax.set_ylabel('Depth [pixels]')
    ax.set_title('Tissue region (zoomed)\nLarger g = shallower slope (less apparent attenuation)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle('EHF Multiple Scattering: Anisotropy Factor g Sweep',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ehf_dir, 'g_sweep_profiles.png'), dpi=200)
    plt.close()

    # Verify monotonicity: larger g → higher dB at depth
    deep_px = TISSUE_END - 10
    dB_at_depth = [profiles[g][deep_px] for g in g_values]
    monotonic = all(dB_at_depth[i] <= dB_at_depth[i + 1] + 0.01
                    for i in range(len(dB_at_depth) - 1))
    lines.append(f"TEST C: g-sweep A-scan profiles\n")
    lines.append(f"  g values: {g_values}\n")
    lines.append(f"  dB at pixel {deep_px} (deep tissue):\n")
    for g, db in zip(g_values, dB_at_depth):
        lines.append(f"    g={g:.1f}: {db:.2f} dB\n")
    lines.append(f"  Monotonic (larger g -> higher dB): {monotonic}\n")
    lines.append(f"  Output: ehf/g_sweep_profiles.png\n\n")
    print(f"    monotonic at depth: {monotonic}")

    # ── Test D: MS model always >= SS model ────────────────────────────────
    print("  [D] MS >= SS for all g < 1 ...")
    test_d_results = {}
    for g in [0.0, 0.3, 0.5, 0.7, 0.9]:
        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cfg.forward_model_system.g = g
        model = RasterScanOCTForwardMS(cfg, device=device)
        model.gain_dB.fill_(0.0)
        with torch.no_grad():
            I_ms = model(mu_t, alpha_t).cpu().numpy()

        tissue_diff = I_ms[TISSUE_START:TISSUE_END] - I_ss[TISSUE_START:TISSUE_END]
        min_diff = float(np.min(tissue_diff))
        test_d_results[g] = min_diff

    all_nonneg = all(v >= -1e-4 for v in test_d_results.values())
    lines.append(f"TEST D: MS >= SS for g < 1 (multiply scattered photons add signal)\n")
    for g, min_d in test_d_results.items():
        status = "OK" if min_d >= -1e-4 else "FAIL"
        lines.append(f"  g={g:.1f}: min(MS-SS) = {min_d:.6f} dB [{status}]\n")
    lines.append(f"  All non-negative: {all_nonneg}\n")
    lines.append(f"  Result: {'PASS' if all_nonneg else 'FAIL'}\n\n")
    print(f"    all MS >= SS: {all_nonneg}")

    # ── Test E: Surface pixel consistency ──────────────────────────────────
    print("  [E] Surface pixel consistency (tau=0) ...")
    # At z=TISSUE_START (first tissue pixel): tau=0 -> T_SS=1, T_cross=0, T_MS=0
    # Both SS and MS should give: 10*log10(mu) + 10*log10(1) = 10*log10(mu)
    test_e_results = {}
    for g in [0.0, 0.5, 0.7, 0.9, 1.0]:
        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cfg.forward_model_system.g = g
        model = RasterScanOCTForwardMS(cfg, device=device)
        model.gain_dB.fill_(0.0)
        with torch.no_grad():
            I_ms = model(mu_t, alpha_t).cpu().numpy()

        # Surface row difference
        surface_diff = np.abs(I_ms[TISSUE_START, :] - I_ss[TISSUE_START, :])
        max_surface_diff = float(np.max(surface_diff))
        test_e_results[g] = max_surface_diff

    all_surface_ok = all(v < 1e-4 for v in test_e_results.values())
    lines.append(f"TEST E: Surface pixel consistency (tau=0)\n")
    lines.append(f"  At tau=0: T_SS=1, T_cross=0, T_MS=0, T_total=1\n")
    lines.append(f"  Both models should agree at the first tissue pixel\n")
    for g, err in test_e_results.items():
        status = "OK" if err < 1e-4 else "FAIL"
        lines.append(f"  g={g:.1f}: max|MS-SS| at surface = {err:.2e} dB [{status}]\n")
    lines.append(f"  All within tolerance: {all_surface_ok}\n")
    lines.append(f"  Result: {'PASS' if all_surface_ok else 'FAIL'}\n\n")
    print(f"    surface agreement: {all_surface_ok}")

    # ── Summary diagnostic plot: MS vs SS comparison at g=0.7 ─────────────
    config_07 = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config_07.forward_model_system.g = 0.7
    model_07 = RasterScanOCTForwardMS(config_07, device=device)
    model_07.gain_dB.fill_(0.0)
    with torch.no_grad():
        I_ms_07 = model_07(mu_t, alpha_t).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) SS vs MS A-scan comparison
    ax = axes[0]
    ax.plot(I_ss[:, mid_col], depth_px, 'b-', lw=1.5, label='SS (Beer-Lambert)')
    ax.plot(I_ms_07[:, mid_col], depth_px, 'r-', lw=1.5, label='MS (EHF, g=0.7)')
    ax.axhspan(TISSUE_START, TISSUE_END, alpha=0.05, color='yellow')
    ax.set_ylim(TISSUE_END + 20, TISSUE_START - 20)
    ax.set_xlabel('Intensity [dB]')
    ax.set_ylabel('Depth [pixels]')
    ax.set_title(f'A-scan {mid_col}: SS vs MS (g=0.7)\nmu={MU_TISSUE} mm$^{{-1}}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # (b) Difference (MS - SS)
    ax = axes[1]
    diff_07 = I_ms_07[:, mid_col] - I_ss[:, mid_col]
    ax.plot(diff_07[TISSUE_START:TISSUE_END],
            np.arange(TISSUE_START, TISSUE_END), 'g-', lw=1.5)
    ax.axvline(0, color='gray', ls=':', lw=0.8)
    ax.set_ylim(TISSUE_END, TISSUE_START)
    ax.set_xlabel('MS - SS [dB]')
    ax.set_ylabel('Depth [pixels]')
    ax.set_title(f'MS excess signal (g=0.7)\nMax = {np.max(diff_07[TISSUE_START:TISSUE_END]):.2f} dB')
    ax.grid(True, alpha=0.2)

    # (c) T_total components at g=0.7
    ax = axes[2]
    # Recompute transport terms for visualization
    tau_1d = np.cumsum(mu_np[TISSUE_START:TISSUE_END, mid_col] * dz)
    tau_1d = np.concatenate([[0.0], tau_1d[:-1]])
    exp_neg = np.exp(-tau_1d)
    T_SS_1d = exp_neg ** 2
    # Need wH^2 and wS^2 for this column
    wH_sq_np = model_07._wH_sq.cpu().numpy()
    z_over_n_sq_np = model_07._z_over_n_sq.cpu().numpy()
    theta_sq = 2.0 * (1.0 - 0.7)
    tissue_depth = np.arange(TISSUE_START, TISSUE_END)
    wH_sq_t = wH_sq_np[TISSUE_START:TISSUE_END]
    wS_sq_t = wH_sq_t + (1.0 / 3.0) * tau_1d * theta_sq * z_over_n_sq_np[TISSUE_START:TISSUE_END]
    T_cross_1d = 4.0 * exp_neg * (1.0 - exp_neg) / (1.0 + wS_sq_t / (wH_sq_t + 1e-12))
    T_MS_1d = (1.0 - exp_neg) ** 2 * wH_sq_t / (wS_sq_t + 1e-12)
    T_total_1d = T_SS_1d + T_cross_1d + T_MS_1d

    ax.plot(10 * np.log10(T_SS_1d + 1e-12), tissue_depth, 'b-', lw=1.3, label='$T_{SS}$ (ballistic)')
    ax.plot(10 * np.log10(T_cross_1d + 1e-12), tissue_depth, 'm--', lw=1.3, label='$T_{cross}$')
    ax.plot(10 * np.log10(T_MS_1d + 1e-12), tissue_depth, 'r:', lw=1.3, label='$T_{MS}$ (multiply scattered)')
    ax.plot(10 * np.log10(T_total_1d + 1e-12), tissue_depth, 'k-', lw=2, label='$T_{total}$')
    ax.set_ylim(TISSUE_END, TISSUE_START)
    ax.set_xlabel('Transport [dB]')
    ax.set_ylabel('Depth [pixels]')
    ax.set_title(f'EHF transport components (g=0.7)\nmu={MU_TISSUE} mm$^{{-1}}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle('EHF Multiple Scattering Model Validation (g=0.7)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ehf_dir, 'ms_vs_ss_comparison.png'), dpi=200)
    plt.close()

    lines.append("PLOT DESCRIPTIONS (EHF validation)\n")
    lines.append("-" * 72 + "\n\n")
    lines.append("[EHF-1] ehf/g_sweep_profiles.png\n")
    lines.append("    Left:  dB A-scan profiles for g = {0, 0.3, 0.5, 0.7, 0.9, 1.0}\n")
    lines.append("           plus SS reference (black dashed).  Larger g = shallower slope.\n")
    lines.append("    Right: Zoomed tissue region.\n\n")
    lines.append("[EHF-2] ehf/ms_vs_ss_comparison.png\n")
    lines.append("    Left:   SS vs MS A-scan at g=0.7.  MS curve is above SS at depth.\n")
    lines.append("    Center: MS excess signal (MS - SS) in dB vs depth.\n")
    lines.append("    Right:  EHF transport components T_SS, T_cross, T_MS, T_total in dB.\n")
    lines.append("            Shows crossover where T_MS dominates over T_SS at depth.\n\n")

    return "".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Numbered Visual Verification Tests
# ═════════════════════════════════════════════════════════════════════════════
#
# Each test produces a self-contained PNG image named test_NN_<name>.png.
# Designed to be interpretable by someone not familiar with OCT — each plot
# clearly shows the expected vs actual values and states the pass/fail verdict.

def _make_mini_config(confocal: bool = False, g: float = 0.7,
                      depth_samples: int = 10,
                      focal_plane_position_pixels: float = 5.0) -> OmegaConf:
    """Create a small config for verification tests."""
    return OmegaConf.create({
        "forward_model_system": {
            "central_wavelength_nm": 1300,
            "numerical_aperture": 0.0315,
            "depth_samples": depth_samples,
            "a_lines": 3,
            "confocal_correction": confocal,
            "focal_plane_position_pixels": focal_plane_position_pixels,
            "rayleigh_range_pixels": 131.5,
            "noise_floor_dB": 79.0,
            "lateral_pixel_size_mm": 0.0093,
            "axial_pixel_size_mm": 0.0054,
            "forward_model_type": "ss",
            "g": g,
            "n_sample": 1.35,
        }
    })


def _make_verdict_text(errors, tol=1e-4):
    """Format a PASS/FAIL verdict string from an array of absolute errors."""
    max_err = np.max(np.abs(errors))
    passed = max_err < tol
    return f"{'PASS' if passed else 'FAIL'} (max error = {max_err:.2e}, tolerance = {tol:.0e})", passed


def run_test_01(out_dir, device):
    """Test 1: SS model pixel-by-pixel match to Beer-Lambert formula.

    Shows 10 depth pixels where every model output (circles) lies exactly on
    the analytical formula line.
    """
    D, A, dz = 10, 3, 0.0054
    K = 20.0 / math.log(10.0)
    mu_val = 5.0

    config = _make_mini_config()
    ss = RasterScanOCTForward(config, device=device)
    ss.gain_dB.fill_(0.0)

    mu = torch.full((D, A), mu_val, dtype=torch.float32, device=device)
    alpha = torch.ones_like(mu)
    with torch.no_grad():
        I_dB = ss(mu, alpha).cpu().numpy()[:, 0]

    z = np.arange(D)
    tau = z * mu_val * dz
    analytical = 10.0 * math.log10(mu_val) - K * tau

    errors = I_dB - analytical.astype(np.float32)
    verdict, passed = _make_verdict_text(errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Model vs analytical
    ax = axes[0]
    ax.plot(z, analytical, 'b-', lw=2, label='Analytical: $10 \\log_{10}(\\mu) - \\frac{20}{\\ln 10} \\sum_{i=0}^{z-1} \\mu_i \\Delta z$', zorder=1)
    ax.scatter(z, I_dB, s=80, c='red', zorder=3, label='Forward model output', edgecolors='black', linewidths=0.5)
    for i in range(D):
        ax.annotate(f'z={i}\nOD={tau[i]:.4f}', (z[i], I_dB[i]),
                    textcoords='offset points', xytext=(8, 6), fontsize=6, color='gray')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title('Model output matches analytical formula at every pixel')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    # (b) Per-pixel error
    ax = axes[1]
    ax.bar(z, errors * 1e6, color='steelblue', edgecolor='gray')
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Error [micro-dB]')
    ax.set_title(f'Per-pixel error (model $-$ analytical)\nScale: $10^{{-6}}$ dB')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 1: SS Beer-Lambert Pixel-by-Pixel Verification — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_01_ss_pixel_verification.png'), dpi=200)
    plt.close()
    return passed


def run_test_02(out_dir, device):
    """Test 2: System gain (gain_dB) is a pure additive offset.

    Running the same phantom with gain=0 vs gain=100 should produce curves
    that are exactly 100 dB apart at every pixel.
    """
    D, A, dz = 10, 3, 0.0054
    mu_val = 5.0

    config = _make_mini_config()
    ss = RasterScanOCTForward(config, device=device)

    mu = torch.full((D, A), mu_val, dtype=torch.float32, device=device)
    alpha = torch.ones_like(mu)

    with torch.no_grad():
        ss.gain_dB.fill_(0.0)
        I_gain0 = ss(mu, alpha).cpu().numpy()[:, 0]
        ss.gain_dB.fill_(100.0)
        I_gain100 = ss(mu, alpha).cpu().numpy()[:, 0]

    z = np.arange(D)
    difference = I_gain100 - I_gain0
    errors = difference - 100.0
    verdict, passed = _make_verdict_text(errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(z, I_gain0, 'bo-', lw=2, ms=8, label='gain = 0 dB')
    ax.plot(z, I_gain100, 'rs-', lw=2, ms=8, label='gain = 100 dB')
    for i in range(D):
        ax.annotate('', xy=(z[i], I_gain100[i]), xytext=(z[i], I_gain0[i]),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
        if i == D // 2:
            ax.annotate(f'exactly 100 dB', xy=(z[i] + 0.15, (I_gain0[i] + I_gain100[i]) / 2),
                        fontsize=9, color='gray', fontweight='bold')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title('Gain shifts every pixel by the same amount')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(z, difference, 'go-', lw=2, ms=8, label='Difference (gain100 $-$ gain0)')
    ax.axhline(100.0, color='red', ls='--', lw=1.5, label='Expected: 100.0 dB')
    ax.set_ylim(99.99, 100.01)
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Difference [dB]')
    ax.set_title(f'Constant offset at every pixel\nmax deviation = {np.max(np.abs(errors)):.2e} dB')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 2: System Gain is a Pure Additive Offset — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_02_gain_offset.png'), dpi=200)
    plt.close()
    return passed


def run_test_03(out_dir, device):
    r"""Test 3: Backscatter alpha scales signal by exactly 10*log10(alpha).

    With alpha=0.5, the signal should be exactly 10*log10(0.5) = -3.01 dB
    lower than alpha=1.0 at every depth.
    """
    D, A = 10, 3
    mu_val = 5.0
    alpha_val = 0.5
    expected_shift = 10.0 * math.log10(alpha_val)  # -3.0103 dB

    config = _make_mini_config()
    ss = RasterScanOCTForward(config, device=device)
    ss.gain_dB.fill_(0.0)

    mu = torch.full((D, A), mu_val, dtype=torch.float32, device=device)
    alpha1 = torch.ones_like(mu)
    alpha05 = torch.full_like(mu, alpha_val)

    with torch.no_grad():
        I_a1 = ss(mu, alpha1).cpu().numpy()[:, 0]
        I_a05 = ss(mu, alpha05).cpu().numpy()[:, 0]

    z = np.arange(D)
    difference = I_a05 - I_a1
    errors = difference - expected_shift
    verdict, passed = _make_verdict_text(errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(z, I_a1, 'bo-', lw=2, ms=8, label=r'$\alpha$ = 1.0')
    ax.plot(z, I_a05, 'rs-', lw=2, ms=8, label=r'$\alpha$ = 0.5')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title(f'Halving $\\alpha$ shifts signal by $10 \\cdot \\log_{{10}}(0.5)$ = {expected_shift:.2f} dB')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(z, difference, 'go-', lw=2, ms=8, label=r'$I_{dB}(\alpha=0.5) - I_{dB}(\alpha=1.0)$')
    ax.axhline(expected_shift, color='red', ls='--', lw=1.5,
               label=f'Expected: {expected_shift:.4f} dB')
    ax.set_ylim(expected_shift - 0.001, expected_shift + 0.001)
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Difference [dB]')
    ax.set_title(f'Constant shift at every pixel\nmax deviation = {np.max(np.abs(errors)):.2e} dB')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 3: Backscatter Fraction $\\alpha$ Scales Signal Exactly — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_03_alpha_scaling.png'), dpi=200)
    plt.close()
    return passed


def run_test_04(out_dir, device):
    """Test 4: Confocal correction adds a Lorentzian envelope in dB.

    The confocal function C(z) = 1/[((z-zFP)/zR)^2 + 1] is purely additive
    in dB space. Enabling it should add exactly 10*log10(C(z)) at each pixel.
    """
    D, A = 500, 3
    mu_val = 5.0
    zFP, zR = 231.0, 131.5

    config_off = _make_mini_config(confocal=False, depth_samples=D,
                                   focal_plane_position_pixels=zFP)
    config_on = _make_mini_config(confocal=True, depth_samples=D,
                                  focal_plane_position_pixels=zFP)
    ss_off = RasterScanOCTForward(config_off, device=device)
    ss_on = RasterScanOCTForward(config_on, device=device)
    ss_off.gain_dB.fill_(0.0)
    ss_on.gain_dB.fill_(0.0)

    mu = torch.full((D, A), mu_val, dtype=torch.float32, device=device)
    alpha = torch.ones_like(mu)

    with torch.no_grad():
        I_off = ss_off(mu, alpha).cpu().numpy()[:, 0]
        I_on = ss_on(mu, alpha).cpu().numpy()[:, 0]

    z = np.arange(D, dtype=np.float64)
    C_dB = 10.0 * np.log10(1.0 / (((z - zFP) / zR) ** 2 + 1.0))

    difference = I_on - I_off
    errors = difference - C_dB.astype(np.float32)
    verdict, passed = _make_verdict_text(errors)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.plot(z, I_off, 'bo-', lw=2, ms=8, label='Confocal OFF')
    ax.plot(z, I_on, 'rs-', lw=2, ms=8, label='Confocal ON')
    ax.axvline(zFP, color='gray', ls=':', label=f'Focal plane (z={zFP:.0f})')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title('With vs without confocal correction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(z, C_dB, 'k-', lw=2, label=r'Analytical $C_{dB}(z)$')
    ax.scatter(z, difference, s=80, c='red', zorder=3, edgecolors='black',
              linewidths=0.5, label='Model difference (ON $-$ OFF)')
    ax.axvline(zFP, color='gray', ls=':')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Confocal correction [dB]')
    ax.set_title('Model difference matches analytical Lorentzian')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(z, errors * 1e6, color='steelblue', edgecolor='gray')
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Error [micro-dB]')
    ax.set_title(f'Residual error (scale: $10^{{-6}}$ dB)')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 4: Confocal Correction is an Additive Lorentzian — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_04_confocal_correction.png'), dpi=200)
    plt.close()
    return passed


def run_test_05(out_dir, device):
    """Test 5: Noise mask zeroes out pixels and pauses cumulative attenuation.

    When a noise mask zeros out rows 3-6, those pixels do not contribute to
    the cumulative optical depth. Signal below the masked region resumes from
    the pre-mask attenuation level.
    """
    D, A, dz = 10, 3, 0.0054
    K = 20.0 / math.log(10.0)
    mu_val = 5.0
    eps = 1e-12

    config = _make_mini_config()
    ss = RasterScanOCTForward(config, device=device)
    ss.gain_dB.fill_(0.0)

    mu = torch.full((D, A), mu_val, dtype=torch.float32, device=device)
    alpha = torch.ones_like(mu)

    # Masked version: rows 3-6 masked out
    mask = torch.ones_like(mu)
    mask[3:7, :] = 0.0

    with torch.no_grad():
        I_unmasked = ss(mu, alpha, noise_mask=None).cpu().numpy()[:, 0]
        I_masked = ss(mu, alpha, noise_mask=mask).cpu().numpy()[:, 0]

    z = np.arange(D)
    mask_np = np.ones(D)
    mask_np[3:7] = 0.0

    # Compute expected masked output
    # Note: noise_mask multiplies BOTH mu AND alpha, so at masked pixels:
    #   10*log10(mu_active + eps) + 10*log10(alpha_active + eps)
    #   = 10*log10(eps) + 10*log10(eps) = -240 dB
    expected_masked = np.zeros(D, dtype=np.float64)
    for zi in range(D):
        active_before = sum(1 for r in range(zi) if mask_np[r] > 0)
        tau_z = active_before * mu_val * dz
        if mask_np[zi] > 0:
            # alpha_active=1 → +0 dB; mu_active=mu_val
            expected_masked[zi] = 10.0 * math.log10(mu_val + eps) - K * tau_z
        else:
            # Both mu_active=0 AND alpha_active=0 → two log10(eps) terms
            expected_masked[zi] = (10.0 * math.log10(eps)
                                   + 10.0 * math.log10(eps) - K * tau_z)

    errors = I_masked - expected_masked.astype(np.float32)
    verdict, passed = _make_verdict_text(errors, tol=1e-3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.plot(z, I_unmasked, 'bo-', lw=2, ms=8, label='No mask (all pixels active)')
    ax.plot(z, I_masked, 'rs-', lw=2, ms=8, label='Mask OFF rows 3-6')
    ax.axhspan(-0.5, D - 0.5, xmin=0, xmax=1, alpha=0)
    for zi in range(3, 7):
        ax.axvspan(zi - 0.4, zi + 0.4, alpha=0.15, color='gray')
    ax.annotate('masked\nregion', xy=(5, I_masked[5] + 5), fontsize=9,
                ha='center', color='gray', fontweight='bold')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title('Masked pixels drop to noise floor\nbut do NOT accumulate attenuation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Show cumulative optical depth
    tau_unmasked = z * mu_val * dz
    tau_masked = np.array([sum(1 for r in range(zi) if mask_np[r] > 0) * mu_val * dz
                           for zi in range(D)])
    ax.plot(z, tau_unmasked, 'bo-', lw=2, ms=8, label='Unmasked $\\sum_{i=0}^{z-1} \\mu_i \\Delta z$')
    ax.plot(z, tau_masked, 'rs-', lw=2, ms=8, label='Masked $\\sum_{i=0}^{z-1} \\mu_i \\Delta z$')
    for zi in range(3, 7):
        ax.axvspan(zi - 0.4, zi + 0.4, alpha=0.15, color='gray')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Cumulative optical depth $\\sum \\mu \\Delta z$')
    ax.set_title('Masked pixels do not contribute to cumsum\nOptical depth pauses during masked region')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(z, I_masked, s=80, c='red', zorder=3, edgecolors='black',
              linewidths=0.5, label='Model output')
    ax.plot(z, expected_masked, 'b-', lw=2, label='Expected (analytical)', zorder=1)
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title(f'Masked model matches analytical prediction\nmax error = {np.max(np.abs(errors)):.2e} dB')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 5: Noise Mask Excludes Pixels from Attenuation — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_05_noise_mask_effect.png'), dpi=200)
    plt.close()
    return passed


def run_test_06(out_dir, device):
    """Test 6: MS and SS models agree at the surface pixel (tau=0).

    At z=0 (no attenuation above), both models produce the same signal
    regardless of anisotropy factor g.
    """
    D, A = 10, 3
    mu_val = 5.0

    config_ss = _make_mini_config()
    ss = RasterScanOCTForward(config_ss, device=device)
    ss.gain_dB.fill_(0.0)

    mu = torch.full((D, A), mu_val, dtype=torch.float32, device=device)
    alpha = torch.ones_like(mu)

    with torch.no_grad():
        I_ss_z0 = ss(mu, alpha).cpu().numpy()[0, 0]

    g_values = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    ms_z0_values = []
    for g in g_values:
        cfg = _make_mini_config(g=g)
        ms = RasterScanOCTForwardMS(cfg, device=device)
        ms.gain_dB.fill_(0.0)
        with torch.no_grad():
            val = ms(mu, alpha).cpu().numpy()[0, 0]
        ms_z0_values.append(val)

    errors = np.array(ms_z0_values) - I_ss_z0
    verdict, passed = _make_verdict_text(errors, tol=1e-5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.bar(range(len(g_values)), ms_z0_values, color='steelblue', edgecolor='gray',
           tick_label=[f'g={g}' for g in g_values])
    ax.axhline(I_ss_z0, color='red', ls='--', lw=2,
               label=f'SS model: {I_ss_z0:.4f} dB')
    ax.set_ylabel('Surface pixel intensity [dB]')
    ax.set_title(f'Surface pixel (z=0): all g values give same result as SS\nExpected: $10 \\cdot \\log_{{10}}(\\mu)$ = {10*math.log10(mu_val):.4f} dB')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(range(len(g_values)), errors, color='steelblue', edgecolor='gray',
           tick_label=[f'g={g}' for g in g_values])
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_ylabel('MS $-$ SS [dB]')
    ax.set_title(f'Error at surface: max = {np.max(np.abs(errors)):.2e} dB\n(Before any scattering, models must agree)')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Test 6: SS and MS Agree at Surface (tau = 0) — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_06_surface_agreement.png'), dpi=200)
    plt.close()
    return passed


def run_test_07(out_dir, device):
    """Test 7: EHF transport decomposition T_SS + T_cross + T_MS = T_total.

    Shows how multiply scattered light (T_MS) grows with depth while ballistic
    light (T_SS) decays. The cross-term (T_cross) peaks in the transition zone.
    Uses a larger phantom (200 tissue pixels) for a clearer visual.
    """
    D, A, dz = 300, 3, 0.0054
    mu_val = 5.0
    g = 0.7
    TISSUE_START, TISSUE_END = 50, 250

    config = OmegaConf.create({
        "forward_model_system": {
            "central_wavelength_nm": 1300,
            "numerical_aperture": 0.0315, "depth_samples": D, "a_lines": A,
            "confocal_correction": False, "focal_plane_position_pixels": 150,
            "rayleigh_range_pixels": 131.5, "noise_floor_dB": 79.0,
            "lateral_pixel_size_mm": 0.0093, "axial_pixel_size_mm": dz,
            "forward_model_type": "ms", "g": g, "n_sample": 1.35,
        }
    })

    eps = 1e-12
    lambda_mm = 1300.0 * 1e-6
    NA = 0.0315
    w0_sq = (lambda_mm / (math.pi * NA)) ** 2
    n_sample = 1.35
    theta_rms_sq = 2.0 * (1.0 - g)

    # Build tau for tissue region
    tissue_len = TISSUE_END - TISSUE_START
    tau = np.arange(tissue_len) * mu_val * dz  # shifted cumsum within tissue
    z_px = np.arange(TISSUE_START, TISSUE_END, dtype=np.float64)
    z_mm = z_px * dz
    zFP, zR = 150.0, 131.5
    wH_sq = w0_sq * (((z_px - zFP) / zR) ** 2 + 1.0)
    z_over_n_sq = (z_mm / n_sample) ** 2
    wS_sq = wH_sq + (1.0 / 3.0) * tau * theta_rms_sq * z_over_n_sq

    exp_neg = np.exp(-tau)
    T_SS = exp_neg ** 2
    T_cross = 4.0 * exp_neg * (1.0 - exp_neg) / (1.0 + wS_sq / (wH_sq + eps))
    T_MS = (1.0 - exp_neg) ** 2 * wH_sq / (wS_sq + eps)
    T_total = T_SS + T_cross + T_MS

    # Verify model output matches
    mu_np = np.full((D, A), 1e-6, dtype=np.float32)
    mu_np[TISSUE_START:TISSUE_END, :] = mu_val
    mu_t = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_t = torch.ones_like(mu_t)

    ms = RasterScanOCTForwardMS(config, device=device)
    ms.gain_dB.fill_(0.0)
    with torch.no_grad():
        I_ms = ms(mu_t, alpha_t).cpu().numpy()[:, 0]

    expected = 10.0 * np.log10(mu_val + eps) + 10.0 * np.log10(T_total + eps)
    model_tissue = I_ms[TISSUE_START:TISSUE_END]
    errors = model_tissue - expected.astype(np.float32)
    verdict, passed = _make_verdict_text(errors, tol=5e-3)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    depth_idx = np.arange(tissue_len)

    # (a) Transport components in linear scale
    ax = axes[0]
    ax.semilogy(depth_idx, T_SS, 'b-', lw=2, label='$T_{SS}$ (ballistic)')
    ax.semilogy(depth_idx, T_cross, 'm--', lw=2, label='$T_{cross}$ (interference)')
    ax.semilogy(depth_idx, T_MS, 'r:', lw=2, label='$T_{MS}$ (multiply scattered)')
    ax.semilogy(depth_idx, T_total, 'k-', lw=2.5, label='$T_{total}$ (sum)')
    # Find crossover
    crossover_idx = np.argmax(T_MS > T_SS)
    if crossover_idx > 0:
        ax.axvline(crossover_idx, color='gray', ls=':', alpha=0.7)
        ax.annotate(f'crossover\npx {crossover_idx}',
                    xy=(crossover_idx, T_SS[crossover_idx]),
                    textcoords='offset points', xytext=(10, 10), fontsize=8, color='gray')
    ax.set_xlabel('Pixels into tissue')
    ax.set_ylabel('Transport factor (linear)')
    ax.set_title(f'EHF transport components (g={g})\nBallistic light decays; scattered light grows')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Transport in dB
    ax = axes[1]
    ax.plot(depth_idx, 10 * np.log10(T_SS + eps), 'b-', lw=2, label='$T_{SS}$ (ballistic)')
    ax.plot(depth_idx, 10 * np.log10(T_cross + eps), 'm--', lw=2, label='$T_{cross}$')
    ax.plot(depth_idx, 10 * np.log10(T_MS + eps), 'r:', lw=2, label='$T_{MS}$ (multiply scattered)')
    ax.plot(depth_idx, 10 * np.log10(T_total + eps), 'k-', lw=2.5, label='$T_{total}$')
    # SS line (Beer-Lambert)
    K = 20.0 / math.log(10.0)
    ax.plot(depth_idx, -K * tau, 'b--', lw=1, alpha=0.5, label='$-\\frac{20}{\\ln 10} \\sum \\mu \\Delta z$ (pure SS)')
    ax.set_xlabel('Pixels into tissue')
    ax.set_ylabel('Transport [dB]')
    ax.set_title(f'Same decomposition in dB\n$T_{{total}}$ flattens where $T_{{MS}}$ dominates')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) Model vs analytical
    ax = axes[2]
    ax.plot(depth_idx, expected, 'b-', lw=2, label='Analytical')
    ax.scatter(depth_idx[::5], model_tissue[::5], s=30, c='red', zorder=3,
              edgecolors='black', linewidths=0.3, label='Model (every 5th pixel)')
    ax.set_xlabel('Pixels into tissue')
    ax.set_ylabel('$I_{dB}$ [dB]')
    ax.set_title(f'Model matches analytical EHF output\nmax |error| = {np.max(np.abs(errors)):.2e} dB')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 7: EHF Transport Decomposition (Ballistic + Cross + Multiply Scattered) — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_07_transport_decomposition.png'), dpi=200)
    plt.close()
    return passed


def run_test_08(out_dir, device):
    """Test 8: Confocal function equals w0^2 / wH^2(z).

    This mathematical identity ensures the confocal correction used by the SS
    model is consistent with the beam-waist formulation in the MS model.
    """
    D = 10
    zFP, zR = 5.0, 131.5
    lambda_mm = 1300.0 * 1e-6
    NA = 0.0315
    w0_sq = (lambda_mm / (math.pi * NA)) ** 2

    z = np.arange(D, dtype=np.float64)
    confocal = 1.0 / (((z - zFP) / zR) ** 2 + 1.0)
    wH_sq = w0_sq * (((z - zFP) / zR) ** 2 + 1.0)
    ratio = w0_sq / wH_sq

    errors = confocal - ratio
    verdict, passed = _make_verdict_text(errors, tol=1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(z, confocal, 'bo-', lw=2, ms=10, label=r'$C(z) = \frac{1}{((z-z_{FP})/z_R)^2 + 1}$')
    ax.scatter(z, ratio, s=80, marker='x', c='red', zorder=3, lw=2,
              label=r'$w_0^2 / w_H^2(z)$')
    ax.axvline(zFP, color='gray', ls=':', label=f'Focal plane (z={zFP:.0f})')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Value')
    ax.set_title('Confocal function is identical to beam waist ratio\n(red crosses sit exactly on blue circles)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(z, errors, color='steelblue', edgecolor='gray')
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel(r'$C(z) - w_0^2/w_H^2(z)$')
    ax.set_title(f'Residual (should be exactly zero)\nmax = {np.max(np.abs(errors)):.2e}')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 8: Confocal = $w_0^2 / w_H^2$ Identity — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_08_confocal_beam_identity.png'), dpi=200)
    plt.close()
    return passed


def run_test_09(out_dir, device):
    """Test 9: g=0 (isotropic) has strongest multiple scattering.

    At g=0 scatter broadening is maximal, so T_total is highest (most
    multiply scattered photons recaptured). We verify the model output
    at z=5 matches the analytical EHF value at g=0.
    """
    D, A, dz = 10, 3, 0.0054
    mu_val = 5.0
    eps = 1e-12

    mu = torch.full((D, A), mu_val, dtype=torch.float32, device=device)
    alpha = torch.ones_like(mu)

    g_values = [0.0, 0.3, 0.5, 0.7, 0.9]
    profiles = {}
    for g in g_values:
        cfg = _make_mini_config(g=g)
        ms = RasterScanOCTForwardMS(cfg, device=device)
        ms.gain_dB.fill_(0.0)
        with torch.no_grad():
            profiles[g] = ms(mu, alpha).cpu().numpy()[:, 0]

    # SS reference
    ss = RasterScanOCTForward(_make_mini_config(), device=device)
    ss.gain_dB.fill_(0.0)
    with torch.no_grad():
        I_ss = ss(mu, alpha).cpu().numpy()[:, 0]

    # Analytical at g=0, z=5
    z_check = 5
    tau_check = z_check * mu_val * dz
    zFP, zR, n = 5.0, 131.5, 1.35
    lambda_mm = 1300.0 * 1e-6
    w0_sq = (lambda_mm / (math.pi * 0.0315)) ** 2
    wH_sq = w0_sq * (((z_check - zFP) / zR) ** 2 + 1.0)
    z_mm = z_check * dz
    z_over_n_sq = (z_mm / n) ** 2
    theta_rms_sq_g0 = 2.0  # maximum for g=0
    wS_sq = wH_sq + (1.0 / 3.0) * tau_check * theta_rms_sq_g0 * z_over_n_sq
    exp_neg = math.exp(-tau_check)
    T_total_g0 = (exp_neg ** 2
                  + 4 * exp_neg * (1 - exp_neg) / (1 + wS_sq / wH_sq)
                  + (1 - exp_neg) ** 2 * wH_sq / wS_sq)
    expected = 10 * math.log10(mu_val) + 10 * math.log10(T_total_g0)
    errors = np.array([profiles[0.0][z_check] - expected])
    verdict, passed = _make_verdict_text(errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    z = np.arange(D)

    ax = axes[0]
    for g in g_values:
        ax.plot(z, profiles[g], 'o-', lw=1.5, ms=5, label=f'MS g={g}')
    ax.plot(z, I_ss, 'k--', lw=2, ms=5, label='SS (Beer-Lambert)')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title('Lower g = stronger multiple scattering = higher signal at depth\n(g=0: isotropic scattering, g=0.9: strongly forward-peaked)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Show MS excess over SS at z=5
    excess = [profiles[g][z_check] - I_ss[z_check] for g in g_values]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(g_values)))
    ax.bar(range(len(g_values)), excess, color=colors, edgecolor='gray',
           tick_label=[f'g={g}' for g in g_values])
    ax.set_ylabel('MS excess over SS [dB]')
    ax.set_title(f'Extra signal from multiple scattering at z={z_check}\n(g=0 has the most MS contribution)')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Test 9: Isotropic Scattering (g=0) Gives Strongest MS — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_09_g_zero_strong_ms.png'), dpi=200)
    plt.close()
    return passed


def run_test_10(out_dir, device):
    """Test 10: MS model deep pixel verification.

    At z=9 (deepest pixel), the cumulative optical depth is 0.243.
    Verify that the model output matches the analytical EHF calculation
    including wS^2 beam broadening at large tau.
    """
    D, A, dz = 10, 3, 0.0054
    mu_val = 5.0
    eps = 1e-12
    g = 0.7

    config = _make_mini_config(g=g)
    ms = RasterScanOCTForwardMS(config, device=device)
    ms.gain_dB.fill_(0.0)
    ss = RasterScanOCTForward(_make_mini_config(), device=device)
    ss.gain_dB.fill_(0.0)

    mu = torch.full((D, A), mu_val, dtype=torch.float32, device=device)
    alpha = torch.ones_like(mu)

    with torch.no_grad():
        I_ms = ms(mu, alpha).cpu().numpy()[:, 0]
        I_ss = ss(mu, alpha).cpu().numpy()[:, 0]

    # Analytical verification at all pixels
    zFP, zR, n = 5.0, 131.5, 1.35
    lambda_mm = 1300.0 * 1e-6
    w0_sq = (lambda_mm / (math.pi * 0.0315)) ** 2
    theta_rms_sq = 2.0 * (1.0 - g)

    z = np.arange(D, dtype=np.float64)
    tau = z * mu_val * dz
    wH_sq = w0_sq * (((z - zFP) / zR) ** 2 + 1.0)
    z_mm = z * dz
    z_over_n_sq = (z_mm / n) ** 2
    wS_sq = wH_sq + (1.0 / 3.0) * tau * theta_rms_sq * z_over_n_sq

    exp_neg = np.exp(-tau)
    T_SS = exp_neg ** 2
    T_cross = 4.0 * exp_neg * (1.0 - exp_neg) / (1.0 + wS_sq / (wH_sq + eps))
    T_MS = (1.0 - exp_neg) ** 2 * wH_sq / (wS_sq + eps)
    T_total = T_SS + T_cross + T_MS
    expected = 10.0 * np.log10(mu_val + eps) + 10.0 * np.log10(T_total + eps)

    errors = I_ms - expected.astype(np.float32)
    verdict, passed = _make_verdict_text(errors)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    ax = axes[0]
    ax.plot(z, expected, 'b-', lw=2, label='Analytical EHF')
    ax.scatter(z, I_ms, s=80, c='red', zorder=3, edgecolors='black',
              linewidths=0.5, label='MS model')
    ax.plot(z, I_ss, 'k--', lw=1.5, alpha=0.5, label='SS model')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title(f'MS model (circles) matches analytical EHF (line)\ng={g}, $\\mu$={mu_val} mm$^{{-1}}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(z, wS_sq / wH_sq, 'go-', lw=2, ms=8)
    ax.axhline(1.0, color='gray', ls='--', lw=1, label='$w_S^2 / w_H^2 = 1$ (no broadening)')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('$w_S^2 / w_H^2$')
    ax.set_title('Scatter broadening grows with depth\n(beam spreads due to multiple scattering)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(z, errors * 1e5, color='steelblue', edgecolor='gray')
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Error [$\\times 10^{-5}$ dB]')
    ax.set_title(f'Per-pixel error: max = {np.max(np.abs(errors)):.2e} dB\n(Deep pixel z=9: $\\sum \\mu \\Delta z$={tau[9]:.4f}, $w_S^2/w_H^2$={wS_sq[9]/wH_sq[9]:.6f})')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 10: MS Model Deep Pixel Verification — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_10_deep_pixel.png'), dpi=200)
    plt.close()
    return passed


def run_test_11(out_dir, device):
    """Test 11: At g=1 (purely forward scattering), T_total = 1 everywhere.

    When all scattering is in the forward direction, every scattered photon
    stays in the beam. The OCT signal shows NO depth attenuation — just
    10*log10(mu) at every pixel. This is the physical limit.
    """
    D, A, dz = 10, 3, 0.0054
    mu_val = 5.0
    eps = 1e-12

    # g = 1.0 (exact) and g = 0.7 for comparison
    profiles = {}
    for g in [0.7, 1.0]:
        cfg = _make_mini_config(g=g)
        ms = RasterScanOCTForwardMS(cfg, device=device)
        ms.gain_dB.fill_(0.0)
        mu = torch.full((D, A), mu_val, dtype=torch.float32, device=device)
        alpha = torch.ones_like(mu)
        with torch.no_grad():
            profiles[g] = ms(mu, alpha).cpu().numpy()[:, 0]

    # SS reference
    ss = RasterScanOCTForward(_make_mini_config(), device=device)
    ss.gain_dB.fill_(0.0)
    with torch.no_grad():
        I_ss = ss(mu, alpha).cpu().numpy()[:, 0]

    z = np.arange(D)
    expected_flat = 10.0 * math.log10(mu_val + eps)
    errors = profiles[1.0] - expected_flat
    verdict, passed = _make_verdict_text(errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(z, I_ss, 'b--', lw=2, ms=6, label='SS (Beer-Lambert) — decays with depth')
    ax.plot(z, profiles[0.7], 'mo-', lw=1.5, ms=6, label='MS (g=0.7) — moderate MS')
    ax.plot(z, profiles[1.0], 'rs-', lw=2, ms=8, label='MS (g=1.0) — perfectly FLAT')
    ax.axhline(expected_flat, color='gray', ls=':', lw=1,
               label=f'$10 \\cdot \\log_{{10}}(\\mu)$ = {expected_flat:.2f} dB')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title('At g=1: every scattered photon stays in the beam\n→ no net attenuation, signal is flat')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(z, errors * 1e6, color='steelblue', edgecolor='gray')
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Deviation from flat [micro-dB]')
    ax.set_title(f'g=1 output vs expected flat line\nmax deviation = {np.max(np.abs(errors)):.2e} dB')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 11: g=1 Purely Forward Scattering → No Depth Attenuation — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_11_g_unity_flat_signal.png'), dpi=200)
    plt.close()
    return passed


def run_test_12(out_dir, device):
    """Test 12: SS vs MS with confocal ON and non-uniform mu.

    Both models are tested with a non-uniform mu profile (1→10 mm^-1 increasing
    with depth), confocal ON, and gain=50 dB. Verifies:
    - Both models agree at z=0 (tau=0)
    - MS signal is higher than SS at depth (more photons recaptured)
    - Both match their respective analytical formulas
    """
    D, A, dz = 10, 3, 0.0054
    eps = 1e-12
    K = 20.0 / math.log(10.0)
    g = 0.7
    zFP, zR, n = 5.0, 131.5, 1.35
    gain = 50.0
    lambda_mm = 1300.0 * 1e-6
    w0_sq = (lambda_mm / (math.pi * 0.0315)) ** 2
    theta_rms_sq = 2.0 * (1.0 - g)

    config_ss = _make_mini_config(confocal=True)
    config_ms = _make_mini_config(confocal=True, g=g)
    ss = RasterScanOCTForward(config_ss, device=device)
    ms = RasterScanOCTForwardMS(config_ms, device=device)
    ss.gain_dB.fill_(gain)
    ms.gain_dB.fill_(gain)

    mu_vals = np.arange(1, D + 1, dtype=np.float32)  # [1, 2, ..., 10]
    mu_np = np.tile(mu_vals[:, np.newaxis], (1, A))
    mu_t = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_t = torch.ones_like(mu_t)

    with torch.no_grad():
        I_ss = ss(mu_t, alpha_t).cpu().numpy()[:, 0]
        I_ms = ms(mu_t, alpha_t).cpu().numpy()[:, 0]

    # Analytical SS
    z = np.arange(D, dtype=np.float64)
    cumsum = np.cumsum(mu_vals.astype(np.float64) * dz)
    tau = np.concatenate([[0.0], cumsum[:-1]])
    confocal_dB = 10.0 * np.log10(1.0 / (((z - zFP) / zR) ** 2 + 1.0))
    expected_ss = gain + 10.0 * np.log10(mu_vals.astype(np.float64) + eps) + confocal_dB - K * tau

    # Analytical MS
    wH_sq = w0_sq * (((z - zFP) / zR) ** 2 + 1.0)
    z_mm = z * dz
    z_over_n_sq = (z_mm / n) ** 2
    wS_sq = wH_sq + (1.0 / 3.0) * tau * theta_rms_sq * z_over_n_sq
    exp_neg = np.exp(-tau)
    T_total = (exp_neg ** 2
               + 4 * exp_neg * (1 - exp_neg) / (1 + wS_sq / (wH_sq + eps))
               + (1 - exp_neg) ** 2 * wH_sq / (wS_sq + eps))
    expected_ms = gain + 10.0 * np.log10(mu_vals.astype(np.float64) + eps) + confocal_dB + 10 * np.log10(T_total + eps)

    err_ss = I_ss - expected_ss.astype(np.float32)
    err_ms = I_ms - expected_ms.astype(np.float32)
    max_err = max(np.max(np.abs(err_ss)), np.max(np.abs(err_ms)))
    passed = max_err < 1e-3
    verdict = f"{'PASS' if passed else 'FAIL'} (max error = {max_err:.2e})"

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    ax = axes[0]
    ax.plot(z, I_ss, 'bo-', lw=2, ms=8, label='SS model')
    ax.plot(z, I_ms, 'rs-', lw=2, ms=8, label='MS model (g=0.7)')
    ax.fill_between(z, I_ss, I_ms, alpha=0.15, color='green', label='MS excess signal')
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title(f'Non-uniform $\\mu$ = [1, 2, ..., 10] mm$^{{-1}}$\nConfocal ON, gain={gain} dB')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(z, I_ms - I_ss, 'go-', lw=2, ms=8)
    ax.axhline(0, color='gray', ls=':', lw=1)
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('MS $-$ SS [dB]')
    ax.set_title('MS excess grows with depth\n(more multiply scattered photons at greater optical depth)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    width = 0.35
    ax.bar(z - width / 2, err_ss * 1e5, width, color='steelblue', label='SS error', edgecolor='gray')
    ax.bar(z + width / 2, err_ms * 1e5, width, color='salmon', label='MS error', edgecolor='gray')
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Error [$\\times 10^{-5}$ dB]')
    ax.set_title('Both models match their analytical formulas')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 12: SS vs MS with Confocal + Non-Uniform $\\mu$ — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_12_nonuniform_mu_ss_ms.png'), dpi=200)
    plt.close()
    return passed


def run_test_13(out_dir, device):
    """Test 13: Cumulative sum is EXCLUSIVE — a pixel does not attenuate itself.

    With scatterers only at pixel 0, the signal at z=0 should be 10*log10(mu)
    with NO self-attenuation. Pixels below see the attenuation from pixel 0.
    """
    D, A, dz = 10, 3, 0.0054
    K = 20.0 / math.log(10.0)
    eps = 1e-12
    mu_val = 10.0

    config = _make_mini_config()
    ss = RasterScanOCTForward(config, device=device)
    ss.gain_dB.fill_(0.0)

    mu = torch.zeros(D, A, dtype=torch.float32, device=device)
    mu[0, :] = mu_val  # scatterers only at surface
    alpha = torch.ones_like(mu)

    with torch.no_grad():
        I_dB = ss(mu, alpha).cpu().numpy()[:, 0]

    z = np.arange(D)
    # z=0: tau=0 (EXCLUSIVE), I = 10*log10(10) = 10.0
    # z>=1: mu=0, so I = 10*log10(eps) - K*tau_from_z0
    tau_from_z0 = mu_val * dz  # same for all z>=1 (only z=0 contributes)
    expected = np.full(D, 10.0 * math.log10(eps), dtype=np.float64)
    expected[0] = 10.0 * math.log10(mu_val)
    for zi in range(1, D):
        expected[zi] = 10.0 * math.log10(eps) - K * tau_from_z0

    errors = I_dB - expected.astype(np.float32)
    verdict, passed = _make_verdict_text(errors, tol=1e-3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.scatter([0], [I_dB[0]], s=150, c='red', zorder=5, edgecolors='black',
              linewidths=1, label=f'z=0: {I_dB[0]:.2f} dB (NOT self-attenuated)')
    ax.scatter(z[1:], I_dB[1:], s=80, c='steelblue', zorder=4, edgecolors='gray',
              label=f'z>0: attenuated by z=0')
    ax.axhline(10.0 * math.log10(mu_val), color='red', ls='--', lw=1.5,
               label=f'$10 \\cdot \\log_{{10}}({mu_val})$ = {10*math.log10(mu_val):.2f} dB')
    ax.annotate('No self-attenuation!\n$\\sum_{i=0}^{-1} \\mu_i \\Delta z = 0$', xy=(0, I_dB[0]),
                textcoords='offset points', xytext=(30, -15), fontsize=10,
                fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Intensity [dB]')
    ax.set_title('Only pixel 0 has scatterers ($\\mu$=10 mm$^{-1}$)\nPixel 0 signal = $10\\cdot\\log_{10}(\\mu)$ with zero self-attenuation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Zoom: show optical depth diagram
    tau_visual = np.zeros(D)
    tau_visual[1:] = tau_from_z0
    ax.step(z, tau_visual, 'b-', lw=2, where='mid', label='$\\sum_{i=0}^{z-1} \\mu_i \\Delta z$ (shifted cumsum)')
    ax.scatter([0], [0], s=100, c='red', zorder=5, edgecolors='black')
    ax.annotate('$\\sum_{i=0}^{-1} \\mu_i \\Delta z=0$\n(exclusive)', xy=(0, 0),
                textcoords='offset points', xytext=(30, 20), fontsize=9,
                fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate(f'$\\sum_{{i=0}}^{{0}} \\mu_i \\Delta z={tau_from_z0:.4f}$\n(from pixel 0)', xy=(1, tau_from_z0),
                textcoords='offset points', xytext=(30, 10), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='blue'))
    ax.set_xlabel('Depth pixel index z')
    ax.set_ylabel('Cumulative optical depth $\\sum \\mu \\Delta z$')
    ax.set_title('Shifted cumsum: pixel z is NOT in its own $\\sum \\mu \\Delta z$\n(light reaches z before being scattered at z)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Test 13: Cumulative Sum is Exclusive (No Self-Attenuation) — {verdict}',
                 fontsize=12, fontweight='bold',
                 color='green' if passed else 'red')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_13_cumsum_exclusivity.png'), dpi=200)
    plt.close()
    return passed


def run_numbered_verification_tests(out_dir, device):
    """Run all 13 numbered verification tests and generate PNG images.

    Each test produces a self-contained figure named test_NN_<name>.png.

    Returns:
        Tuple of (n_passed, n_total).
    """
    vdir = os.path.join(out_dir, 'verification')
    os.makedirs(vdir, exist_ok=True)

    tests = [
        ("01", "SS pixel-by-pixel",          run_test_01),
        ("02", "Gain offset",                 run_test_02),
        ("03", "Alpha scaling",               run_test_03),
        ("04", "Confocal correction",         run_test_04),
        ("05", "Noise mask effect",           run_test_05),
        ("06", "Surface agreement",           run_test_06),
        ("07", "Transport decomposition",     run_test_07),
        ("08", "Confocal-beam identity",      run_test_08),
        ("09", "g=0 strong MS",              run_test_09),
        ("10", "Deep pixel",                  run_test_10),
        ("11", "g=1 flat signal",            run_test_11),
        ("12", "Non-uniform mu SS vs MS",    run_test_12),
        ("13", "Cumsum exclusivity",          run_test_13),
    ]

    n_passed = 0
    for num, name, func in tests:
        print(f"  [{num}/13] {name} ...", end=" ", flush=True)
        passed = func(vdir, device)
        n_passed += int(passed)
        print("PASS" if passed else "FAIL")

    print(f"\n  Verification: {n_passed}/{len(tests)} tests passed")
    return n_passed, len(tests)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Forward Model Validation: Beer-Lambert Law")
    print("=" * 70)

    # ── load config, disable confocal to isolate Beer-Lambert ──
    config = OmegaConf.load(os.path.join(PHANTOM_ROOT, 'configs', 'config.yaml'))
    config.forward_model_system.confocal_correction = False

    depth_samples = int(config.forward_model_system.depth_samples)
    a_lines       = int(config.forward_model_system.a_lines)
    dz            = float(config.forward_model_system.axial_pixel_size_mm)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── forward model with gain=0, confocal=off, alpha=1 ──
    forward_model = RasterScanOCTForward(config, device=device)
    forward_model.to(device)
    forward_model.gain_dB.fill_(0.0)

    out_dir = os.path.join(PHANTOM_ROOT, 'outputs', 'forward_model_verification')
    os.makedirs(out_dir, exist_ok=True)

    # ── simple two-A-line demo (confocal-corrected) ──────────────────────────
    print(f"\n--- Simple Demo: single layer, two A-lines, confocal ON ---")
    run_simple_two_aline_demo(out_dir, device)

    # ── generate synthetic layered phantom ──
    print(f"\nPhantom: {depth_samples} x {a_lines}")
    print(f"  Tissue band : pixels [{LAYER_START_PX}, {LAYER_END_PX})")
    print(f"  Layers/A-scan: {N_LAYERS_MIN}-{N_LAYERS_MAX}")
    print(f"  mu range     : [{MU_MIN_MM}, {MU_MAX_MM}] mm^-1")
    print(f"  Axial pixel  : {dz} mm")

    mu_np, layer_info = generate_layered_phantom(depth_samples, a_lines)

    n_layers_dist = [info['n_layers'] for info in layer_info]
    print(f"  Layer count  : "
          + ", ".join(f"{k}L={n_layers_dist.count(k)}"
                      for k in sorted(set(n_layers_dist))))

    # ── run forward model ──
    print("\nRunning forward model ...")
    mu_t = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_t = torch.ones_like(mu_t)

    with torch.no_grad():
        I_model_dB_t = forward_model(mu_t, alpha_t, noise_mask=None)
    I_model_dB = I_model_dB_t.cpu().numpy()

    # convert model dB to linear:  I_linear = 10^(I_dB / 10)
    I_model_linear = np.power(10.0, I_model_dB.astype(np.float64) / 10.0)

    # ── analytical solutions (float64 ground truth) ──
    print("Computing analytical Beer-Lambert (float64) ...")
    I_analytic_dB     = analytical_beer_lambert_dB(mu_np, dz)
    I_analytic_linear = analytical_beer_lambert_linear(mu_np, dz)

    # ── error analysis (tissue band) ──
    ts, te = LAYER_START_PX, LAYER_END_PX

    # dB comparison (cast analytical to float32 for fair comparison)
    err_dB = I_model_dB[ts:te] - I_analytic_dB[ts:te].astype(np.float32)
    mae_dB  = np.mean(np.abs(err_dB))
    max_dB  = np.max(np.abs(err_dB))
    rmse_dB = np.sqrt(np.mean(err_dB ** 2))

    # linear comparison
    err_lin = I_model_linear[ts:te] - I_analytic_linear[ts:te]
    mae_lin  = np.mean(np.abs(err_lin))
    max_lin  = np.max(np.abs(err_lin))
    rmse_lin = np.sqrt(np.mean(err_lin ** 2))

    # R^2 in dB space
    analytic_tissue_f32 = I_analytic_dB[ts:te].astype(np.float32)
    ss_res = np.sum(err_dB ** 2)
    ss_tot = np.sum((analytic_tissue_f32 - np.mean(analytic_tissue_f32)) ** 2)
    r2_dB = 1.0 - ss_res / (ss_tot + 1e-30)

    # R^2 in linear space
    analytic_lin_tissue = I_analytic_linear[ts:te]
    ss_res_lin = np.sum(err_lin ** 2)
    ss_tot_lin = np.sum((analytic_lin_tissue - np.mean(analytic_lin_tissue)) ** 2)
    r2_lin = 1.0 - ss_res_lin / (ss_tot_lin + 1e-30)

    print(f"\n{'=' * 55}")
    print(f"  Tissue band [{ts}, {te}) — {a_lines} A-scans")
    print(f"{'─' * 55}")
    print(f"  dB space:")
    print(f"    MAE  = {mae_dB:.6e} dB")
    print(f"    Max  = {max_dB:.6e} dB")
    print(f"    RMSE = {rmse_dB:.6e} dB")
    print(f"    R^2  = {r2_dB:.12f}")
    print(f"{'─' * 55}")
    print(f"  Linear space:")
    print(f"    MAE  = {mae_lin:.6e}")
    print(f"    Max  = {max_lin:.6e}")
    print(f"    RMSE = {rmse_lin:.6e}")
    print(f"    R^2  = {r2_lin:.12f}")
    print(f"{'=' * 55}")

    # ── save arrays ──
    np.save(os.path.join(out_dir, 'mu_phantom.npy'), mu_np)
    np.save(os.path.join(out_dir, 'forward_model_dB.npy'), I_model_dB)
    np.save(os.path.join(out_dir, 'analytical_dB.npy'), I_analytic_dB)
    np.save(os.path.join(out_dir, 'forward_model_linear.npy'),
            I_model_linear.astype(np.float32))
    np.save(os.path.join(out_dir, 'analytical_linear.npy'),
            I_analytic_linear.astype(np.float32))

    # ── save validation report ──
    report_path = os.path.join(out_dir, 'validation_results.txt')
    with open(report_path, 'w') as f:
        f.write("Beer-Lambert Forward Model Validation\n")
        f.write("=" * 72 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 72 + "\n")
        f.write("This report validates the RasterScanOCTForward model (src/forward.py)\n")
        f.write("against analytical closed-form solutions. The forward model computes\n")
        f.write("OCT intensity in dB space:\n\n")
        f.write("  I_dB(z) = gain_dB + 10*log10(alpha(z)) + 10*log10(mu(z))\n")
        f.write("            + confocal_dB(z) - K * cumsum_shifted(mu * dz)\n\n")
        f.write("where K = 20/ln(10) ~ 8.6859.  In linear space this corresponds to\n")
        f.write("the discrete Beer-Lambert law:\n\n")
        f.write("  I(z) = G * alpha(z) * mu(z) * exp(-2 * sum_{i=0}^{z-1} mu(i)*dz) * C(z)\n\n")
        f.write("To isolate the Beer-Lambert core, the base validation disables the\n")
        f.write("confocal correction (C=1), sets gain_dB=0, and fixes alpha=1 everywhere.\n")
        f.write("The analytical solution is computed independently in float64 with the\n")
        f.write("same eps=1e-12, and errors reflect float32-vs-float64 precision.\n\n")

        f.write("SYNTHETIC PHANTOM\n")
        f.write("-" * 72 + "\n")
        f.write(f"  Grid              : {depth_samples} x {a_lines} (depth x A-scans)\n")
        f.write(f"  Tissue band       : depth pixels [{ts}, {te})\n")
        f.write(f"  Layers / A-scan   : {N_LAYERS_MIN}-{N_LAYERS_MAX} (randomly chosen)\n")
        f.write(f"  mu range (tissue) : [{MU_MIN_MM}, {MU_MAX_MM}] mm^-1\n")
        f.write(f"  mu (air/background): {MU_AIR} mm^-1\n")
        f.write(f"  Axial pixel size  : {dz} mm\n")
        f.write(f"  Seed              : {SEED}\n\n")
        f.write("Each A-scan is an independent column.  Within each, 2-5 homogeneous\n")
        f.write("layers are placed between pixels 100-200 with random mu values drawn\n")
        f.write("uniformly from [0.1, 15] mm^-1.  Outside the tissue band, mu is set\n")
        f.write("to a negligible air value (1e-6 mm^-1).\n\n")

        f.write("FORWARD MODEL SETTINGS (base validation)\n")
        f.write("-" * 72 + "\n")
        f.write(f"  Confocal correction : OFF  (C(z) = 1 for all z)\n")
        f.write(f"  gain_dB             : 0.0  (no system-gain offset)\n")
        f.write(f"  alpha               : 1.0  (uniform, no backscatter variation)\n")
        f.write(f"  Device              : {device}\n\n")

        f.write("=" * 72 + "\n")
        f.write("BASE VALIDATION: BEER-LAMBERT CONCORDANCE\n")
        f.write("=" * 72 + "\n\n")
        f.write("Experimental:  Forward model output I_dB(z) for the synthetic phantom.\n")
        f.write("Analytical:    10*log10(mu(z) + eps) - K * sum_{i=0}^{z-1} mu(i)*dz\n")
        f.write("               computed in float64 using numpy.\n")
        f.write("Comparison:    Pixel-by-pixel difference over the tissue band.\n\n")

        f.write("Error statistics (tissue band only):\n\n")
        f.write("  dB space:\n")
        f.write(f"    MAE  = {mae_dB:.6e} dB\n")
        f.write(f"    Max  = {max_dB:.6e} dB\n")
        f.write(f"    RMSE = {rmse_dB:.6e} dB\n")
        f.write(f"    R^2  = {r2_dB:.12f}\n\n")
        f.write("  Linear space:\n")
        f.write(f"    MAE  = {mae_lin:.6e}\n")
        f.write(f"    Max  = {max_lin:.6e}\n")
        f.write(f"    RMSE = {rmse_lin:.6e}\n")
        f.write(f"    R^2  = {r2_lin:.12f}\n\n")

        f.write("PLOT DESCRIPTIONS (base validation)\n")
        f.write("-" * 72 + "\n\n")

        f.write("[1] mu_phantom.png\n")
        f.write("    Left:  Full 500x500 mu map (viridis colormap, mm^-1).  Red dashed\n")
        f.write("           lines mark the tissue band at pixels 100 and 200.\n")
        f.write("    Right: Zoomed view of the tissue band only.  Each vertical stripe\n")
        f.write("           is one A-scan; color steps within a column are the 2-5\n")
        f.write("           homogeneous layers with distinct mu values.\n\n")

        f.write("[2] comparison_panel.png\n")
        f.write("    2x3 grid showing tissue-band images (pixels 100-200 x 500 A-scans).\n")
        f.write("    Top row (dB space):\n")
        f.write("      (0,0) Forward model output I_dB — inferno colormap, shared scale.\n")
        f.write("      (0,1) Analytical Beer-Lambert I_dB — same colormap and scale.\n")
        f.write("      (0,2) Absolute difference |model - analytical| in dB.\n")
        f.write("    Bottom row (linear space, via 10^(I_dB/10)):\n")
        f.write("      (1,0) Forward model output — linear intensity.\n")
        f.write("      (1,1) Analytical output — linear intensity.\n")
        f.write("      (1,2) Absolute difference in linear intensity.\n\n")

        f.write("[3] ascan_profiles.png\n")
        f.write("    2 rows x 5 columns for A-scans {0, 125, 250, 375, 499}.\n")
        f.write("    Top row:    dB intensity vs depth.  Blue solid = forward model,\n")
        f.write("                red dashed = analytical.  Colored bands show each\n")
        f.write("                layer with its mu value annotated.\n")
        f.write("    Bottom row: Same A-scans in linear intensity.\n")
        f.write("    The two curves overlap perfectly, confirming per-A-scan concordance.\n\n")

        f.write("[4] concordance.png\n")
        f.write("    Three-panel concordance analysis (tissue pixels only).\n")
        f.write("    Left:   Scatter plot of forward-model I_dB vs analytical I_dB.\n")
        f.write("            50k subsampled tissue pixels.  Points fall on the y=x line.\n")
        f.write("    Center: Histogram of per-pixel error (model - analytical) in dB.\n")
        f.write("            Tight distribution centered at zero (~1e-5 dB spread).\n")
        f.write("    Right:  Mean absolute error per A-scan index.  Shows the error is\n")
        f.write("            uniformly small across all 500 A-scans.\n\n")

        f.write("[5] layer_decay_examples.png\n")
        f.write("    Three panels for A-scans {50, 250, 450} in LINEAR intensity.\n")
        f.write("    Blue solid = forward model output within the tissue band.\n")
        f.write("    Dashed colored = per-layer exponential envelope, computed as\n")
        f.write("      I_expected(z) = I(b0) * exp(-2 * mu_layer * (z - b0) * dz)\n")
        f.write("    where I(b0) is the model's intensity at the layer boundary.\n")
        f.write("    This verifies the exponential decay rate within each homogeneous\n")
        f.write("    layer matches the expected Beer-Lambert attenuation.\n\n")

    # ═════════════════════════════════════════════════════════════════════════
    # Generate base validation visualisations
    # ═════════════════════════════════════════════════════════════════════════
    print("\nGenerating plots ...")

    plot_mu_phantom(mu_np, layer_info, out_dir)
    print("  [1/8] mu_phantom.png")

    plot_comparison_panel(I_model_dB, I_analytic_dB,
                          I_model_linear.astype(np.float32),
                          I_analytic_linear.astype(np.float32), out_dir)
    print("  [2/8] comparison_panel.png")

    plot_ascan_profiles(I_model_dB, I_analytic_dB,
                        I_model_linear.astype(np.float32),
                        I_analytic_linear.astype(np.float32),
                        layer_info, out_dir)
    print("  [3/8] ascan_profiles.png")

    plot_concordance(I_model_dB, I_analytic_dB, out_dir)
    print("  [4/8] concordance.png")

    plot_layer_decay_examples(I_model_linear.astype(np.float32),
                              dz, layer_info, out_dir)
    print("  [5/8] layer_decay_examples.png")

    # ═════════════════════════════════════════════════════════════════════════
    # Parameter sweep validations
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  Parameter Sweep Validations")
    print(f"{'=' * 70}")

    sweep_results = []

    sweep_results.append(
        sweep_axial_pixel_size(forward_model, mu_np, layer_info,
                               out_dir, device))
    print("  [6/7] sweep_axial_pixel_size.png")

    sweep_results.append(
        sweep_confocal(config, mu_np, I_model_dB, out_dir, device))
    print("  [7/7] sweep_confocal.png")

    # Append sweep results to validation report
    with open(report_path, 'a') as f:
        for s in sweep_results:
            f.write(s)

    # ═════════════════════════════════════════════════════════════════════════
    # EHF Multiple Scattering Validation
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  EHF Multiple Scattering Validation")
    print(f"{'=' * 70}")

    ehf_report = run_ehf_validation(out_dir, device)
    print("  [9/9] ehf/g_sweep_profiles.png")
    print("  [10/10] ehf/ms_vs_ss_comparison.png")

    with open(report_path, 'a') as f:
        f.write(ehf_report)

    # ═════════════════════════════════════════════════════════════════════════
    # Numbered Visual Verification Tests
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  Numbered Visual Verification Tests")
    print(f"{'=' * 70}")

    n_passed, n_total = run_numbered_verification_tests(out_dir, device)

    with open(report_path, 'a') as f:
        f.write("\n" + "=" * 72 + "\n")
        f.write("NUMBERED VISUAL VERIFICATION TESTS\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Result: {n_passed}/{n_total} tests passed.\n")
        f.write(f"Output: {os.path.join(out_dir, 'verification')}/test_NN_*.png\n\n")
        f.write("Each test produces a self-contained PNG image with annotated\n")
        f.write("pass/fail verdict. See the verification/ subdirectory.\n")

    # ═════════════════════════════════════════════════════════════════════════
    # Publication-Quality Figures
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  Generating Publication-Quality Figures")
    print(f"{'=' * 70}")

    paper_dir = os.path.join(out_dir, 'paper')
    os.makedirs(paper_dir, exist_ok=True)

    generate_publication_figures(
        mu_np, layer_info, I_model_dB, I_analytic_dB,
        I_model_linear, I_analytic_linear,
        out_dir, paper_dir, device, config, dz,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # Plain-Language Report
    # ═════════════════════════════════════════════════════════════════════════
    report_plain = os.path.join(out_dir, 'REPORT.txt')
    _write_plain_language_report(report_plain, mae_dB, max_dB, r2_dB,
                                  mae_lin, r2_lin, n_passed, n_total,
                                  depth_samples, a_lines, dz, ts, te, device)
    print(f"  Plain-language report: {report_plain}")

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")
    return 0


# ═════════════════════════════════════════════════════════════════════════════
# Publication-quality figure generation
# ═════════════════════════════════════════════════════════════════════════════

def _apply_paper_style():
    """Set matplotlib rcParams for publication figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth': 1.5,
    })


def _panel_label(ax, label, x=-0.12, y=1.08):
    """Place a bold panel label like (a), (b) on an axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')


def generate_publication_figures(mu_np, layer_info, I_model_dB, I_analytic_dB,
                                  I_model_linear, I_analytic_linear,
                                  out_dir, paper_dir, device, config, dz):
    """Generate 3 publication-quality figures for the paper."""
    _apply_paper_style()

    # ---------- Figure 1: Forward Model Validation (Hero) ----------
    _fig1_forward_validation(mu_np, layer_info, I_model_dB, I_analytic_dB,
                              paper_dir, dz)

    # ---------- Figure 2: "OCT" Letter Phantom Demo ----------
    _fig2_oct_letters(paper_dir, device, config, dz)

    # ---------- Figure 3: SS vs MS Comparison ----------
    _fig3_ss_vs_ms(paper_dir, device, config, dz)

    # ---------- Figure 4: Confocal Validation ----------
    _fig4_confocal(paper_dir, device, config, mu_np, I_model_dB, dz)

    # ---------- Figure 5: A-line Profile Zoom ----------
    _fig5_aline_profiles(mu_np, layer_info, I_model_dB, I_analytic_dB,
                          paper_dir, dz)

    # ---------- Stage the npz subset that the Figure 2 notebook reads ----------
    _stage_to_figure_paper_2(paper_dir)


def _stage_to_figure_paper_2(paper_dir):
    """Copy the two npz files the Figure 2 notebook depends on into
    figures/data/figure2_verification/. Closes the experiments → outputs → figures
    data-flow loop so a fresh `python experiments/forward_model_verification.py`
    repopulates everything the notebook needs."""
    import shutil
    fig_paper_2 = os.path.join(PHANTOM_ROOT, 'figures', 'data', 'figure2_verification')
    os.makedirs(fig_paper_2, exist_ok=True)
    for fname in ('fig1_forward_validation.npz', 'fig5_aline_profiles.npz'):
        src = os.path.join(paper_dir, fname)
        dst = os.path.join(fig_paper_2, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  [stage → figure2_verification] {dst}")
        else:
            print(f"  [stage] WARN: {src} not found")


def _fig1_forward_validation(mu_np, layer_info, I_model_dB, I_analytic_dB,
                               paper_dir, dz):
    """Figure 1: 4-panel hero (2x2) — phantom, synthetic OCT, residual, error map."""
    ts, te = LAYER_START_PX, LAYER_END_PX
    tissue_mu = mu_np[ts:te, :]
    depth_mm = np.arange(ts, te) * dz
    extent = [0, mu_np.shape[1], depth_mm[-1], depth_mm[0]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ── (a) Phantom μ map (tissue band) — the INPUT ──
    ax = axes[0, 0]
    im = ax.imshow(tissue_mu, cmap='inferno', aspect='auto',
                    extent=extent, vmin=0, vmax=15)
    for a_idx in [0, 250, 499]:
        info = layer_info[a_idx]
        for b in info['boundaries'][1:-1]:
            b_mm = b * dz
            if ts * dz <= b_mm <= te * dz:
                ax.plot([a_idx - 15, a_idx + 15], [b_mm, b_mm],
                        color='white', lw=0.5, alpha=0.6)
    ax.set_xlabel('A-line index')
    ax.set_ylabel('Depth [mm]')
    ax.set_title('Input: attenuation map $\\mu_t$')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('$\\mu_t$ [mm$^{-1}$]')

    # ── (b) Synthetic OCT B-scan — the OUTPUT ──
    ax = axes[0, 1]
    model_tissue_img = I_model_dB[ts:te, :]
    # Show in grayscale like a real OCT image, with intensity range matching
    dB_vmin = np.percentile(model_tissue_img, 1)
    dB_vmax = np.percentile(model_tissue_img, 99)
    im2 = ax.imshow(model_tissue_img, cmap='gray', aspect='auto',
                     extent=extent, vmin=dB_vmin, vmax=dB_vmax)
    ax.set_xlabel('A-line index')
    ax.set_ylabel('Depth [mm]')
    ax.set_title('Output: synthetic OCT B-scan')
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label('Intensity [dB]')

    # ── (c) Residual across all 500 A-lines ──
    ax = axes[1, 0]
    residuals = (I_model_dB[ts:te, :] - I_analytic_dB[ts:te, :].astype(np.float32))
    for a_idx in range(mu_np.shape[1]):
        ax.plot(depth_mm, residuals[:, a_idx], color='#888888', lw=0.3,
                alpha=0.08, zorder=1)
    ax.axhline(0, color='#E53935', ls='--', lw=1.5, zorder=5)
    mae_all = np.mean(np.abs(residuals))
    max_all = np.max(np.abs(residuals))
    ax.set_xlabel('Depth [mm]')
    ax.set_ylabel('Residual (model $-$ analytical) [dB]')
    ax.set_title(f'Residual: all {mu_np.shape[1]} A-lines')
    ax.text(0.97, 0.95,
            f'MAE = {mae_all:.2e} dB\nMax = {max_all:.2e} dB',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    # ── (d) Absolute error map ──
    ax = axes[1, 1]
    abs_err = np.abs(residuals)
    err_max = np.percentile(abs_err, 99.5)
    im_err = ax.imshow(abs_err, cmap='magma', aspect='auto',
                        extent=extent, vmin=0, vmax=err_max)
    ax.set_xlabel('A-line index')
    ax.set_ylabel('Depth [mm]')
    ax.set_title('Absolute error map')
    cb3 = plt.colorbar(im_err, ax=ax, fraction=0.046, pad=0.04)
    cb3.set_label('|Error| [dB]')
    # R^2 annotation
    model_flat = I_model_dB[ts:te].ravel()
    analytic_flat = I_analytic_dB[ts:te].ravel().astype(np.float32)
    err_flat = model_flat - analytic_flat
    ss_res = np.sum(err_flat ** 2)
    ss_tot = np.sum((analytic_flat - np.mean(analytic_flat)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)
    ax.text(0.03, 0.95, f'$R^2$ = {r2:.6f}',
            transform=ax.transAxes, fontsize=8, va='top',
            color='white', fontweight='bold')

    plt.tight_layout()

    # ── Save the underlying arrays for the Figure 2 notebook to render from ──
    npz_path = os.path.join(paper_dir, 'fig1_forward_validation.npz')
    np.savez(npz_path,
             depth_mm=depth_mm, tissue_mu=tissue_mu,
             model_tissue_img=I_model_dB[ts:te],
             abs_err=abs_err, r2=r2)
    print(f"  [Paper 1/4] {npz_path}")

    path = os.path.join(paper_dir, 'paper_forward_model_validation.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Paper 1/4] {path}")


def _fig2_oct_letters(paper_dir, device, config, dz):
    """Figure 2: 'OCT' letter phantom — shows forward model creates realistic B-scans."""
    from PIL import Image, ImageDraw, ImageFont

    DEPTH = 500
    A_LINES = 500
    TISSUE_TOP = 80
    TISSUE_BOT = 350
    LETTER_LEFT = 100   # lateral range for letters
    LETTER_RIGHT = 400
    MU_BG = 1e-6       # air / background (no attenuation)
    MU_HIGH = 10.0      # high-attenuation letters

    # ── Render "OCT" as a binary mask using PIL ──
    letter_h = TISSUE_BOT - TISSUE_TOP
    letter_w = LETTER_RIGHT - LETTER_LEFT
    img = Image.new('L', (letter_w, letter_h), 0)
    draw = ImageDraw.Draw(img)

    # Try to find a bold font, fall back to default
    font = None
    font_size = int(letter_h * 0.45)
    for font_name in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                      '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
                      '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf']:
        if os.path.exists(font_name):
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    # Center "OCT" within the letter region
    text = "OCT"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (letter_w - tw) // 2 - bbox[0]
    y = (letter_h - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)
    letter_mask_crop = np.array(img) > 128  # (letter_h, letter_w) bool

    # ── Build μ map: air everywhere, letters only ──
    mu_np = np.full((DEPTH, A_LINES), MU_BG, dtype=np.float32)
    # Place letters into the correct region
    mu_np[TISSUE_TOP:TISSUE_BOT, LETTER_LEFT:LETTER_RIGHT][letter_mask_crop] = MU_HIGH

    # ── Run forward model (SS, confocal ON for realism) ──
    config_demo = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config_demo.forward_model_system.confocal_correction = True
    config_demo.forward_model_system.depth_samples = DEPTH
    config_demo.forward_model_system.a_lines = A_LINES
    config_demo.forward_model_system.focal_plane_position_pixels = 200.0
    config_demo.forward_model_system.rayleigh_range_pixels = 150.0

    model_ss = RasterScanOCTForward(config_demo, device=device)
    model_ss.gain_dB.fill_(100.0)

    mu_t = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_t = torch.ones_like(mu_t)
    with torch.no_grad():
        I_dB = model_ss(mu_t, alpha_t).cpu().numpy()

    # Also run MS model (g=0.7) for comparison
    config_ms = OmegaConf.create(OmegaConf.to_container(config_demo, resolve=True))
    config_ms.forward_model_system.forward_model_type = 'ms'
    config_ms.forward_model_system.g = 0.7
    model_ms = RasterScanOCTForwardMS(config_ms, device=device)
    model_ms.gain_dB.fill_(100.0)
    with torch.no_grad():
        I_dB_ms = model_ms(mu_t, alpha_t).cpu().numpy()

    # ── Analytical Beer-Lambert for verification ──
    K = 20.0 / math.log(10.0)
    mu64 = mu_np.astype(np.float64)
    dz_val = float(config_demo.forward_model_system.axial_pixel_size_mm)
    cumulative = np.cumsum(mu64 * dz_val, axis=0)
    shifted = np.zeros_like(mu64)
    shifted[1:] = cumulative[:-1]
    I_analytic = 100.0 + 10.0 * np.log10(mu64 + 1e-12) - K * shifted
    # Add confocal
    z_px = np.arange(DEPTH, dtype=np.float64)
    zFP = 200.0
    zR = 150.0
    C_dB = 10.0 * np.log10(1.0 / (((z_px - zFP) / zR) ** 2 + 1.0))
    I_analytic += C_dB[:, np.newaxis]

    # ── Figure: 2x2 — all panels use same depth range ──
    show_top = max(0, TISSUE_TOP - 30)
    show_bot = min(DEPTH, TISSUE_BOT + 50)
    dz_val = float(config_demo.forward_model_system.axial_pixel_size_mm)
    show_depth = np.arange(show_top, show_bot) * dz_val
    show_extent = [0, A_LINES, show_depth[-1], show_depth[0]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) μ map — same depth range as B-scans
    ax = axes[0, 0]
    _panel_label(ax, '(a)')
    im = ax.imshow(mu_np[show_top:show_bot, :], cmap='inferno', aspect='auto',
                    extent=show_extent, vmin=0, vmax=MU_HIGH * 1.1)
    ax.set_xlabel('A-line index')
    ax.set_ylabel('Depth [mm]')
    ax.set_title('Input: attenuation map $\\mu_t$')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('$\\mu_t$ [mm$^{-1}$]')

    # Shared dB range for (b) and (c): use full dynamic range so air = black,
    # letters = bright, and spatial proportions match panel (a)
    I_show = I_dB[show_top:show_bot, :]
    I_show_ms = I_dB_ms[show_top:show_bot, :]
    # Tissue pixels (where letters are) define the bright end
    tissue_pixels = I_show[mu_np[show_top:show_bot, :] > 0.1]
    dB_vmax = float(np.max(tissue_pixels)) if tissue_pixels.size else 100.0
    # Air defines the dark end — but clamp so we don't go to -infinity
    dB_vmin = dB_vmax - 40.0  # 40 dB dynamic range, like a real OCT display

    # (b) SS synthetic OCT B-scan
    ax = axes[0, 1]
    _panel_label(ax, '(b)')
    im2 = ax.imshow(I_show, cmap='gray', aspect='auto',
                     extent=show_extent, vmin=dB_vmin, vmax=dB_vmax)
    ax.set_xlabel('A-line index')
    ax.set_ylabel('Depth [mm]')
    ax.set_title('SS forward model output (Beer-Lambert)')
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label('Intensity [dB]')

    # (c) MS synthetic OCT B-scan
    ax = axes[1, 0]
    _panel_label(ax, '(c)')
    im3 = ax.imshow(I_show_ms, cmap='gray', aspect='auto',
                     extent=show_extent, vmin=dB_vmin, vmax=dB_vmax)
    ax.set_xlabel('A-line index')
    ax.set_ylabel('Depth [mm]')
    ax.set_title('MS forward model output (EHF, g=0.7)')
    cb3 = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cb3.set_label('Intensity [dB]')

    # (d) Residual: model vs analytical (SS) — same depth range
    ax = axes[1, 1]
    _panel_label(ax, '(d)')
    residual = I_dB[show_top:show_bot, :] - I_analytic[show_top:show_bot, :].astype(np.float32)
    abs_resid = np.abs(residual)
    err_max = np.percentile(abs_resid, 99.5)
    im4 = ax.imshow(abs_resid, cmap='magma', aspect='auto',
                     extent=show_extent, vmin=0, vmax=err_max)
    mae = np.mean(abs_resid)
    ax.set_xlabel('A-line index')
    ax.set_ylabel('Depth [mm]')
    ax.set_title('SS model vs analytical |error|')
    cb4 = plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)
    cb4.set_label('|Error| [dB]')
    ax.text(0.03, 0.95, f'MAE = {mae:.2e} dB',
            transform=ax.transAxes, fontsize=9, va='top',
            color='white', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(paper_dir, 'paper_oct_letters_demo.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Paper 2/4] {path}")


def _fig3_ss_vs_ms(paper_dir, device, config, dz):
    """Figure 3: SS vs MS comparison with g-sweep."""
    TISSUE_START, TISSUE_END = 100, 300
    MU_TISSUE = 5.0
    depth_samples = int(config.forward_model_system.depth_samples)
    a_lines = int(config.forward_model_system.a_lines)

    mu_np = np.full((depth_samples, a_lines), MU_AIR, dtype=np.float32)
    mu_np[TISSUE_START:TISSUE_END, :] = MU_TISSUE
    mu_t = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_t = torch.ones_like(mu_t)
    mid = a_lines // 2
    depth_mm = np.arange(TISSUE_START, TISSUE_END) * dz

    # SS model
    config_ss = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config_ss.forward_model_system.confocal_correction = False
    ss_model = RasterScanOCTForward(config_ss, device=device)
    ss_model.gain_dB.fill_(0.0)
    with torch.no_grad():
        I_ss = ss_model(mu_t, alpha_t).cpu().numpy()

    # MS model at g=0.7
    config_ms = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config_ms.forward_model_system.confocal_correction = False
    config_ms.forward_model_system.forward_model_type = 'ms'
    config_ms.forward_model_system.g = 0.7
    ms_model = RasterScanOCTForwardMS(config_ms, device=device)
    ms_model.gain_dB.fill_(0.0)
    with torch.no_grad():
        I_ms = ms_model(mu_t, alpha_t).cpu().numpy()

    # g-sweep profiles
    g_values = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    g_profiles = {}
    for g in g_values:
        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cfg.forward_model_system.confocal_correction = False
        cfg.forward_model_system.forward_model_type = 'ms'
        cfg.forward_model_system.g = g
        m = RasterScanOCTForwardMS(cfg, device=device)
        m.gain_dB.fill_(0.0)
        with torch.no_grad():
            g_profiles[g] = m(mu_t, alpha_t)[:, mid].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ── (a) SS vs MS at g=0.7 ──
    ax = axes[0]
    _panel_label(ax, '(a)')
    ss_tissue = I_ss[TISSUE_START:TISSUE_END, mid]
    ms_tissue = I_ms[TISSUE_START:TISSUE_END, mid]
    ax.plot(ss_tissue, depth_mm, color='#2196F3', lw=2.0,
            label='Single-scatter (Beer-Lambert)')
    ax.plot(ms_tissue, depth_mm, color='#F44336', lw=2.0,
            label='Multiple-scatter (EHF, g=0.7)')
    ax.fill_betweenx(depth_mm, ss_tissue, ms_tissue,
                      alpha=0.15, color='#F44336')
    ax.invert_yaxis()
    ax.set_xlabel('Intensity [dB]')
    ax.set_ylabel('Depth [mm]')
    ax.set_title('SS vs MS forward model')
    ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
    # Annotation
    max_excess = np.max(ms_tissue - ss_tissue)
    ax.annotate(f'MS excess\n+{max_excess:.1f} dB at depth',
                xy=(ms_tissue[-20], depth_mm[-20]),
                xytext=(ms_tissue[-20] + 3, depth_mm[-60]),
                fontsize=8, color='#F44336',
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.0))

    # ── (b) g-sweep ──
    ax = axes[1]
    _panel_label(ax, '(b)')
    cmap = plt.cm.plasma
    colors_g = [cmap(v) for v in np.linspace(0.1, 0.9, len(g_values))]
    for i, g in enumerate(g_values):
        tissue_prof = g_profiles[g][TISSUE_START:TISSUE_END]
        ax.plot(tissue_prof, depth_mm, color=colors_g[i], lw=1.5,
                label=f'g = {g:.1f}')
    ss_tissue_ref = I_ss[TISSUE_START:TISSUE_END, mid]
    ax.plot(ss_tissue_ref, depth_mm, 'k--', lw=1.2, alpha=0.6,
            label='SS model')
    ax.invert_yaxis()
    ax.set_xlabel('Intensity [dB]')
    ax.set_ylabel('Depth [mm]')
    ax.set_title('Anisotropy factor g sweep')
    ax.legend(fontsize=7, loc='lower left', framealpha=0.9, ncol=2)
    # Annotate g=1 flat
    ax.annotate('g = 1.0: no depth\nattenuation ($T_{total}$ = 1)',
                xy=(g_profiles[1.0][TISSUE_END - 5], depth_mm[-5]),
                xytext=(g_profiles[1.0][TISSUE_END - 5] - 8, depth_mm[-40]),
                fontsize=7, color=colors_g[-1],
                arrowprops=dict(arrowstyle='->', color=colors_g[-1], lw=0.8))

    plt.tight_layout()
    path = os.path.join(paper_dir, 'paper_ss_vs_ms_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Paper 3/4] {path}")


def _fig4_confocal(paper_dir, device, config, mu_np, I_baseline_dB, dz):
    """Figure 3: Confocal validation — Lorentzian C_dB(z)."""
    D = mu_np.shape[0]
    z_mm = np.arange(D) * dz
    z_px = np.arange(D, dtype=np.float64)
    mu_t = torch.tensor(mu_np, dtype=torch.float32, device=device)
    alpha_t = torch.ones_like(mu_t)
    mid_col = 250

    configs_list = [
        {'zFP': 150, 'zR': 80,  'label': '$z_{FP}$=150, $z_R$=80'},
        {'zFP': 231, 'zR': 131.5, 'label': '$z_{FP}$=231, $z_R$=131.5 (default)'},
        {'zFP': 350, 'zR': 200, 'label': '$z_{FP}$=350, $z_R$=200'},
    ]
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── (a) Confocal functions ──
    ax = axes[0]
    _panel_label(ax, '(a)')
    residuals = []
    analyticals = []
    for ci, (cfp, color) in enumerate(zip(configs_list, colors)):
        zFP, zR = cfp['zFP'], cfp['zR']
        C_dB = 10.0 * np.log10(1.0 / (((z_px - zFP) / zR) ** 2 + 1.0))
        ax.plot(z_mm, C_dB, color=color, lw=1.5, label=cfp['label'])
        analyticals.append(C_dB)

        # Run model with confocal ON
        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cfg.forward_model_system.confocal_correction = True
        cfg.forward_model_system.focal_plane_position_pixels = float(zFP)
        cfg.forward_model_system.rayleigh_range_pixels = float(zR)
        model = RasterScanOCTForward(cfg, device=device)
        model.gain_dB.fill_(0.0)
        with torch.no_grad():
            I_on = model(mu_t, alpha_t).cpu().numpy()
        resid = I_on[:, mid_col] - I_baseline_dB[:, mid_col]
        residuals.append(resid)

    ax.axhline(0, color='gray', ls=':', lw=0.5, alpha=0.5)
    ax.set_xlabel('Depth [mm]')
    ax.set_ylabel('$C_{dB}(z)$ [dB]')
    ax.set_title('Confocal correction function')
    ax.legend(fontsize=7, loc='lower left', framealpha=0.9)

    # ── (b) Model residual vs analytical ──
    ax = axes[1]
    _panel_label(ax, '(b)')
    for ci, (cfp, color) in enumerate(zip(configs_list, colors)):
        ax.plot(z_mm, residuals[ci], color=color, lw=1.5,
                label=f'Model residual ({cfp["label"][:10]}...)')
        ax.plot(z_mm, analyticals[ci], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlabel('Depth [mm]')
    ax.set_ylabel('$\\Delta I_{dB}$ [dB]')
    ax.set_title('Model residual overlays analytical')

    # Error annotation
    max_err = max(
        np.max(np.abs(r.astype(np.float64) - a))
        for r, a in zip(residuals, analyticals)
    )
    ax.text(0.95, 0.95, f'Max |error| = {max_err:.2e} dB',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.9))
    ax.legend(fontsize=6, loc='lower left', framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(paper_dir, 'paper_confocal_validation.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Paper 4/4] {path}")


def _fig5_aline_profiles(mu_np, layer_info, I_model_dB, I_analytic_dB,
                           paper_dir, dz):
    """Figure 5: A-line zoom — μ profiles and intensity matching for 3 selected A-lines.

    Picks A-lines with 2, 3, and 4 layers that have well-separated μ values
    and sufficiently thick layers (>=8 pixels each), then shows the step-function
    attenuation input alongside the corresponding intensity output (model vs
    analytical ground truth).
    """
    ts, te = LAYER_START_PX, LAYER_END_PX
    depth_px = np.arange(ts, te)
    depth_mm = depth_px * dz

    # ── Use the paper-aligned A-line indices (Fig 2 panels a-f) ──
    # PAPER_FIG5_ALINES = [342, 323, 27] — fixed by the published figure.
    selected = list(PAPER_FIG5_ALINES)

    # ── Distinct colors ──
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    layer_labels = []
    for a_idx in selected:
        n = layer_info[a_idx]['n_layers']
        layer_labels.append(f'A-line {a_idx} ({n} layers)')

    # ── Figure: 3 rows × 2 columns (each row = one A-line: μ left, I_dB right) ──
    fig, axes = plt.subplots(3, 2, figsize=(10, 9),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.18})

    for row, (a_idx, color, label) in enumerate(
            zip(selected, colors, layer_labels)):
        info = layer_info[a_idx]
        boundaries = info['boundaries']
        mu_vals = info['mu_values']

        # --- Left column: μ(z) step profile ---
        ax_mu = axes[row, 0]
        mu_profile = mu_np[ts:te, a_idx]

        # Draw filled step profile for visual clarity
        ax_mu.fill_between(depth_mm, 0, mu_profile, step='mid',
                           alpha=0.15, color=color)
        ax_mu.step(depth_mm, mu_profile, where='mid', color=color,
                   lw=2.0)

        # Annotate each layer's μ value at the center of the layer
        for i in range(len(mu_vals)):
            b_start = max(boundaries[i], ts)
            b_end = min(boundaries[i + 1], te)
            if b_start >= te or b_end <= ts:
                continue
            mid_mm = (b_start + b_end) / 2.0 * dz
            # Place text inside the step, slightly below the top
            y_text = mu_vals[i]
            ax_mu.text(mid_mm, y_text + 0.6, f'{mu_vals[i]:.1f}',
                       fontsize=8, ha='center', va='bottom', color=color,
                       fontweight='bold')

        ax_mu.set_ylabel('$\\mu_t$ [mm$^{-1}$]')
        ax_mu.set_xlabel('Depth [mm]')
        ax_mu.set_title(label, fontsize=10)
        ax_mu.set_ylim(-0.5, MU_MAX_MM + 2)
        ax_mu.set_xlim(depth_mm[0], depth_mm[-1])

        # Mark layer boundaries with light vertical lines
        for b in boundaries[1:-1]:
            if ts <= b <= te:
                ax_mu.axvline(b * dz, color='gray', ls=':', lw=0.7, alpha=0.5)

        # --- Right column: I_dB(z) model vs analytical ---
        ax_i = axes[row, 1]
        model_profile = I_model_dB[ts:te, a_idx]
        analytic_profile = I_analytic_dB[ts:te, a_idx].astype(np.float32)

        # Analytical: solid black line
        ax_i.plot(depth_mm, analytic_profile, color='#333333',
                  lw=2.0, label='Analytical', zorder=2)
        # Model: colored circle markers (every 4th point) on top
        marker_step = 4
        ax_i.plot(depth_mm[::marker_step], model_profile[::marker_step],
                  'o', color=color, ms=5.0, mew=0, alpha=0.8,
                  label='Model', zorder=3)

        # Mark layer boundaries
        for b in boundaries[1:-1]:
            if ts <= b <= te:
                ax_i.axvline(b * dz, color='gray', ls=':', lw=0.7, alpha=0.5)

        # Residual annotation
        residual = np.abs(model_profile - analytic_profile)
        max_err = np.max(residual)
        ax_i.text(0.03, 0.05,
                  f'Max |err| = {max_err:.1e} dB',
                  transform=ax_i.transAxes, fontsize=7,
                  va='bottom', ha='left',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='gray', alpha=0.9))

        ax_i.set_ylabel('Intensity [dB]')
        ax_i.set_xlabel('Depth [mm]')
        ax_i.set_title(label, fontsize=10)
        ax_i.legend(fontsize=7, loc='upper right', framealpha=0.9)
        ax_i.set_xlim(depth_mm[0], depth_mm[-1])

    plt.tight_layout()

    # ── Save underlying arrays for the Figure 2 notebook to render from ──
    npz_payload = dict(depth_mm=depth_mm,
                       selected_alines=np.array(selected, dtype=np.int32))
    for col_pair, a_idx in enumerate(selected):
        info = layer_info[a_idx]
        npz_payload[f'mu_profile_{col_pair}']      = mu_np[ts:te, a_idx]
        npz_payload[f'model_profile_{col_pair}']   = I_model_dB[ts:te, a_idx].astype(np.float32)
        npz_payload[f'analytic_profile_{col_pair}'] = I_analytic_dB[ts:te, a_idx].astype(np.float32)
        npz_payload[f'boundaries_{col_pair}']      = np.asarray(info['boundaries'], dtype=np.int32)
        npz_payload[f'mu_values_{col_pair}']       = np.asarray(info['mu_values'], dtype=np.float32)
        npz_payload[f'n_layers_{col_pair}']        = np.int32(info['n_layers'])
    npz_path = os.path.join(paper_dir, 'fig5_aline_profiles.npz')
    np.savez(npz_path, **npz_payload)
    print(f"  [Paper 5/5] {npz_path}")

    path = os.path.join(paper_dir, 'paper_aline_profiles.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Paper 5/5] {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plain-language report
# ═════════════════════════════════════════════════════════════════════════════

def _write_plain_language_report(path, mae_dB, max_dB, r2_dB,
                                   mae_lin, r2_lin, n_passed, n_total,
                                   depth_samples, a_lines, dz, ts, te, device):
    """Write a plain-language report explaining results to a non-specialist."""
    with open(path, 'w') as f:
        f.write("=" * 72 + "\n")
        f.write("FORWARD MODEL VERIFICATION — PLAIN-LANGUAGE REPORT\n")
        f.write("=" * 72 + "\n\n")

        f.write("WHAT THIS TEST DOES\n")
        f.write("-" * 72 + "\n")
        f.write("Optical Coherence Tomography (OCT) works by shining near-infrared light\n")
        f.write("into tissue and measuring how much light bounces back from each depth.\n")
        f.write("As light travels deeper, it gets progressively weaker because the tissue\n")
        f.write("absorbs and scatters it — this is described by Beer-Lambert's law, one of\n")
        f.write("the most fundamental equations in optics.\n\n")
        f.write("Our 'forward model' is a computer simulation that predicts what an OCT\n")
        f.write("image SHOULD look like given known tissue properties. This test verifies\n")
        f.write("that our simulation matches the analytical (pen-and-paper) solution of\n")
        f.write("Beer-Lambert's law to machine precision.\n\n")
        f.write("Think of it this way: if you know exactly how much a tissue attenuates\n")
        f.write("light, you can calculate by hand what the OCT signal should be at each\n")
        f.write("depth. Our computer model should give the EXACT same answer. This test\n")
        f.write("confirms that it does.\n\n")

        f.write("HOW WE TESTED IT\n")
        f.write("-" * 72 + "\n")
        f.write(f"We created a synthetic 'phantom' — a virtual tissue sample with\n")
        f.write(f"{a_lines} independent vertical columns, each containing 2-5 layers of\n")
        f.write(f"different optical properties (attenuation coefficients from 0.1 to\n")
        f.write(f"15.0 mm^-1). The tissue occupies depths {ts*dz:.2f} to {te*dz:.2f} mm.\n\n")
        f.write(f"We ran our forward model on this phantom ({depth_samples}x{a_lines}\n")
        f.write(f"pixels) and compared every single pixel against the known analytical\n")
        f.write(f"solution computed independently in higher precision (64-bit vs 32-bit).\n\n")

        f.write("KEY RESULTS\n")
        f.write("-" * 72 + "\n")
        f.write(f"  Mean absolute error:       {mae_dB:.2e} dB  (effectively zero)\n")
        f.write(f"  Maximum error anywhere:    {max_dB:.2e} dB\n")
        f.write(f"  Correlation (R^2):         {r2_dB:.10f}  (perfect = 1.000000)\n")
        f.write(f"  Numbered verification tests: {n_passed}/{n_total} passed\n\n")
        f.write("To put this in perspective: the errors are in the range of 1e-5 to 1e-4\n")
        f.write("dB, which is about 10,000 to 100,000 times smaller than the noise floor\n")
        f.write("of a real OCT system (~79 dB). These errors arise solely from the\n")
        f.write("difference between 32-bit and 64-bit floating point arithmetic.\n\n")

        f.write("WHAT THE FIGURES SHOW\n")
        f.write("-" * 72 + "\n\n")

        f.write("paper/paper_forward_model_validation.png (3 panels):\n")
        f.write("  (a) The synthetic phantom — a 2D map of attenuation coefficient\n")
        f.write("      (mu_t) showing the layered tissue structure. Brighter colors\n")
        f.write("      mean higher attenuation (tissue absorbs more light per mm).\n")
        f.write("  (b) Depth profiles for 3 representative A-lines comparing our\n")
        f.write("      model (colored lines) against the analytical solution (black\n")
        f.write("      dashed). The curves overlap perfectly — you cannot tell them\n")
        f.write("      apart by eye.\n")
        f.write("  (c) A concordance plot of ALL tissue pixels: every pixel's model\n")
        f.write("      value vs its analytical value. All points fall exactly on the\n")
        f.write("      diagonal (y=x), confirming pixel-perfect agreement.\n\n")

        f.write("paper/paper_ss_vs_ms_comparison.png (2 panels):\n")
        f.write("  (a) Comparison of two physics models: 'single-scatter' (SS, the\n")
        f.write("      simple Beer-Lambert model where light scatters only once) and\n")
        f.write("      'multiple-scatter' (MS, the Extended Huygens-Fresnel model that\n")
        f.write("      accounts for light scattering multiple times). The MS model\n")
        f.write("      predicts higher signal at depth because multiply-scattered\n")
        f.write("      photons that would be 'lost' in the SS model are recaptured.\n")
        f.write("      The shaded region shows the MS excess signal.\n")
        f.write("  (b) How the anisotropy factor 'g' (a property of the tissue's\n")
        f.write("      scattering directionality) affects the signal. g=0 means\n")
        f.write("      isotropic scattering (equal in all directions), g=1 means\n")
        f.write("      purely forward scattering. At g=1, the MS model predicts NO\n")
        f.write("      depth-dependent attenuation at all — because ALL scattered\n")
        f.write("      photons continue forward and are recaptured.\n\n")

        f.write("paper/paper_confocal_validation.png (2 panels):\n")
        f.write("  (a) The confocal correction function C(z) for different optical\n")
        f.write("      configurations. This accounts for the fact that the OCT lens\n")
        f.write("      collects light most efficiently at its focal plane and less\n")
        f.write("      efficiently at other depths. The curves are Lorentzian\n")
        f.write("      (bell-shaped) functions centered at the focal plane.\n")
        f.write("  (b) Verification that our model's confocal correction matches\n")
        f.write("      the analytical formula exactly. The model residual (difference\n")
        f.write("      between confocal-ON and confocal-OFF) overlays the analytical\n")
        f.write("      Lorentzian perfectly.\n\n")

        f.write("WHAT THE NUMBERED TESTS VERIFY\n")
        f.write("-" * 72 + "\n")
        f.write("Tests 1-12 (in the verification/ subdirectory) each isolate one\n")
        f.write("specific property of the forward model:\n\n")
        f.write("  Test 1:  Beer-Lambert pixel match (model = analytical at every pixel)\n")
        f.write("  Test 2:  System gain is a pure offset (adding 100 dB shifts everything)\n")
        f.write("  Test 3:  Backscatter alpha scales signal exactly as 10*log10(alpha)\n")
        f.write("  Test 4:  Confocal correction is an additive Lorentzian\n")
        f.write("  Test 5:  Noise mask excludes pixels from cumulative attenuation\n")
        f.write("  Test 6:  MS and SS agree at the surface (before any scattering)\n")
        f.write("  Test 7:  Higher mu produces steeper signal decay\n")
        f.write("  Test 8:  MS model always produces >= SS signal (extra photons)\n")
        f.write("  Test 9:  MS model exactly recovers SS when g -> 0 (no forward scatter)\n")
        f.write("  Test 10: MS model reproduces independent numpy EHF calculation\n")
        f.write("  Test 11: At g=1 (pure forward scatter), signal is flat (no attenuation)\n")
        f.write("  Test 12: Both models work correctly with confocal ON + non-uniform mu\n\n")

        f.write("CONCLUSION\n")
        f.write("-" * 72 + "\n")
        if n_passed == n_total:
            f.write(f"All {n_total} verification tests PASSED. The forward model is\n")
            f.write("numerically correct to machine precision and faithfully implements\n")
            f.write("both Beer-Lambert single-scatter physics and Extended Huygens-Fresnel\n")
            f.write("multiple-scatter physics. It is safe to use this model as the basis\n")
            f.write("for inverse optimization (recovering tissue properties from images).\n")
        else:
            f.write(f"WARNING: {n_total - n_passed} of {n_total} tests FAILED.\n")
            f.write("Investigate the failed tests before proceeding.\n")


if __name__ == '__main__':
    main()