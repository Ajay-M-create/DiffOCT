#!/usr/bin/env python
"""Frame-level side-by-side: DiffOCT vs CF on one B-scan, plus the A-line fits.

Builds a diagnostic panel for a single B-scan (default 5.0% rep2, B-scan 256):

  row 1  target B-scan | DiffOCT predicted B-scan | residual (target - predicted)
  row 2  DiffOCT mu map | CF per-A-line mu map    | per-A-line mu comparison
  row 3  intensity vs depth for three A-lines: dB with the DiffOCT prediction
         overlaid, and ln(I/confocal) with CF's linear fit overlaid

The CF mu comes from `outputs/cf_allpixels/cf_allpixels_results.mat`
(ILPhantom_AllPixels_AllFrames.m), which fits one slope per A-line over that
A-line's own IntMask band. It is one value per A-line, so it renders as a
band-filled column rather than a depth-resolved map - that difference is itself
the point: CF cannot resolve mu in depth, DiffOCT can.

The CF fit is RECOMPUTED here in Python for the plotted A-lines and checked
against the MATLAB value, so the overlay cannot silently drift from the data.

Usage:
    python experiments/panel_diffoct_vs_cf_frame.py
    PHANTOM_INDEX=9 BSCAN=256 OUT_SUBDIR=phantom_uniform_init_fp2 \
        python experiments/panel_diffoct_vs_cf_frame.py
"""

import os
import sys
import numpy as np
import h5py
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.helpers import discover_despeckled_phantoms

OUT_SUBDIR = os.environ.get("OUT_SUBDIR", "phantom_uniform_init_fp2")
OUT_BASE = os.path.join(PHANTOM_ROOT, "outputs", OUT_SUBDIR)
MAT_DIR = os.path.join(PHANTOM_ROOT, "matlab_CF_scripts_results")
CF_MAT = os.path.join(PHANTOM_ROOT, "outputs", "cf_allpixels", "cf_allpixels_results.mat")
P_IDX = int(os.environ.get("PHANTOM_INDEX", 9))      # 9 = 5.0% rep2
BSCAN = int(os.environ.get("BSCAN", 256))
ALINES = [int(v) for v in os.environ.get("ALINES", "125,250,375").split(",")]

NOISE_FLOOR_DB = 78.910          # MATLAB value from ILPhantom_AllPixels_AllFrames
Z_FP, Z_R = 261.0, 131.5
Z_PX_MM = 11.0 / 2048.0
SCALE = -1.333 / (2 * Z_PX_MM)
K_DB = 20.0 / np.log(10.0)
DZ_MM = 0.004
Z_LO, Z_HI = 11, 50


def confocal(z):
    return 1.0 / (((z - Z_FP) / Z_R) ** 2 + 1.0)


def main():
    phantoms = discover_despeckled_phantoms(
        os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206"))
    ph = phantoms[P_IDX]
    conc = ph["concentration"]
    label = f"IL_{conc}pct_rep{ph['replicate']}"
    print(f"{label}  B-scan {BSCAN}")

    # ── target ──
    with tifffile.TiffFile(ph["path"]) as t:
        raw = t.asarray(key=BSCAN)                 # scalar key -> (500,500)
    dB_raw = raw.astype(np.float64) * (130.0 / 65535.0)
    N_lin = 10.0 ** (NOISE_FLOOR_DB / 10.0)
    lin = np.clip(10.0 ** (dB_raw / 10.0) - N_lin, 1e-12, None)
    tgt_dB = 10.0 * np.log10(lin)
    D, A = tgt_dB.shape

    # ── IntMask band ──
    with h5py.File(os.path.join(MAT_DIR, "IntMask.mat"), "r") as f:
        im = f["phantomMask"][P_IDX, BSCAN, :, :].T.astype(bool)   # (depth, aline)

    # ── DiffOCT ──
    fdir = os.path.join(OUT_BASE, label, f"bscan_{BSCAN:03d}")
    mu = np.load(os.path.join(fdir, "ss", "mu_final_mm.npy")).astype(np.float64)
    gain = float(np.load(os.path.join(fdir, "intmask_stats.npz"))["gain_dB"])
    z1 = np.arange(1, D + 1, dtype=np.float64)
    conf_dB = 10.0 * np.log10(confocal(z1))[:, None]
    cum = np.cumsum(mu * DZ_MM, axis=0)
    shifted = np.vstack([np.zeros((1, A)), cum[:-1, :]])
    with np.errstate(divide="ignore"):
        pred_dB = (gain + 10.0 * np.log10(np.maximum(mu, 1e-12))
                   + conf_dB - K_DB * shifted)
    print(f"  gain_dB = {gain:.2f}")

    # ── CF per-A-line mu (MATLAB all-pixels run) ──
    with h5py.File(CF_MAT, "r") as f:
        ref = f["muPerAline"][P_IDX, 0]
        cf_mu_vol = np.array(f[ref])               # (512 bscans, 500 alines)
    cf_mu_aline = cf_mu_vol[BSCAN, :].astype(np.float64)
    print(f"  CF mu this frame: mean={np.nanmean(cf_mu_aline):.3f}")

    # render CF mu as a band-filled frame
    cf_map = np.full((D, A), np.nan)
    for x in range(A):
        col = im[:, x]
        if col.any():
            cf_map[col, x] = cf_mu_aline[x]

    # DiffOCT per-A-line mean over the same band
    dl_mu_aline = np.full(A, np.nan)
    for x in range(A):
        col = im[:, x]
        if col.sum() >= 10:
            dl_mu_aline[x] = mu[col, x].mean()
    print(f"  DiffOCT mu this frame: mean={np.nanmean(dl_mu_aline):.3f}")

    # ── recompute CF's fit for the plotted A-lines (validates the overlay) ──
    lin_cor = lin / confocal(z1)[:, None]
    fits = {}
    for x in ALINES:
        col = np.where(im[:, x])[0]
        if len(col) < 10:
            continue
        y = np.log(np.maximum(lin_cor[col, x], 1e-30))
        zz = col.astype(np.float64)
        s, b = np.polyfit(zz, y, 1)
        fits[x] = dict(z=col, y=y, slope=s, intercept=b, mu=s * SCALE,
                       mu_matlab=cf_mu_aline[x], mu_diffoct=dl_mu_aline[x])
        print(f"  A-line {x}: CF mu recomputed={s*SCALE:7.3f}  "
              f"MATLAB={cf_mu_aline[x]:7.3f}  DiffOCT={dl_mu_aline[x]:7.3f}")

    # ── figure ──
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(17, 15))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 0.95], hspace=0.32,
                  wspace=0.25)

    msk = im
    vmin = float(np.percentile(tgt_dB[msk], 1))
    vmax = float(np.percentile(tgt_dB[msk], 99))
    show = lambda a: np.where(msk, a, np.nan)

    ax = fig.add_subplot(gs[0, 0])
    i0 = ax.imshow(show(tgt_dB), cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title("Target B-scan (dB, N(z)-subtracted)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Depth (px)"); plt.colorbar(i0, ax=ax, fraction=0.046, label="dB")

    ax = fig.add_subplot(gs[0, 1])
    i1 = ax.imshow(show(pred_dB), cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title("DiffOCT predicted B-scan (dB)", fontsize=12, fontweight="bold")
    plt.colorbar(i1, ax=ax, fraction=0.046, label="dB")

    ax = fig.add_subplot(gs[0, 2])
    resid = show(tgt_dB - pred_dB)
    lim = float(np.nanpercentile(np.abs(resid), 99))
    i2 = ax.imshow(resid, cmap="RdBu_r", aspect="auto", vmin=-lim, vmax=lim)
    rms = float(np.sqrt(np.nanmean(resid[msk] ** 2)))
    ax.set_title(f"Residual target - predicted (RMS={rms:.2f} dB)",
                 fontsize=12, fontweight="bold")
    plt.colorbar(i2, ax=ax, fraction=0.046, label="dB")

    mu_hi = float(np.nanpercentile(np.concatenate(
        [mu[msk], cf_mu_aline[np.isfinite(cf_mu_aline)]]), 99))

    ax = fig.add_subplot(gs[1, 0])
    i3 = ax.imshow(show(mu), cmap="plasma", aspect="auto", vmin=0, vmax=mu_hi)
    ax.set_title(f"DiffOCT $\\mu$ (depth-resolved), mean={np.nanmean(dl_mu_aline):.2f}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Depth (px)"); ax.set_xlabel("A-line")
    plt.colorbar(i3, ax=ax, fraction=0.046, label="mm$^{-1}$")

    ax = fig.add_subplot(gs[1, 1])
    i4 = ax.imshow(cf_map, cmap="plasma", aspect="auto", vmin=0, vmax=mu_hi)
    ax.set_title(f"CF $\\mu$ (one value per A-line), mean={np.nanmean(cf_mu_aline):.2f}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("A-line")
    plt.colorbar(i4, ax=ax, fraction=0.046, label="mm$^{-1}$")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(cf_mu_aline, color="#2E8B57", lw=0.8, alpha=0.85, label="CF (per A-line fit)")
    ax.plot(dl_mu_aline, color="#E8943A", lw=0.8, alpha=0.85, label="DiffOCT (band mean)")
    ax.axhline(np.nanmean(cf_mu_aline), color="#2E8B57", ls="--", lw=1.4)
    ax.axhline(np.nanmean(dl_mu_aline), color="#E8943A", ls="--", lw=1.4)
    for x in ALINES:
        ax.axvline(x, color="k", ls=":", lw=0.9, alpha=0.6)
    ax.set_xlabel("A-line"); ax.set_ylabel(r"$\mu$ [mm$^{-1}$]")
    ax.set_title("Per-A-line $\\mu$ (dashed = frame mean)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); ax.set_ylim(0, mu_hi)

    # ── row 3: A-line profiles ──
    for k, x in enumerate(ALINES):
        if x not in fits:
            continue
        f_ = fits[x]
        ax = fig.add_subplot(gs[2, k])
        zc = f_["z"]
        ax.plot(zc, f_["y"], color="0.35", lw=1.0, label="ln(I / confocal), data")
        ax.plot(zc, f_["intercept"] + f_["slope"] * zc, color="#2E8B57", lw=2.2,
                label=f"CF linear fit -> $\\mu$={f_['mu']:.2f}")
        # DiffOCT prediction in the same units: dB -> ln, confocal removed
        pr = (pred_dB[zc, x] - 10.0 * np.log10(confocal(zc + 1.0))) \
            * (np.log(10.0) / 10.0)
        ax.plot(zc, pr + (f_["y"].mean() - pr.mean()), color="#E8943A", lw=1.8,
                label=f"DiffOCT pred -> $\\mu$={f_['mu_diffoct']:.2f}")
        ax.set_xlabel("Depth (px)")
        if k == 0:
            ax.set_ylabel("ln(I / confocal)  [offset-matched]")
        ax.set_title(f"A-line {x}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.25)

    fig.suptitle(f"{label}  B-scan {BSCAN}   |   DiffOCT ({OUT_SUBDIR}) vs CF all-pixels",
                 fontsize=15, fontweight="bold", y=0.995)
    out = os.path.join(OUT_BASE, f"panel_vs_cf_{label}_bscan{BSCAN:03d}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
