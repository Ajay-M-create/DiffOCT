#!/usr/bin/env python
"""Rebuttal figure: DiffOCT tracks the A-scan, CF only fits a line through it.

Argument the figure makes
-------------------------
CF and DiffOCT disagree on mu not because either is "wrong" but because they
answer different questions. CF assumes the depth profile IS a straight line in
log space and reports its slope. DiffOCT reproduces the profile itself. On a
real A-scan the profile is not a straight line, so:

  * CF's residual has systematic, depth-dependent structure (it cannot bend),
  * DiffOCT's residual is small and unstructured,
  * and CF's per-A-line mu is wildly unstable (2 to >12 mm^-1 on a UNIFORM
    phantom) because a line fit to 40 speckly samples is a high-variance
    estimator. CF's published values block-average 8x32 neighbourhoods first,
    precisely to hide this.

Panels
------
  (a) all 512 B-scan mean depth profiles, faint, with the mean data / CF fit /
      DiffOCT prediction overlaid
  (b) residual vs depth: 512 faint curves per method + median and IQR
  (c) per-A-line RMS residual, CF vs DiffOCT
  (d) per-A-line mu distribution, CF vs DiffOCT, with CF's published block value

Everything is computed on CF's own terms: ln(I/confocal) inside each A-line's
own IntMask band, CF's noise floor, CF's confocal parameters. The CF fit is
recomputed here and cross-checked against the MATLAB all-pixels result.

Usage:
    python experiments/figure_cf_vs_diffoct_profiles.py
    PHANTOM_INDEX=1 python experiments/figure_cf_vs_diffoct_profiles.py
"""

import os
import sys
import numpy as np
import h5py
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.helpers import discover_despeckled_phantoms

OUT_SUBDIR = os.environ.get("OUT_SUBDIR", "phantom_uniform_init_fp2")
OUT_BASE = os.path.join(PHANTOM_ROOT, "outputs", OUT_SUBDIR)
MAT_DIR = os.path.join(PHANTOM_ROOT, "matlab_CF_scripts_results")
CF_MAT = os.path.join(PHANTOM_ROOT, "outputs", "cf_allpixels", "cf_allpixels_results.mat")
P_IDX = int(os.environ.get("PHANTOM_INDEX", 9))
ALINE_STEP = int(os.environ.get("ALINE_STEP", 4))    # subsample A-lines per frame

NOISE_FLOOR_DB = 78.910
Z_FP, Z_R = 261.0, 131.5
SCALE = -1.333 / (2 * (11.0 / 2048.0))
K_DB = 20.0 / np.log(10.0)
DZ_MM = 0.004
NB = 40                                              # band length, surface+11..+50
LN10_10 = np.log(10.0) / 10.0                        # dB -> natural log

CF_PUBLISHED = {1.0: 4.478, 1.5: 5.947, 2.0: 5.774, 3.0: 7.804, 5.0: 9.280}
C_CF, C_DL, C_DATA = "#2E8B57", "#E8943A", "0.35"


def confocal(z):
    return 1.0 / (((z - Z_FP) / Z_R) ** 2 + 1.0)


def main():
    ph = discover_despeckled_phantoms(
        os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206"))[P_IDX]
    conc = ph["concentration"]
    label = f"IL_{conc}pct_rep{ph['replicate']}"
    print(f"{label}  (all 512 B-scans, every {ALINE_STEP}th A-line)")

    with tifffile.TiffFile(ph["path"]) as t:
        raw = t.asarray()                                   # (512, 500, 500)
    dB = raw.astype(np.float64) * (130.0 / 65535.0)
    lin = np.clip(10.0 ** (dB / 10.0) - 10.0 ** (NOISE_FLOOR_DB / 10.0), 1e-12, None)
    del raw, dB
    with h5py.File(os.path.join(MAT_DIR, "IntMask.mat"), "r") as f:
        im_all = f["phantomMask"][P_IDX, :, :, :]           # (512, aline, depth)
    im_all = im_all.transpose(0, 2, 1).astype(bool)          # (512, depth, aline)

    D = lin.shape[1]
    z1 = np.arange(1, D + 1, dtype=np.float64)
    conf = confocal(z1)[:, None]
    conf_dB = 10.0 * np.log10(conf)
    rel = np.arange(NB, dtype=np.float64)

    prof_data, prof_cf, prof_dl = [], [], []
    rms_cf, rms_dl, mu_cf, mu_dl = [], [], [], []

    for fi in range(512):
        fdir = os.path.join(OUT_BASE, label, f"bscan_{fi:03d}")
        mu = np.load(os.path.join(fdir, "ss", "mu_final_mm.npy")).astype(np.float64)
        gain = float(np.load(os.path.join(fdir, "intmask_stats.npz"))["gain_dB"])
        cum = np.cumsum(mu * DZ_MM, axis=0)
        shifted = np.vstack([np.zeros((1, mu.shape[1])), cum[:-1, :]])
        with np.errstate(divide="ignore"):
            pred_dB = (gain + 10.0 * np.log10(np.maximum(mu, 1e-12))
                       + conf_dB - K_DB * shifted)
        # natural-log, confocal-removed  (CF's fitting space)
        y_dat = np.log(np.maximum(lin[fi] / conf, 1e-30))
        y_prd = (pred_dB - conf_dB) * LN10_10

        im = im_all[fi]
        fd, fc, fl = [], [], []
        for x in range(0, im.shape[1], ALINE_STEP):
            col = np.flatnonzero(im[:, x])
            if col.size < NB:
                continue
            idx = col[:NB]
            d = y_dat[idx, x]
            p = y_prd[idx, x]
            s, b = np.polyfit(rel, d, 1)                    # CF: line through the data
            c = b + s * rel
            # offset-match the prediction so only SHAPE is compared
            p = p - p.mean() + d.mean()
            fd.append(d - d[0]); fc.append(c - d[0]); fl.append(p - d[0])
            rms_cf.append(np.sqrt(np.mean((d - c) ** 2)))
            rms_dl.append(np.sqrt(np.mean((d - p) ** 2)))
            mu_cf.append(s * SCALE)
            mu_dl.append(mu[idx, x].mean())
        if fd:
            prof_data.append(np.mean(fd, axis=0))
            prof_cf.append(np.mean(fc, axis=0))
            prof_dl.append(np.mean(fl, axis=0))
        if fi % 128 == 0:
            print(f"  frame {fi}/512")

    P = np.array(prof_data); C = np.array(prof_cf); L = np.array(prof_dl)
    rms_cf = np.array(rms_cf); rms_dl = np.array(rms_dl)
    mu_cf = np.array(mu_cf); mu_dl = np.array(mu_dl)
    print(f"  {P.shape[0]} frame profiles, {rms_cf.size} A-lines")
    print(f"  RMS residual: CF={rms_cf.mean():.4f}  DiffOCT={rms_dl.mean():.4f}  "
          f"ratio={rms_cf.mean()/rms_dl.mean():.2f}x")
    print(f"  mu: CF mean={mu_cf.mean():.3f} sd={mu_cf.std():.3f} | "
          f"DiffOCT mean={mu_dl.mean():.3f} sd={mu_dl.std():.3f}")

    # cross-check the recomputed CF fit against the MATLAB all-pixels result
    try:
        with h5py.File(CF_MAT, "r") as f:
            ml = np.array(f[f["muPerAline"][P_IDX, 0]])     # (512, 500)
        print(f"  cross-check vs MATLAB all-pixels: "
              f"{np.nanmean(ml):.3f} vs recomputed {mu_cf.mean():.3f}")
    except Exception as e:                                   # noqa: BLE001
        print(f"  [cross-check skipped: {e}]")

    # ── figure ──
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    # (a) profiles
    a = ax[0, 0]
    for r in P:
        a.plot(rel, r, color=C_DATA, lw=0.4, alpha=0.06)
    a.plot(rel, P.mean(0), color="black", lw=2.6, label="Measured A-scan (mean)", zorder=5)
    a.plot(rel, C.mean(0), color=C_CF, lw=2.4, ls="--",
           label="CF: straight-line fit", zorder=6)
    a.plot(rel, L.mean(0), color=C_DL, lw=2.4,
           label="DiffOCT: forward-model prediction", zorder=6)
    a.set_xlabel("Depth into tissue band (px, surface+11 .. +50)", fontsize=12)
    a.set_ylabel("ln(I / confocal), referenced to band top", fontsize=12)
    a.set_title(f"(a) Depth profiles, all 512 B-scans of {label}",
                fontsize=12.5, fontweight="bold")
    a.legend(fontsize=10.5, loc="lower left"); a.grid(alpha=0.25)

    # (b) residuals
    a = ax[0, 1]
    rc = C - P; rl = L - P
    for r in rc:
        a.plot(rel, r, color=C_CF, lw=0.4, alpha=0.05)
    for r in rl:
        a.plot(rel, r, color=C_DL, lw=0.4, alpha=0.05)
    for arr, col, name in ((rc, C_CF, "CF"), (rl, C_DL, "DiffOCT")):
        med = np.median(arr, 0)
        a.plot(rel, med, color=col, lw=2.8, zorder=6,
               label=f"{name} median (|max|={np.abs(med).max():.3f})")
        a.fill_between(rel, np.percentile(arr, 25, axis=0),
                       np.percentile(arr, 75, axis=0), color=col, alpha=0.22, zorder=4)
    a.axhline(0, color="black", lw=1.0, ls=":")
    a.set_xlabel("Depth into tissue band (px)", fontsize=12)
    a.set_ylabel("Model - data  [ln units]", fontsize=12)
    a.set_title("(b) Residual: CF cannot bend, DiffOCT follows the A-scan",
                fontsize=12.5, fontweight="bold")
    a.legend(fontsize=10.5); a.grid(alpha=0.25)

    # (c) RMS residual
    a = ax[1, 0]
    hi = float(np.percentile(np.concatenate([rms_cf, rms_dl]), 99.5))
    bins = np.linspace(0, hi, 90)
    a.hist(rms_cf, bins=bins, color=C_CF, alpha=0.65, label=f"CF (mean {rms_cf.mean():.3f})")
    a.hist(rms_dl, bins=bins, color=C_DL, alpha=0.65,
           label=f"DiffOCT (mean {rms_dl.mean():.3f})")
    a.axvline(rms_cf.mean(), color=C_CF, lw=2, ls="--")
    a.axvline(rms_dl.mean(), color=C_DL, lw=2, ls="--")
    a.set_xlabel("Per-A-line RMS residual [ln units]", fontsize=12)
    a.set_ylabel("A-line count", fontsize=12)
    a.set_title(f"(c) Goodness of fit: DiffOCT is {rms_cf.mean()/rms_dl.mean():.1f}x closer "
                f"to the data", fontsize=12.5, fontweight="bold")
    a.legend(fontsize=10.5); a.grid(alpha=0.25)

    # (d) mu distributions
    a = ax[1, 1]
    lo, hi = 0, float(np.percentile(mu_cf, 99))
    bins = np.linspace(lo, hi, 100)
    a.hist(mu_cf, bins=bins, color=C_CF, alpha=0.6,
           label=f"CF per-A-line (sd {mu_cf.std():.2f})")
    a.hist(mu_dl, bins=bins, color=C_DL, alpha=0.6,
           label=f"DiffOCT per-A-line (sd {mu_dl.std():.2f})")
    a.axvline(CF_PUBLISHED[conc], color="black", lw=2.4, ls="-",
              label=f"CF published (block-averaged) = {CF_PUBLISHED[conc]:.2f}")
    a.axvline(mu_dl.mean(), color=C_DL, lw=2, ls="--")
    a.set_xlabel(r"$\mu$ [mm$^{-1}$]", fontsize=12)
    a.set_ylabel("A-line count", fontsize=12)
    a.set_title(r"(d) Per-A-line $\mu$: CF is high-variance on a UNIFORM phantom",
                fontsize=12.5, fontweight="bold")
    a.legend(fontsize=10); a.grid(alpha=0.25)

    fig.suptitle(f"Why CF and DiffOCT disagree: {conc}% Intralipid, focal position 2 "
                 f"(n = 512 B-scans)", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUT_BASE, f"figure_cf_vs_diffoct_profiles_{label}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    np.savez_compressed(
        os.path.join(OUT_BASE, f"figure_cf_vs_diffoct_profiles_{label}.npz"),
        rel=rel, prof_data=P, prof_cf=C, prof_diffoct=L,
        rms_cf=rms_cf, rms_diffoct=rms_dl, mu_cf=mu_cf, mu_diffoct=mu_dl,
        cf_published=CF_PUBLISHED[conc], concentration=conc)


if __name__ == "__main__":
    main()
