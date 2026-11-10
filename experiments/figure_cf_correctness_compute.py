#!/usr/bin/env python
"""Compute + cache every quantity needed by the CF-vs-DiffOCT correctness figure.

Addresses Reviewer 1's correctness cluster on COMMSENG-26-0633:

  R1.3  the 0.68 vs 1.16 slope drop, unexplained for a homogeneous phantom
        where "both methods invert the same equation"
  R1.4  Fig 3l IQR (~2.5-5 for 1% focal-1) vs the reported 3.88 +/- 0.27
  R1.5  what the Fig 3l outliers are, given noisy pixels were masked
  R1.7  the method's response to speckle

Writes one npz; `figure_cf_correctness.py` does the plotting. Split so the slow
part (reading ~3000 per-frame arrays and two TIFF volumes) runs once.

Usage:
    python experiments/figure_cf_correctness_compute.py
"""

import os
import sys
import numpy as np
import h5py
import tifffile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.helpers import discover_despeckled_phantoms

NEW = os.path.join(PHANTOM_ROOT, "outputs", "phantom_uniform_init_fp2")
ORIG = "/home/gridsan/groups/erelab/Ajay/DiffOCT-Phantom-Experiments/outputs/phantom_validation_ss"
MAT_DIR = os.path.join(PHANTOM_ROOT, "matlab_CF_scripts_results")
OUT_NPZ = os.path.join(PHANTOM_ROOT, "outputs", "cf_correctness_figure_data.npz")

CONCS = [1.0, 1.5, 2.0, 3.0, 5.0]
FP2_IDX = [1, 3, 5, 7, 9]
NB = 40
NOISE_FLOOR_DB = 78.910
Z_FP, Z_R = 261.0, 131.5
SCALE = -1.333 / (2 * (11.0 / 2048.0))
K_DB = 20.0 / np.log(10.0)
DZ_MM = 0.004
LN10_10 = np.log(10.0) / 10.0

# Measured elsewhere in this investigation; carried through for the slope panel.
CF_PUBLISHED = np.array([4.478, 5.947, 5.774, 7.804, 9.280])   # attCoef2, FP2
CF_ALLPIXELS = np.array([5.038, 6.787, 6.636, 8.581, 9.847])   # ILPhantom_AllPixels
DIFFOCT_MEAN = np.array([4.597, 5.581, 5.789, 6.970, 7.727])   # IntMask depth-mean
DIFFOCT_CFEST = np.array([5.138, 6.065, 6.047, 7.697, 8.909])  # CF estimator on prediction


def confocal(z):
    return 1.0 / (((z - Z_FP) / Z_R) ** 2 + 1.0)


def main():
    ph = discover_despeckled_phantoms(
        os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206"))
    out = {}

    # ─────────────────────────────────────────────────────────────────
    # R1.3(a): mu(z) inside the band, per concentration.
    # The mechanism: mu enters the SS model TWICE, as backscatter
    # 10log10(mu) and as attenuation -K*cumsum(mu*dz). A mu that ramps
    # DOWN with depth supplies part of the observed decay through the
    # backscatter term, so its depth-MEAN sits below the log-slope that
    # CF measures. Both describe the same A-scan.
    # ─────────────────────────────────────────────────────────────────
    print("R1.3(a) mu(z) depth profiles ...")
    FR = list(range(0, 512, 16))
    mu_prof = np.zeros((5, NB))
    slope_implied = np.zeros(5)
    depth_mean = np.zeros(5)
    for ci, p_idx in enumerate(FP2_IDX):
        conc = ph[p_idx]["concentration"]
        label = f"IL_{conc}pct_rep2"
        with h5py.File(os.path.join(MAT_DIR, "IntMask.mat"), "r") as f:
            im = f["phantomMask"][p_idx, FR, :, :]
        im = im.transpose(0, 2, 1).astype(bool)
        acc = np.zeros(NB); n = 0; imp = []; dm = []
        for j, fi in enumerate(FR):
            mu = np.load(os.path.join(NEW, label, f"bscan_{fi:03d}",
                                      "ss", "mu_final_mm.npy")).astype(np.float64)
            for x in range(0, 500, 4):
                col = np.flatnonzero(im[j][:, x])
                if col.size < NB:
                    continue
                v = mu[col[:NB], x]
                acc += v; n += 1
                dm.append(v.mean())
                lnI = np.log(np.maximum(v, 1e-12)) - K_DB * LN10_10 * np.cumsum(v * DZ_MM)
                imp.append(np.polyfit(np.arange(NB, dtype=float), lnI, 1)[0] * SCALE)
        mu_prof[ci] = acc / max(n, 1)
        slope_implied[ci] = np.mean(imp)
        depth_mean[ci] = np.mean(dm)
        print(f"   {conc}%  mu(top)={mu_prof[ci][:5].mean():.3f} "
              f"mu(bot)={mu_prof[ci][-5:].mean():.3f} "
              f"mean={depth_mean[ci]:.3f} slope-implied={slope_implied[ci]:.3f}")
    out.update(mu_prof=mu_prof, slope_implied=slope_implied, depth_mean=depth_mean)

    # ─────────────────────────────────────────────────────────────────
    # R1.3(b) + R1.7: profile fit quality and per-A-line spread, 5.0%.
    # ─────────────────────────────────────────────────────────────────
    print("R1.3(b)/R1.7 profile fits, 5.0% ...")
    p_idx = 9
    label = "IL_5.0pct_rep2"
    with tifffile.TiffFile(ph[p_idx]["path"]) as t:
        raw = t.asarray()
    lin = np.clip(10.0 ** ((raw.astype(np.float64) * (130.0 / 65535.0)) / 10.0)
                  - 10.0 ** (NOISE_FLOOR_DB / 10.0), 1e-12, None)
    del raw
    with h5py.File(os.path.join(MAT_DIR, "IntMask.mat"), "r") as f:
        im_all = f["phantomMask"][p_idx, :, :, :].transpose(0, 2, 1).astype(bool)
    D = lin.shape[1]
    conf = confocal(np.arange(1, D + 1, dtype=np.float64))[:, None]
    conf_dB = 10.0 * np.log10(conf)
    rel = np.arange(NB, dtype=float)

    pd_, pc_, pl_, rc_, rl_, mc_, ml_ = [], [], [], [], [], [], []
    for fi in range(0, 512, 2):
        fdir = os.path.join(NEW, label, f"bscan_{fi:03d}")
        mu = np.load(os.path.join(fdir, "ss", "mu_final_mm.npy")).astype(np.float64)
        gain = float(np.load(os.path.join(fdir, "intmask_stats.npz"))["gain_dB"])
        cum = np.cumsum(mu * DZ_MM, axis=0)
        sh = np.vstack([np.zeros((1, mu.shape[1])), cum[:-1, :]])
        with np.errstate(divide="ignore"):
            pdB = gain + 10.0 * np.log10(np.maximum(mu, 1e-12)) + conf_dB - K_DB * sh
        ydat = np.log(np.maximum(lin[fi] / conf, 1e-30))
        yprd = (pdB - conf_dB) * LN10_10
        fd, fc, fl = [], [], []
        for x in range(0, 500, 4):
            col = np.flatnonzero(im_all[fi][:, x])
            if col.size < NB:
                continue
            idx = col[:NB]
            d = ydat[idx, x]; p = yprd[idx, x]
            s, b = np.polyfit(rel, d, 1)
            c = b + s * rel
            p = p - p.mean() + d.mean()
            fd.append(d - d[0]); fc.append(c - d[0]); fl.append(p - d[0])
            rc_.append(np.sqrt(np.mean((d - c) ** 2)))
            rl_.append(np.sqrt(np.mean((d - p) ** 2)))
            mc_.append(s * SCALE); ml_.append(mu[idx, x].mean())
        if fd:
            pd_.append(np.mean(fd, 0)); pc_.append(np.mean(fc, 0)); pl_.append(np.mean(fl, 0))
    out.update(rel=rel, prof_data=np.array(pd_), prof_cf=np.array(pc_),
               prof_dl=np.array(pl_), rms_cf=np.array(rc_), rms_dl=np.array(rl_),
               mu_cf_aline=np.array(mc_), mu_dl_aline=np.array(ml_))
    print(f"   RMS residual CF={np.mean(rc_):.4f} DiffOCT={np.mean(rl_):.4f} "
          f"({np.mean(rc_)/np.mean(rl_):.2f}x)")
    del lin, im_all

    # ─────────────────────────────────────────────────────────────────
    # R1.4: the phantom R1 actually cites - 1.0% FOCAL 1 (rep1), from the
    # ORIGINAL Figure 3 run. Per-pixel spread vs per-frame-mean spread.
    # ─────────────────────────────────────────────────────────────────
    print("R1.4 per-pixel vs per-frame, IL_1.0pct_rep1 (focal 1, original run) ...")
    lab1 = "IL_1.0pct_rep1"
    fmeans = []
    for fi in range(512):
        p = os.path.join(ORIG, lab1, f"bscan_{fi:03d}", "intmask_mu.npy")
        if os.path.exists(p):
            fmeans.append(np.load(p).mean())
    fmeans = np.array(fmeans)
    px = np.load(os.path.join(ORIG, "boxplot_pixel_data.npz"),
                 allow_pickle=True)["IL_1.0pct_rep1"].astype(np.float64)
    q1, med, q3 = np.percentile(px, [25, 50, 75])
    out.update(r14_frame_means=fmeans, r14_pixels=px[::37],
               r14_pix_q1=q1, r14_pix_med=med, r14_pix_q3=q3,
               r14_pix_mean=px.mean(),
               r14_frame_mean=fmeans.mean(), r14_frame_sd=fmeans.std())
    print(f"   per-pixel IQR = {q1:.2f}-{q3:.2f} (mean {px.mean():.2f}, n={px.size:,})")
    print(f"   per-frame mean = {fmeans.mean():.3f} +/- {fmeans.std():.3f} "
          f"(n={fmeans.size})  <- the reported '3.88 +/- 0.27'")

    # ─────────────────────────────────────────────────────────────────
    # R1.5: where do the Fig 3l outliers live? 2D density of mu vs depth
    # in band, plus the fraction below 1 mm^-1 by concentration.
    # ─────────────────────────────────────────────────────────────────
    print("R1.5 outlier anatomy ...")
    H = np.zeros((NB, 120)); edges = np.linspace(0, 18, 121)
    frac_lo = np.zeros(5); frac_at_min = np.zeros(5)
    for ci, p_idx in enumerate(FP2_IDX):
        conc = ph[p_idx]["concentration"]
        label = f"IL_{conc}pct_rep2"
        v = np.load(os.path.join(NEW, "boxplot_pixel_data.npz"),
                    allow_pickle=True)[label].astype(np.float64)
        frac_lo[ci] = np.mean(v < 1.0)
        frac_at_min[ci] = np.mean(v <= 0.02)
        if p_idx != 9:
            continue
        with h5py.File(os.path.join(MAT_DIR, "IntMask.mat"), "r") as f:
            im = f["phantomMask"][p_idx, FR, :, :].transpose(0, 2, 1).astype(bool)
        for j, fi in enumerate(FR):
            mu = np.load(os.path.join(NEW, label, f"bscan_{fi:03d}",
                                      "ss", "mu_final_mm.npy")).astype(np.float64)
            for x in range(0, 500, 2):
                col = np.flatnonzero(im[j][:, x])
                if col.size < NB:
                    continue
                vv = mu[col[:NB], x]
                for k in range(NB):
                    b = np.searchsorted(edges, vv[k]) - 1
                    if 0 <= b < 120:
                        H[k, b] += 1
    out.update(r15_hist=H, r15_edges=edges, r15_frac_below1=frac_lo,
               r15_frac_at_min=frac_at_min)
    print(f"   frac mu<1: {np.round(frac_lo*100,2)} %   frac at mu_min: "
          f"{np.round(frac_at_min*100,3)} %")

    # ─────────────────────────────────────────────────────────────────
    # Example A-scans for the profile row. Selected by PERCENTILE of the
    # per-A-line (CF - DiffOCT) difference within one B-scan, not hand-picked,
    # so the three span the real range of behaviour rather than flattering
    # either method: CF overshoots / they agree / CF undershoots.
    # ─────────────────────────────────────────────────────────────────
    print("A-scan examples ...")
    FI = 256
    with tifffile.TiffFile(ph[9]["path"]) as t:
        raw = t.asarray(key=FI)
    lin1 = np.clip(10.0 ** ((raw.astype(np.float64) * (130.0 / 65535.0)) / 10.0)
                   - 10.0 ** (NOISE_FLOOR_DB / 10.0), 1e-12, None)
    with h5py.File(os.path.join(MAT_DIR, "IntMask.mat"), "r") as f:
        im1 = f["phantomMask"][9, FI, :, :].T.astype(bool)
    fdir = os.path.join(NEW, "IL_5.0pct_rep2", f"bscan_{FI:03d}")
    mu1 = np.load(os.path.join(fdir, "ss", "mu_final_mm.npy")).astype(np.float64)
    gain1 = float(np.load(os.path.join(fdir, "intmask_stats.npz"))["gain_dB"])
    Dd = lin1.shape[0]
    conf1 = confocal(np.arange(1, Dd + 1, dtype=np.float64))[:, None]
    confdB1 = 10.0 * np.log10(conf1)
    cum1 = np.cumsum(mu1 * DZ_MM, axis=0)
    sh1 = np.vstack([np.zeros((1, mu1.shape[1])), cum1[:-1, :]])
    with np.errstate(divide="ignore"):
        pdB1 = gain1 + 10.0 * np.log10(np.maximum(mu1, 1e-12)) + confdB1 - K_DB * sh1
    ydat1 = np.log(np.maximum(lin1 / conf1, 1e-30))
    yprd1 = (pdB1 - confdB1) * LN10_10

    cand = []
    for x in range(500):
        col = np.flatnonzero(im1[:, x])
        if col.size < NB:
            continue
        idx = col[:NB]
        dprof = ydat1[idx, x]
        sl, bb = np.polyfit(rel, dprof, 1)
        cand.append((x, idx, dprof, bb + sl * rel, yprd1[idx, x],
                     sl * SCALE, mu1[idx, x].mean()))
    diffs = np.array([c[5] - c[6] for c in cand])
    picks = [int(np.argsort(diffs)[int(q * (len(diffs) - 1))]) for q in (0.95, 0.50, 0.05)]
    ex_d, ex_c, ex_p, ex_meta = [], [], [], []
    for k in picks:
        x, idx, dprof, cfit, pprof, mu_cf, mu_dl = cand[k]
        pprof = pprof - pprof.mean() + dprof.mean()
        ex_d.append(dprof); ex_c.append(cfit); ex_p.append(pprof)
        ex_meta.append([x, mu_cf, mu_dl])
        print(f"   A-line {x}: CF={mu_cf:.2f}  DiffOCT={mu_dl:.2f}  diff={mu_cf-mu_dl:+.2f}")
    out.update(ex_data=np.array(ex_d), ex_cf=np.array(ex_c), ex_pred=np.array(ex_p),
               ex_meta=np.array(ex_meta), ex_frame=FI)

    out.update(concs=np.array(CONCS), cf_published=CF_PUBLISHED,
               cf_allpixels=CF_ALLPIXELS, diffoct_mean=DIFFOCT_MEAN,
               diffoct_cfest=DIFFOCT_CFEST)
    np.savez_compressed(OUT_NPZ, **out)
    print(f"\nSaved {OUT_NPZ}")


if __name__ == "__main__":
    main()
