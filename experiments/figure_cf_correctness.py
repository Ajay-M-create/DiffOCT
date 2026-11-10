#!/usr/bin/env python
"""Results figure: why CF and DiffOCT report different mu, and why both are right.

Reads `outputs/cf_correctness_figure_data.npz` (from
figure_cf_correctness_compute.py) and renders a six-panel figure targeting
Reviewer 1's correctness comments on COMMSENG-26-0633.

  (a) R1.3  mu(z) inside the band: DiffOCT resolves a depth ramp, CF reports one
            number. The ramp is the mechanism.
  (b) R1.3  both models reproduce the measured A-scan equally well, so neither
            is mis-fitting the data.
  (c) R1.3  measure DiffOCT's own output CF's way and the slope gap mostly closes.
  (d) R1.4  per-pixel spread vs per-frame-mean spread, the two numbers R1 found
            contradictory.
  (e) R1.5  where the Fig 3l outliers live.
  (f) R1.7  response to speckle: fit closeness and estimator variance.
  (g-i) R1.3/R1.7  individual A-scans: DiffOCT follows the depth structure in
            the data, CF forces a straight line through it. Chosen by
            percentile of the per-A-line (CF - DiffOCT) difference, so they
            span the real range rather than being hand-picked.

Usage:
    python experiments/figure_cf_correctness.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
NPZ = os.path.join(PHANTOM_ROOT, "outputs", "cf_correctness_figure_data.npz")
OUT = os.path.join(PHANTOM_ROOT, "outputs", "figure_cf_correctness")

C_CF, C_DL, C_DATA = "#2E8B57", "#E8943A", "0.25"


def fit(x, y):
    co = np.polyfit(x, y, 1)
    p = np.polyval(co, x)
    return co[0], co[1], 1 - np.sum((y - p) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)


def panel_label(ax, s, dx=-0.13, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="top", ha="left")


def main():
    if not os.path.isfile(NPZ):
        sys.exit(f"missing {NPZ} - run figure_cf_correctness_compute.py first")
    d = np.load(NPZ, allow_pickle=True)
    concs = d["concs"]
    rel = d["rel"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 1.0, "axes.labelsize": 12, "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5, "legend.frameon": True, "legend.framealpha": 0.92,
    })
    fig, axes = plt.subplots(3, 3, figsize=(18.5, 15.6))
    cmap = plt.get_cmap("plasma")
    ccol = [cmap(i / (len(concs) - 1) * 0.85) for i in range(len(concs))]

    # ── (a) mu(z): the mechanism ────────────────────────────────────────
    ax = axes[0, 0]
    mp = d["mu_prof"]
    for i, c in enumerate(concs):
        ax.plot(rel, mp[i], color=ccol[i], lw=2.4, label=f"{c:.1f}%")
        ax.axhline(d["cf_published"][i], color=ccol[i], lw=1.3, ls=":", alpha=0.9)
    ax.annotate("", xy=(37, mp[-1][-5:].mean()), xytext=(37, mp[-1][:5].mean()),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.6))
    ax.text(34.5, mp[-1].mean(),
            f"{mp[-1][:5].mean() - mp[-1][-5:].mean():.1f} mm$^{{-1}}$\nramp",
            fontsize=10, ha="right", va="center", fontweight="bold")
    ax.set_xlabel("Depth into tissue band (px, surface+11 to +50)")
    ax.set_ylabel(r"Recovered $\mu$ [mm$^{-1}$]")
    ax.set_title("DiffOCT resolves $\\mu$ in depth;\nCF reports one value (dotted)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, ncol=2, title="Intralipid", title_fontsize=9)
    ax.grid(alpha=0.25)
    panel_label(ax, "a")

    # ── (b) both reproduce the A-scan ───────────────────────────────────
    ax = axes[0, 1]
    P, C, L = d["prof_data"], d["prof_cf"], d["prof_dl"]
    for r in P[::4]:
        ax.plot(rel, r, color=C_DATA, lw=0.4, alpha=0.07)
    ax.plot(rel, P.mean(0), color="black", lw=2.8, label="Measured A-scan", zorder=5)
    ax.plot(rel, C.mean(0), color=C_CF, lw=2.3, ls="--", label="CF straight-line fit", zorder=6)
    ax.plot(rel, L.mean(0), color=C_DL, lw=2.3, label="DiffOCT forward model", zorder=6)
    ax.set_xlabel("Depth into tissue band (px)")
    ax.set_ylabel("ln(I / confocal), referenced to band top")
    ax.set_title("Both describe the same A-scan\n(5.0%, all 512 B-scans)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="lower left"); ax.grid(alpha=0.25)
    ins = ax.inset_axes([0.55, 0.60, 0.42, 0.36])
    for arr, col in ((C - P, C_CF), (L - P, C_DL)):
        ins.plot(rel, np.median(arr, 0), color=col, lw=1.8)
        ins.fill_between(rel, np.percentile(arr, 25, 0), np.percentile(arr, 75, 0),
                         color=col, alpha=0.22)
    ins.axhline(0, color="k", lw=0.8, ls=":")
    ins.set_title("residual", fontsize=9); ins.tick_params(labelsize=7.5)
    ins.grid(alpha=0.2)
    panel_label(ax, "b")

    # ── (c) slope reconciliation ────────────────────────────────────────
    ax = axes[0, 2]
    series = [
        ("DiffOCT, depth-mean $\\mu$", d["diffoct_mean"], C_DL, "-", "o"),
        ("DiffOCT, measured CF's way", d["diffoct_cfest"], C_DL, "--", "s"),
        ("CF, published (blocks)", d["cf_published"], C_CF, "-", "^"),
        ("CF, all pixels/all frames", d["cf_allpixels"], C_CF, ":", "v"),
    ]
    xl = np.array([0.7, 5.4])
    for name, y, col, ls, mk in series:
        s, b, r2 = fit(concs, y)
        ax.plot(xl, b + s * xl, color=col, ls=ls, lw=1.9, alpha=0.9)
        ax.plot(concs, y, mk, color=col, ms=8, mec="black", mew=0.7, ls="none",
                label=f"{name}: slope {s:.2f}")
    ax.set_xticks(concs); ax.set_xticklabels([f"{c:.1f}" for c in concs])
    ax.set_xlabel("Intralipid concentration (%)")
    ax.set_ylabel(r"$\mu$ [mm$^{-1}$]")
    ax.set_title("The gap is the estimator, not the data\n"
                 "(measuring DiffOCT CF's way recovers the slope)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
    panel_label(ax, "c")

    # ── (d) R1.4 per-pixel vs per-frame ─────────────────────────────────
    ax = axes[1, 0]
    px = d["r14_pixels"]; fm = d["r14_frame_means"]
    q1, q3 = float(d["r14_pix_q1"]), float(d["r14_pix_q3"])
    fmu, fsd = float(d["r14_frame_mean"]), float(d["r14_frame_sd"])
    bins = np.linspace(0, max(np.percentile(px, 99.5), fmu + 1), 120)
    ax.hist(px, bins=bins, density=True, color="#6baed6", alpha=0.75,
            label=f"Per-pixel $\\mu$  (n = {int(d['r14_pixels'].size*37):,})")
    ax.hist(fm, bins=bins, density=True, color="#d94701", alpha=0.85,
            label=f"Per-B-scan mean  (n = {fm.size})")
    ax.axvspan(q1, q3, color="#6baed6", alpha=0.18)
    ax.annotate(f"per-pixel IQR\n{q1:.2f} to {q3:.2f}", xy=((q1+q3)/2, ax.get_ylim()[1]*0.55),
                fontsize=9.5, ha="center", color="#08519c", fontweight="bold")
    ax.annotate(f"{fmu:.2f} $\\pm$ {fsd:.2f}", xy=(fmu, ax.get_ylim()[1]*0.88),
                xytext=(fmu + 1.6, ax.get_ylim()[1]*0.93), fontsize=10,
                color="#d94701", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#d94701", lw=1.4))
    ax.set_xlabel(r"$\mu$ [mm$^{-1}$]"); ax.set_ylabel("Probability density")
    ax.set_title("Two different spreads, both correct\n(1.0% focal position 1)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5); ax.grid(alpha=0.25)
    panel_label(ax, "d")

    # ── (e) R1.5 outlier anatomy ────────────────────────────────────────
    ax = axes[1, 1]
    H, edges = d["r15_hist"], d["r15_edges"]
    Hn = H / np.maximum(H.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(Hn, aspect="auto", origin="upper", cmap="plasma",
                   extent=[edges[0], edges[-1], rel[-1], rel[0]],
                   vmax=np.percentile(Hn[Hn > 0], 99))
    ax.plot(d["mu_prof"][-1], rel, color="white", lw=2.4)
    ax.plot(d["mu_prof"][-1], rel, color="black", lw=1.0, ls="--")
    ax.axvline(1.0, color="cyan", lw=1.6, ls=":")
    ax.text(1.3, 3, f"$\\mu$ < 1: {100*float(d['r15_frac_below1'][-1]):.1f}% of pixels\n"
                    f"at $\\mu_{{min}}$: {100*float(d['r15_frac_at_min'][-1]):.2f}%",
            color="white", fontsize=9.5, va="top", fontweight="bold")
    ax.set_xlabel(r"$\mu$ [mm$^{-1}$]"); ax.set_ylabel("Depth into band (px)")
    ax.set_title("Outliers are a deep-tissue tail,\nnot clipped or masked pixels (5.0%)",
                 fontsize=12.5, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, label="fraction of pixels at depth")
    panel_label(ax, "e")

    # ── (f) R1.7 speckle response ───────────────────────────────────────
    ax = axes[1, 2]
    rc, rl = d["rms_cf"], d["rms_dl"]
    hi = float(np.percentile(np.concatenate([rc, rl]), 99))
    bins = np.linspace(0, hi, 80)
    ax.hist(rc, bins=bins, color=C_CF, alpha=0.7, label=f"CF  (mean {rc.mean():.3f})")
    ax.hist(rl, bins=bins, color=C_DL, alpha=0.7, label=f"DiffOCT  (mean {rl.mean():.3f})")
    ax.axvline(rc.mean(), color=C_CF, lw=2, ls="--")
    ax.axvline(rl.mean(), color=C_DL, lw=2, ls="--")
    mc, ml = d["mu_cf_aline"], d["mu_dl_aline"]
    ax.text(0.97, 0.55,
            f"Per-A-line $\\mu$ spread\n"
            f"CF  SD = {mc.std():.2f} mm$^{{-1}}$\n"
            f"DiffOCT SD = {ml.std():.2f} mm$^{{-1}}$",
            transform=ax.transAxes, fontsize=10, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.6", alpha=0.95))
    ax.set_xlabel("Per-A-line RMS residual [ln units]")
    ax.set_ylabel("A-line count")
    ax.set_title(f"DiffOCT follows speckle {rc.mean()/rl.mean():.1f}x closer,\n"
                 f"with {mc.std()/ml.std():.1f}x less estimator variance",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper right"); ax.grid(alpha=0.25)
    panel_label(ax, "f")

    # ── (g-i) individual A-scans ────────────────────────────────────────
    ex_d, ex_c, ex_p, meta = d["ex_data"], d["ex_cf"], d["ex_pred"], d["ex_meta"]
    tags = ["CF over-reads (95th pct)", "typical (median)", "CF under-reads (5th pct)"]
    for k in range(3):
        ax = axes[2, k]
        x, mu_cf, mu_dl = meta[k]
        ax.plot(rel, ex_d[k], color=C_DATA, lw=1.3, marker="o", ms=2.6,
                label="measured A-scan")
        ax.plot(rel, ex_c[k], color=C_CF, lw=2.4, ls="--",
                label=f"CF line fit: $\\mu$ = {mu_cf:.2f}")
        ax.plot(rel, ex_p[k], color=C_DL, lw=2.2,
                label=f"DiffOCT model: $\\mu$ = {mu_dl:.2f}")
        ax.set_xlabel("Depth into tissue band (px)")
        if k == 0:
            ax.set_ylabel("ln(I / confocal)")
        ax.set_title(f"A-line {int(x)}, {tags[k]}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.8, loc="upper right")
        ax.grid(alpha=0.25)
        panel_label(ax, "ghi"[k], dx=-0.13 if k == 0 else -0.09)

    fig.suptitle("Curve fitting and DiffOCT estimate different quantities from the same data",
                 fontsize=16.5, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}.png and .pdf")

    # numbers for the caption / Results text
    s_dm = fit(concs, d["diffoct_mean"])[0]
    s_ce = fit(concs, d["diffoct_cfest"])[0]
    s_cf = fit(concs, d["cf_published"])[0]
    s_ap = fit(concs, d["cf_allpixels"])[0]
    with open(f"{OUT}_numbers.txt", "w") as f:
        w = lambda t: (print(t), f.write(t + "\n"))
        w("Numbers for the caption and Results text")
        w("=" * 62)
        w(f"slope DiffOCT depth-mean      : {s_dm:.4f}")
        w(f"slope DiffOCT measured CF-way : {s_ce:.4f}  ({100*(s_ce-s_dm)/(s_cf-s_dm):.0f}% of the gap)")
        w(f"slope CF published            : {s_cf:.4f}")
        w(f"slope CF all pixels/all frames: {s_ap:.4f}")
        for i, c in enumerate(concs):
            w(f"  {c:.1f}%  mu(top)={d['mu_prof'][i][:5].mean():.3f}  "
              f"mu(bottom)={d['mu_prof'][i][-5:].mean():.3f}  "
              f"ramp={d['mu_prof'][i][:5].mean()-d['mu_prof'][i][-5:].mean():+.3f}  "
              f"depth-mean={d['depth_mean'][i]:.3f}  slope-implied={d['slope_implied'][i]:.3f}")
        w(f"R1.4  per-pixel IQR {float(d['r14_pix_q1']):.2f} to {float(d['r14_pix_q3']):.2f}, "
          f"mean {float(d['r14_pix_mean']):.2f}")
        w(f"R1.4  per-B-scan mean {float(d['r14_frame_mean']):.3f} +/- "
          f"{float(d['r14_frame_sd']):.3f} (n={d['r14_frame_means'].size})")
        w(f"R1.5  frac mu<1 by conc: {np.round(100*d['r15_frac_below1'],2)} %")
        w(f"R1.5  frac at mu_min   : {np.round(100*d['r15_frac_at_min'],3)} %")
        w(f"R1.7  RMS residual CF {rc.mean():.4f} vs DiffOCT {rl.mean():.4f} "
          f"({rc.mean()/rl.mean():.2f}x)")
        w(f"R1.7  per-A-line mu SD CF {mc.std():.3f} vs DiffOCT {ml.std():.3f}")
        w(f"A-scan examples from B-scan {int(d['ex_frame'])}:")
        for k, t in enumerate(["95th pct", "median", "5th pct"]):
            x, mcf, mdl = d["ex_meta"][k]
            w(f"   A-line {int(x):3d} ({t:9s}): CF={mcf:6.2f}  DiffOCT={mdl:6.2f}  "
              f"diff={mcf-mdl:+6.2f}")
    print(f"Saved {OUT}_numbers.txt")


if __name__ == "__main__":
    main()
