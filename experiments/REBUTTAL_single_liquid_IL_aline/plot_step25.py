"""Validate the TV-on rerun and render CF-window MAPE in the Figure S1 style."""
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, StrMethodFormatter, AutoMinorLocator
import numpy as np
from scipy.interpolate import PchipInterpolator

import summarize as original
from rerun_step25 import BASE, BETA, ROOT, RUN, backend
KEY = str(BETA)

STAGE = ROOT / "figures/data/figureS1_optimizer_scaling/liquid"
NAME = "rebuttal_single_liquid_IL_aline_scaling"
METHODS = [("Random Search", "tab:orange", "^"), ("CMA-ES", "tab:red", "D"),
           ("Adam", "tab:blue", "o"), ("L-BFGS-B", "tab:green", "s")]
RC = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
      "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
      "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
      "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
      "axes.linewidth": .8, "lines.linewidth": 1.5, "pdf.fonttype": 42}


def collect():
    b = backend()
    config, data = b.load_data()
    assert config["sizes"] == list(range(25, 476, 25)) and config["tv_weights"] == [BETA]
    original.DATA, original.RESULTS = RUN / "data", RUN / "results"
    rows, aggregate, provenance = original.collect(config, data)
    assert len(rows) == 608
    assert provenance["runner_sha256"] == config["base_runner_sha256"]
    mask = data["cf_comparison"].astype(bool)
    assert mask.sum() == 100 and data["valid"][mask].all()
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(100))
    reference = float(data["cf_mu"])
    assert reference > 0
    for row in rows:
        with np.load(RUN / "results" / f"tv{BETA:g}_n{row['n']:04d}_seed{row['seed']}.npz") as f:
            key = row["method"].lower().replace(" ", "_").replace("-", "_")
            mu = f[f"{key}_mu"][mask].astype(float)
        # Mean of pixelwise absolute percentage errors, not percentage error of mean mu.
        mape = 100 * np.mean(np.abs(mu - reference) / reference)
        assert mape + 1e-8 >= row["cf_relative_disagreement_percent"]
        row["cf_pixel_mape_percent"] = float(mape)
    for n in config["sizes"]:
        for method in config["methods"]:
            vals = [r["cf_pixel_mape_percent"] for r in rows if r["n"] == n and r["method"] == method]
            aggregate[KEY][str(n)][method]["cf_pixel_mape_percent"] = {
                "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    metadata = dict(
        definition="100 * mean over original CF-window pixels of abs(mu_recovered - mu_CF) / mu_CF at each run's best-MSE iterate; compute separately per seed, then report mean and sample SD across eight seeds",
        cf_window_pixels=int(mask.sum()), cf_mu_mm_inverse=reference,
        fit_objective="1000 * MSE in corrected log-intensity, using all 475 valid native observations, plus axial TV with beta=1",
        plotted_error="CF-referenced attenuation MAPE within the original 100-pixel CF window",
        reference_caveat="CF is not per-pixel ground truth; the fixed gain also contains CF information",
        timing="Measured total solver wall time divided by actual objective-evaluation count",
        curves="PCHIP interpolation of log10(mean) versus log10(N); descriptive guides through all measured means",
        sizes=config["sizes"], seeds=config["seeds"], runs=len(rows),
    )
    return config, rows, aggregate, provenance, metadata


def style_legend(legend):
    frame = legend.get_frame()
    frame.set_boxstyle("round,pad=0.20")
    frame.set_facecolor("black")
    frame.set_edgecolor("none")
    frame.set_alpha(.25)
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_color("black")
        text.set_fontweight("bold")


def render(config, aggregate, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig, left = plt.subplots(figsize=(7.0, 4.6))
        right = left.twinx()
        left.spines[["top", "right"]].set_visible(False)
        for side in ["left", "bottom"]:
            left.spines[side].set_linewidth(.9)
        right.spines["top"].set_visible(False)
        right.spines["right"].set_visible(True)
        right.spines["right"].set_linewidth(.9)
        left.tick_params(axis="both", labelsize=13, width=.8, length=3.5)
        right.tick_params(axis="y", labelsize=13, width=.8, length=3.5)
        xs = np.array(config["sizes"], dtype=float)
        dense = np.geomspace(xs.min(), xs.max(), 800)
        for method, color, marker in METHODS:
            for ax, metric, line, width, filled in [
                (left, "cf_pixel_mape_percent", "-", 2.2, True),
                (right, "ms_per_eval", "--", 1.5, False)]:
                vals = [aggregate[KEY][str(int(n))][method][metric] for n in xs]
                means = np.array([v["mean"] for v in vals])
                sd = np.array([v["sd"] for v in vals])
                assert np.all(means > 0)
                smooth = 10**PchipInterpolator(np.log10(xs), np.log10(means))(np.log10(dense))
                ax.plot(dense, smooth, color=color, ls=line, lw=width,
                        alpha=.95 if filled else .9025, zorder=3 if filled else 2)
                ax.errorbar(xs, means, yerr=sd, color=color, marker=marker,
                            markersize=6 if filled else 5, ls="", capsize=3,
                            elinewidth=1, alpha=.5, markerfacecolor=color if filled else "none",
                            markeredgecolor="white" if filled else color,
                            markeredgewidth=.6 if filled else 1.2, zorder=4 if filled else 3)
        left.set_xscale("log")
        left.set_xlabel(r"Problem size $K$ (attenuation parameters)", fontsize=13)
        left.set_ylabel("MAPE (%)", fontsize=13)
        right.set_ylabel("Time per evaluation (ms)", fontsize=13)
        ticks = [25, 50, 75, 100, 200, 300, 475]
        left.set_xticks(ticks, labels=[str(n) for n in ticks])
        left.tick_params(axis="x", which="minor", bottom=False, top=False)
        left.set_xlim(21.25, 570)
        # Linear metric axes; leave space above the data for the legends.
        for ax, metric, tick_step, headroom in [
            (left, "cf_pixel_mape_percent", 20., 1.15),
            (right, "ms_per_eval", .5, 1.5)]:
            peak = max(aggregate[KEY][str(int(n))][method][metric]["mean"]
                       + aggregate[KEY][str(int(n))][method][metric]["sd"]
                       for n in xs for method, _, _ in METHODS)
            upper = tick_step * np.ceil(peak * headroom / tick_step)
            ax.set_yscale("linear")
            ax.set_ylim(0, upper)
            ax.yaxis.set_major_locator(MultipleLocator(tick_step))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis="y", which="minor", length=2, labelsize=0)
        left.grid(True, which="major", axis="both", alpha=.40, ls=":", lw=.7, zorder=0)
        left.grid(True, which="minor", axis="y", alpha=.18, ls=":", lw=.4, zorder=0)
        right.grid(False)
        handles = [Line2D([], [], color=c, marker=m, ls="-", lw=2.2, markersize=6,
                          markerfacecolor=c, markeredgecolor="white", markeredgewidth=.7,
                          label="L-BFGS" if name == "L-BFGS-B" else name) for name,c,m in METHODS]
        leg = left.legend(handles=handles, loc="upper left", bbox_to_anchor=(.020,.720,.295,.245),
                          bbox_transform=left.transAxes, mode="expand", borderaxespad=0,
                          fontsize=9, frameon=True, fancybox=True, title="Optimizer",
                          title_fontsize=9, handlelength=1.8)
        left.add_artist(leg)
        style_legend(leg)
        handles = [Line2D([],[], color="#333333", lw=2.2, ls="-", label="MAPE"),
                   Line2D([],[], color="#333333", lw=1.5, ls="--", label="ms/eval")]
        # Place the metric legend beside the optimizer legend to keep the
        # approximately 25% CF-reference curves visible at the smaller sizes.
        leg = left.legend(handles=handles, loc="upper left", bbox_to_anchor=(.355,.840,.250,.125),
                          bbox_transform=left.transAxes, mode="expand", borderaxespad=0,
                          fontsize=9, frameon=True, fancybox=True, handlelength=2.2,
                          title="N = 8 runs", title_fontsize=9)
        style_legend(leg)
        fig.tight_layout()
        for suffix, kwargs in [("pdf",{}),("jpeg",{"dpi":600}),("png",{"dpi":160})]:
            fig.savefig(directory/f"{NAME}.{suffix}", bbox_inches="tight", pad_inches=.1, **kwargs)
        plt.close(fig)


def report(config, aggregate):
    lines = ["# Liquid Intralipid A-line: axial TV on, step-25 sweep", "",
             "608 fresh solver runs: N = 25, 50, 75, ..., 475; four solvers; eight seeds.", "",
             "The signal, gain, bounds, initialization rule, and 20,000-evaluation cap match the prior experiment.", "",
             "All solvers minimize corrected-intensity MSE plus axial TV (beta=1). The plotted MAPE is the mean pixelwise absolute percentage difference of recovered attenuation from CF within the original 100-pixel CF window. It is not the percentage error of mean attenuation. CF is a reference estimate and contributes to the fixed gain; this is not independent ground-truth accuracy.", "",
             "| N | Solver | CF MAPE (%), mean ± SD | MSE (dB²), mean | ms/eval, mean | Total seconds, mean |", "|---:|---|---:|---:|---:|---:|"]
    for n in [25,475]:
        for m,_,_ in METHODS:
            r = aggregate[KEY][str(n)][m]
            v = r["cf_pixel_mape_percent"]
            lines.append(f"| {n} | {m} | {v['mean']:.3f} ± {v['sd']:.3f} | {r['mse_db2']['mean']:.6g} | {r['ms_per_eval']['mean']:.4f} | {r['wall_sec']['mean']:.3f} |")
    lines += ["", "Curves are PCHIP guides through the measured means in log coordinates. Error bars show one sample SD across seeds. The figure uses the fonts, colors, markers, line styles, dual axes, and gray legends of S1.a.", "",
              "Compare solvers at the same N. This sweep retains existing signal correlations; it does not isolate the effects of intrinsic correlation or despeckling and has no axial or lateral TV term.", ""]
    (RUN/"RESULTS.md").write_text("\n".join(lines))


def main():
    config, rows, aggregate, provenance, metadata = collect()
    b = backend()
    b.write_json(RUN/"summary.json", dict(config=config, provenance=provenance, aggregate=aggregate,
                                          runs=len(rows), verified_saved_objectives=True, plot_metadata=metadata))
    b.write_json(RUN/"plot_metadata.json", metadata)
    with (RUN/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render(config, aggregate, RUN/"figures")
    report(config, aggregate)
    STAGE.mkdir(parents=True, exist_ok=True)
    for directory in [STAGE, BASE/"figures"]:
        for suffix in ["pdf","jpeg","png"]:
            destination = directory/f"{NAME}.{suffix}"
            archive = directory/f"rebuttal_single_liquid_IL_aline_tv_comparison.{suffix}"
            if destination.exists() and not archive.exists():
                shutil.copy2(destination, archive)
            shutil.copy2(RUN/"figures"/destination.name, destination)
    for name in ["summary.json","metrics.csv","plot_metadata.json","RESULTS.md"]:
        shutil.copy2(RUN/name, STAGE/name)
        shutil.copy2(RUN/name, BASE/name)
    shutil.copy2(RUN/"data/config.json", STAGE/"config.json")
    print(f"Validated {len(rows)} runs and rendered {STAGE/NAME}", flush=True)


if __name__ == "__main__":
    main()
