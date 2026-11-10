"""Validate completed benchmark outputs and generate figures and a factual report."""
import csv
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np

from benchmark import HERE, ROOT, DATA, RESULTS, METHODS, load_data, sha256, write_json

OUT = HERE / "figures"
STAGE = ROOT / "figures/data/figureS1_optimizer_scaling/liquid"
STYLE = {
    "Random Search": ("tab:orange", "^"), "CMA-ES": ("tab:red", "D"),
    "Adam": ("tab:blue", "o"), "L-BFGS-B": ("tab:green", "s"),
}
ORDER = list(STYLE)
METRICS = ["best_objective", "rmse_db", "mse_db2", "tv_native", "wall_sec", "ms_per_eval",
           "fwd_evals", "cf_window_mean_mu", "cf_relative_disagreement_percent"]


def collect(config, data):
    rows, aggregate = [], {}
    scripts, devices = set(), set()
    for beta in config["tv_weights"]:
        aggregate[str(beta)] = {}
        for n in config["sizes"]:
            records = {m: [] for m in METHODS}
            for seed in config["seeds"]:
                stem = f"tv{beta:g}_n{n:04d}_seed{seed}"
                path = RESULTS / f"{stem}.json"
                r = json.loads(path.read_text())
                assert (r["n"], r["beta"], r["seed"]) == (n, beta, seed)
                assert r["provenance"]["data_sha256"] == config["aline_sha256"]
                assert r["provenance"]["config_sha256"] == sha256(DATA / "config.json")
                assert r["provenance"]["budget"] == config["max_objective_evaluations"]
                arrays = RESULTS / f"{stem}.npz"
                assert sha256(arrays) == r["arrays_sha256"]
                scripts.add(r["provenance"]["script_sha256"])
                devices.add(r["provenance"]["gpu"])
                with np.load(arrays) as f:
                    group = f["group_index"]
                    assert len(np.unique(group)) == n
                    for method in METHODS:
                        m = r["methods"][method]
                        assert m["fwd_evals"] <= config["max_objective_evaluations"]
                        key = method.lower().replace(" ", "_").replace("-", "_")
                        mu, pred, hist = f[f"{key}_mu"], f[f"{key}_pred_db"], f[f"{key}_history"]
                        np.testing.assert_array_equal(mu, f[f"{key}_theta"][group])
                        mse = np.mean((pred[data["valid"]].astype(float) - data["target_db"][data["valid"]]) ** 2)
                        tv = np.mean(np.abs(np.diff(mu.astype(float))))
                        np.testing.assert_allclose(mse, m["mse_db2"], rtol=2e-5, atol=1e-8)
                        np.testing.assert_allclose(mse + beta * tv, m["best_objective"], rtol=2e-5, atol=1e-8)
                        assert hist[-1, 0] == m["fwd_evals"]
                        assert np.all(np.diff(hist[:, 2]) <= 1e-8)
                        records[method].append(m)
                        rows.append(dict(beta=beta, n=n, seed=seed, method=method,
                                         **{k: m[k] for k in METRICS}, termination=m["termination"]))
            aggregate[str(beta)][str(n)] = {
                method: {metric: dict(mean=float(np.mean([r[metric] for r in values])),
                                      sd=float(np.std([r[metric] for r in values], ddof=1)))
                         for metric in METRICS}
                for method, values in records.items()}
    assert len(scripts) == 1, "Mixed runner versions"
    assert len(devices) == 1 and None not in devices, "Mixed GPU models or CPU timing"
    return rows, aggregate, dict(gpu=next(iter(devices)), runner_sha256=next(iter(scripts)))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.08)
    fig.savefig(OUT / f"{name}.jpeg", dpi=600, bbox_inches="tight", pad_inches=.08,
                pil_kwargs={"quality": 95})
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def series(ax, config, aggregate, beta, metric, linestyle="-"):
    sizes = config["sizes"]
    for method in ORDER:
        color, marker = STYLE[method]
        records = [aggregate[str(beta)][str(n)][method][metric] for n in sizes]
        mean = np.array([r["mean"] for r in records])
        sd = np.array([r["sd"] for r in records])
        ax.errorbar(sizes, mean, yerr=sd, color=color, marker=marker,
                    ls=linestyle, lw=1.5, ms=3.4 if method == "L-BFGS-B" else 5, capsize=2,
                    markerfacecolor=color if linestyle == "-" else "white",
                    markeredgecolor=color, elinewidth=.8, label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(21, config["sizes"][-1] * 1.14)
    ticks = [25, 50, 100, 200, config["sizes"][-1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(alpha=.25, ls=":", which="major")
    ax.spines[["top", "right"]].set_visible(False)


def main_figure(config, aggregate):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), sharex=True, sharey="row")
    for col, beta in enumerate(config["tv_weights"]):
        series(axes[0, col], config, aggregate, beta, "best_objective")
        series(axes[1, col], config, aggregate, beta, "ms_per_eval", "--")
        axes[0, col].set_title(f"({'ab'[col]}) Axial TV " + ("off" if beta == 0 else "on"),
                               weight="bold", loc="left")
        axes[1, col].set_title(f"({'cd'[col]}) Time per evaluation", weight="bold", loc="left")
        axes[0, col].set_ylabel("Best objective (MSE + β TV)")
        axes[1, col].set_ylabel("Time per evaluation (ms)")
        axes[1, col].set_xlabel(r"$N$ (attenuation parameters)")
    for row, metric in enumerate(["best_objective", "ms_per_eval"]):
        values = [aggregate[str(b)][str(n)][m][metric]
                  for b in config["tv_weights"] for n in config["sizes"] for m in ORDER]
        lower = min(r["mean"] - r["sd"] for r in values)
        upper = max(r["mean"] + r["sd"] for r in values)
        assert lower > 0, "Log axis requires positive mean-minus-SD limits"
        axes[row, 0].set_ylim(lower * .8, upper * 1.25)
    handles = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], lw=1.5, label=m) for m in ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.51, 1.0))
    fig.text(.5, .006, "1% liquid Intralipid · one fixed A-line · 475 valid native observations · 8 seeds per point",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .945))
    save(fig, "rebuttal_single_liquid_IL_aline_tv_comparison")


def diagnostic_figure(config, aggregate):
    fig, axes = plt.subplots(3, 2, figsize=(9, 8.7), sharex=True, sharey="row")
    for col, beta in enumerate(config["tv_weights"]):
        for row, (metric, label) in enumerate([
            ("rmse_db", "Intensity RMSE (dB)"), ("wall_sec", "Total runtime (s)"),
            ("cf_relative_disagreement_percent", "CF-relative mean μ difference (%)")]):
            series(axes[row, col], config, aggregate, beta, metric)
            axes[row, col].set_ylabel(label)
        axes[0, col].set_title("Axial TV " + ("off" if beta == 0 else "on"), weight="bold")
        axes[-1, col].set_xlabel(r"$N$ (attenuation parameters)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Diagnostics: CF is a shared reference, not per-pixel ground truth", fontsize=11)
    fig.tight_layout()
    save(fig, "diagnostics")


def convergence_figure(config):
    n = config["sizes"][-1]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6))
    for row, beta in enumerate(config["tv_weights"]):
        with np.load(RESULTS / f"tv{beta:g}_n{n:04d}_seed0.npz") as f:
            for method in ORDER:
                key = method.lower().replace(" ", "_").replace("-", "_")
                hist = f[f"{key}_history"]
                for col in range(2):
                    axes[row, col].loglog(hist[:, col], hist[:, 2], color=STYLE[method][0],
                                          label=method, lw=1.5)
        for col in range(2):
            axes[row, col].grid(alpha=.25, ls=":")
            axes[row, col].set_ylabel("Best objective so far")
            axes[row, col].set_title("Axial TV " + ("off" if beta == 0 else "on"))
        axes[row, 0].set_xlabel("Objective evaluations")
        axes[row, 1].set_xlabel("Elapsed time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Convergence at N = {n}; prespecified seed 0", fontsize=11)
    fig.tight_layout()
    save(fig, "convergence_native")


def profile_figure(config, data):
    n = config["sizes"][-1]
    depth = np.arange(len(data["target_db"])) * float(data["dz_mm"])
    cf = data["cf_comparison"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey="row")
    for col, beta in enumerate(config["tv_weights"]):
        measured = np.where(data["valid"], data["target_db"], np.nan)
        axes[0, col].plot(depth, measured, color="black", lw=.8, alpha=.65, label="Measured")
        with np.load(RESULTS / f"tv{beta:g}_n{n:04d}_seed0.npz") as f:
            for method in ORDER:
                key = method.lower().replace(" ", "_").replace("-", "_")
                for row, suffix in [(0, "pred_db"), (1, "mu")]:
                    axes[row, col].plot(depth, f[f"{key}_{suffix}"], color=STYLE[method][0], lw=1., label=method)
        axes[1, col].plot(depth[cf], np.full(cf.sum(), float(data["cf_mu"])), color="black", ls="--",
                          lw=1.3, label="CF reference (fit window)")
        for row in range(2):
            axes[row, col].axvspan(depth[cf].min(), depth[cf].max(), color="0.7", alpha=.12)
            axes[row, col].grid(alpha=.2, ls=":")
        axes[0, col].set_title("Axial TV " + ("off" if beta == 0 else "on"), weight="bold")
        axes[0, col].set_ylabel("Corrected intensity (dB)")
        axes[1, col].set_ylabel(r"Attenuation $\mu$ (mm$^{-1}$)")
        axes[1, col].set_xlabel("Depth from CF-window start (mm)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(handles=[axes[1, 0].lines[-1]], fontsize=7)
    fig.suptitle(f"Native-grid fits and recovered profiles, N = {n}, seed 0", fontsize=11)
    fig.tight_layout()
    save(fig, "profiles_native")


def report(config, aggregate, provenance):
    lines = ["# Real liquid A-line solver comparison", "",
             "All requested runs completed and saved-state objective checks passed.", "",
             f"Data: {config['selection']}. {config['native_depth_samples']} native depth samples, "
             f"{config['valid_observations']} valid observations; eight numerical seeds per setting.", "",
             f"Timing hardware: {provenance['gpu']}. All solvers had a maximum of 20,000 "
             "objective evaluations, including the initial point. Actual counts and stopping reasons are in metrics.csv.", ""]
    for beta in config["tv_weights"]:
        lines += [f"## Axial TV {'off' if beta == 0 else 'on'}", "",
                  "| N | Solver | Objective, mean ± SD | RMSE (dB) | ms/evaluation | Total seconds | CF mean-μ difference (%) |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for n in [config["sizes"][0], config["sizes"][-1]]:
            for method in ORDER:
                r = aggregate[str(beta)][str(n)][method]
                lines.append(f"| {n} | {method} | {r['best_objective']['mean']:.5g} ± {r['best_objective']['sd']:.3g} | "
                             f"{r['rmse_db']['mean']:.4g} | {r['ms_per_eval']['mean']:.4g} | {r['wall_sec']['mean']:.4g} | "
                             f"{r['cf_relative_disagreement_percent']['mean']:.4g} |")
        lines.append("")
    lines += ["## What this experiment can establish", "",
              "Compare solver objective values within the same N and TV setting. Low objective "
              "means a better fit to that optimization problem, not necessarily more accurate physical attenuation.", "",
              "The target retains the acquisition's existing correlations and despeckling. The TV comparison "
              "tests adding axial regularization. There is no raw-versus-despeckled control here, and one A-line "
              "does not test lateral TV coupling or establish a result across all real acquisitions.", "",
              "CF-relative mean-μ differences are secondary and conditional on the CF-derived fixed gain. "
              "They are not ground-truth MAPE. Error bars describe random initialization/solver variation on "
              "one fixed signal, not independent biological replicates.", "",
              "The largest real-data problem has 476 attenuation parameters. This is a narrower dimension "
              "range than the original full-B-scan experiment; conclusions should be restricted accordingly.", ""]
    (HERE / "RESULTS_tv_comparison.md").write_text("\n".join(lines))


def main():
    config, data = load_data()
    rows, aggregate, provenance = collect(config, data)
    OUT.mkdir(exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "axes.linewidth": .8})
    with (HERE / "metrics_tv_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(HERE / "summary_tv_comparison.json", dict(config=config, provenance=provenance, aggregate=aggregate,
                                           runs=len(rows), verified_saved_objectives=True))
    main_figure(config, aggregate)
    diagnostic_figure(config, aggregate)
    convergence_figure(config)
    profile_figure(config, data)
    report(config, aggregate, provenance)
    for name in ["rebuttal_single_liquid_IL_aline_tv_comparison", "diagnostics", "convergence_native", "profiles_native"]:
        for suffix in ["pdf", "jpeg", "png"]:
            path = OUT / f"{name}.{suffix}"
            shutil.copy2(path, STAGE / path.name)
    for path in [HERE / "summary_tv_comparison.json", HERE / "metrics_tv_comparison.csv", HERE / "RESULTS_tv_comparison.md"]:
        shutil.copy2(path, STAGE / path.name)
    print(f"Validated {len(rows)} runs; figures saved to {OUT} and {STAGE}")


if __name__ == "__main__":
    main()
