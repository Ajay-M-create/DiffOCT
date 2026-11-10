"""Paired frame summaries and N=8 boxplots for the fixed-budget phantom pilot."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pilot import sha256, write_json

LABELS = {"cf": "CF", "full_depth": "DiffOCT\nfull depth", "cf_window": "DiffOCT\nCF window"}
COLORS = {"cf": "#444444", "full_depth": "#0072B2", "cf_window": "#D55E00"}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run, domain="both"):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    frames = provenance["frames"]
    assert config["seed_repeats"] == 1 and "stop_after" not in config
    assert config["summary_state"].startswith("Final iteration")
    if sha256(run / "prepared.npz") != provenance["prepared_sha256"]:
        raise RuntimeError("Prepared input changed")
    with np.load(run / "prepared.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    mask = data["comparison_mask"]
    counts = mask.sum(axis=1)
    cf_mu = data["cf_mu"].astype(np.float64)
    paired = (counts > 0) & np.isfinite(cf_mu) & (cf_mu > 0) & ~data["gain_fallback"]
    assert len(frames) == paired.shape[0] and paired.any(axis=1).all()
    cf_means = np.array([cf_mu[i, paired[i]].mean() for i in range(len(frames))])
    domains = config["fitting_domains"] if domain == "both" else [domain]
    rows, results = [], {}
    for name in domains:
        folder = run / "results" / name
        completion = json.loads((folder / "seed_0_complete.json").read_text())
        if sha256(folder / "seed_0.npz") != completion["result_sha256"]:
            raise RuntimeError("Optimization result changed")
        with np.load(folder / "seed_0.npz") as archive:
            result = {key: archive[key] for key in archive.files}
        start, stop = int(result["crop_start"]), int(result["crop_stop"])
        mu = np.zeros_like(data["target_db"], dtype=np.float64)
        mu[:, start:stop] = result["mu_final"]
        prefix = "full_depth_" if name == "full_depth" else ""
        transport, loss = data[prefix + "transport_mask"], data[prefix + "loss_mask"]
        active = np.where(transport, mu, 0.0)
        tau = np.cumsum(active * config["dz_mm"], axis=1) - active * config["dz_mm"]
        np.testing.assert_array_equal(result["gain_db"], data["gain_db"].astype(np.float32))
        prediction = 10 * np.log10(np.where(transport, mu, 1.0)) - 20 / np.log(10) * tau + result["gain_db"][:, None, :]
        residual = prediction - data["target_db"]
        aline = np.divide(np.sum(mu * mask, axis=1), counts, out=np.full(counts.shape, np.nan), where=counts > 0)
        frame_rows, aline_rows = [], []
        for i, frame in enumerate(frames):
            cf = cf_mu[i, paired[i]]
            diff = aline[i, paired[i]] - cf
            roi = mask[i] & paired[i][None, :]
            mse = float(np.mean(residual[i][loss[i]] ** 2))
            np.testing.assert_allclose(mse, float(result["mse_final"][i]), rtol=2e-4, atol=2e-5)
            row = dict(domain=name, matlab_bscan=frame["matlab_bscan"], python_bscan=frame["python_bscan"],
                       cf_mean_mm_inv=float(cf.mean()), diffoct_mean_mm_inv=float(aline[i, paired[i]].mean()),
                       bias_mm_inv=float(diff.mean()), relative_bias_percent=float(100 * diff.mean() / cf.mean()),
                       aline_mae_mm_inv=float(np.abs(diff).mean()), aline_rmse_mm_inv=float(np.sqrt(np.mean(diff ** 2))),
                       paired_alines=int(paired[i].sum()), comparison_pixels=int(roi.sum()), fit_pixels=int(loss[i].sum()),
                       excluded_comparison_pixels=int(data["excluded_mask"][i].sum()), fit_mse_db2=mse,
                       comparison_mse_db2=float(np.mean(residual[i][roi] ** 2)), reported_iteration=config["iterations"],
                       best_iteration_diagnostic=int(result["iteration_best"][i]),
                       gain_median_db=float(np.median(result["gain_db"][i])), gain_min_db=float(result["gain_db"][i].min()),
                       gain_max_db=float(result["gain_db"][i].max()), gain_fallback_alines=int(data["gain_fallback"][i].sum()),
                       objective=float(result["objective_final"][i]),
                       comparison_lower_bound_fraction=float(np.mean(mu[i][roi] <= config["mu_bounds"][0] * 1.001)),
                       comparison_upper_bound_fraction=float(np.mean(mu[i][roi] >= config["mu_bounds"][1] * .999)))
            frame_rows.append(row)
            for x in range(counts.shape[1]):
                aline_rows.append(dict(matlab_bscan=frame["matlab_bscan"], python_bscan=frame["python_bscan"],
                                      x_python=x, x_matlab=x + 1, cf_mu_mm_inv=cf_mu[i, x],
                                      cf_filtered_mu_mm_inv=data["cf_mu_filtered"][i, x], diffoct_mean_mu_mm_inv=aline[i, x],
                                      valid_comparison_pixels=int(counts[i, x]), paired=bool(paired[i, x]),
                                      fixed_gain_db=float(result["gain_db"][i, x]), gain_fallback=bool(data["gain_fallback"][i, x])))
        write_json(folder / "summary.json", frame_rows)
        write_csv(folder / "frame_summary.csv", frame_rows)
        write_csv(folder / "alines.csv", aline_rows)
        history = json.loads((folder / "seed_0_history.json").read_text())
        assert history[-1]["iteration"] == config["iterations"]
        rows.extend(frame_rows)
        results[name] = dict(means=np.array([r["diffoct_mean_mm_inv"] for r in frame_rows]), history=history)
    if domain != "both":
        return rows
    values = {"cf": cf_means, **{name: results[name]["means"] for name in domains}}
    methods = []
    for name, means in values.items():
        methods.append(dict(method=name, n_frames=len(means), mean_mm_inv=float(means.mean()), median_mm_inv=float(np.median(means)),
                            sd_across_frames_mm_inv=float(means.std(ddof=1)) if len(means) > 1 else None, q1_mm_inv=float(np.quantile(means, .25)),
                            q3_mm_inv=float(np.quantile(means, .75)), min_mm_inv=float(means.min()), max_mm_inv=float(means.max()),
                            mean_paired_bias_mm_inv=float(np.mean(means - cf_means)),
                            relative_mean_bias_percent=float(100 * np.mean(means - cf_means) / cf_means.mean())))
    frame_rows = []
    for i, frame in enumerate(frames):
        frame_rows.append(dict(matlab_bscan=frame["matlab_bscan"], python_bscan=frame["python_bscan"],
                               paired_alines=int(paired[i].sum()), comparison_pixels=int((mask[i] & paired[i][None, :]).sum()),
                               **{name + "_mean_mm_inv": float(means[i]) for name, means in values.items()}))
    write_csv(run / "frame_means.csv", frame_rows)
    write_csv(run / "summary.csv", methods)
    write_csv(run / "fit_diagnostics.csv", rows)
    write_json(run / "summary.json", dict(frames=frames, methods=methods, frame_means=frame_rows, comparison=rows,
               reported_iteration=config["iterations"], gain_disclosure=config["gain_disclosure"],
               aggregation="Valid CF-window pixels -> A-line mean -> equal-weight paired A-line mean per frame -> equally weighted frame means; fallback-gain A-lines excluded",
               seed_repeats=1, raw_paired_estimates=True, early_stopping=False))
    make_figures(run, config, frames, values, results)
    table_rows = []
    for r in methods:
        sd = f"{r['sd_across_frames_mm_inv']:.6f}" if r['sd_across_frames_mm_inv'] is not None else "—"
        table_rows.append(f"| {r['method']} | {r['mean_mm_inv']:.6f} | {sd} | {r['mean_paired_bias_mm_inv']:+.6f} | {r['relative_mean_bias_percent']:+.2f}% |")
    table = "\n".join(table_rows)
    frame_ids = ", ".join(str(f["matlab_bscan"]) for f in frames)
    (run / "README.md").write_text(f"""# {config['concentration_percent']:g}% {config['acquisition_labels'][0]}: {len(frames)} frames, fixed CF-derived gain, {config['iterations']:,} iterations

MATLAB B-scans: **{frame_ids}**. One initialization per frame, matched between
the two fitting domains. Every optimization completes exactly {config['iterations']:,}
steps, with a cosine schedule of the same length. Report the final state;
there is no early stopping or selection of an earlier checkpoint for comparison.

## Distribution of {len(frames)} paired frame means

| Method | Mean μ (mm⁻¹) | SD across frames | Mean paired bias vs CF | Relative mean bias |
| --- | ---: | ---: | ---: | ---: |
{table}

Each plotted observation is one frame mean. DiffOCT first averages valid pixels
within each A-line's supplied 100-pixel CF window, then averages A-lines with
positive finite CF μ and a valid CF-derived gain. CF uses exactly those same
A-lines. Each frame has equal weight. No outlier filtering is applied.
These are B-scans from one {config['acquisition_labels'][0]} acquisition.

## Fitting and calibration

Prepared fitting domains: {', '.join(domains)}. All report μ in the same valid
CF window. Noise is subtracted in
linear measured intensity: `(I-N)/(S*H)`, then converted to dB. TIFFs already
contain despeckled log intensity; no further smoothing is applied.
Individual invalid corrected pixels are excluded. Missing observations retain
latent optical-path μ inside the modeled domain. α is fixed to 1.
Gain variant: **{config.get('gain_variant', 'per_aline_cf')}**.
{config['gain']}

The underlying per-A-line conversion is:

`{config['gain_formula']}`

When preparing CF-derived per-A-line gains, invalid entries receive the frame's
median valid gain. Scalar variants subsequently replace every gain with the
acquisition scalar. All variants exclude invalid CF-conversion A-lines from
the paired comparison so the eligible A-lines stay identical.
The saved `gain_fallback` flags record invalid CF conversions. Scalar variants
retain these exclusions to compare the same A-lines across gain variants.
Scalar calibration metadata: {config.get("scalar_calibration", "Not applicable")}.

## Interpretation

CF and DiffOCT share a CF-derived gain. This comparison is conditional on that
gain: conversion includes μ_CF and therefore agreement is not independent
validation of attenuation accuracy. The manuscript’s original method re-profiles gain every iteration. This run
uses the fixed-gain model described in the handoff.

## Files

- `boxplots.png` / `.pdf`: paired frame means per method, all points shown.
- `convergence.png` / `.pdf`: optimization histories for every selected frame.
- `frame_means.csv`: the observations used in each boxplot.
- `summary.csv` / `.json`: method summaries and frame-level diagnostics.
- `fit_diagnostics.csv`: final-state residuals, masks, gains and bound checks.
- `results/*/alines.csv`: A-line values and explicit pairing flags.
- `results/*/seed_0.npz`: final μ/gain/MSE and best-state diagnostics.
- `config.json`, `provenance.json`, `code/`: reproducible settings and snapshots.
""")
    write_json(run / "status.json", dict(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                                        optimizations=len(rows), frames=len(frames), fitting_domains=domains,
                                        iterations=config["iterations"], early_stopping=False))
    print(json.dumps(methods, indent=2), flush=True)
    return rows


def make_figures(run, config, frames, values, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    methods = list(values)
    frame_colors = plt.get_cmap("tab10")(np.arange(len(frames)))
    fig, ax = plt.subplots(figsize=(9.5, 6), constrained_layout=True)
    box = ax.boxplot(list(values.values()), tick_labels=[LABELS[n] for n in methods], widths=.42,
                     patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 1.6})
    for patch, name in zip(box["boxes"], methods):
        patch.set_facecolor(COLORS[name]); patch.set_alpha(.25)
    offsets = np.linspace(-.13, .13, len(frames))
    for i, frame in enumerate(frames):
        positions = np.arange(1, len(methods) + 1) + offsets[i]
        means = [values[name][i] for name in methods]
        ax.plot(positions, means, color=frame_colors[i], alpha=.45, lw=.8, zorder=2)
        ax.scatter(positions, means, color=frame_colors[i], s=40, edgecolor="white", lw=.4, zorder=3)
    handles = [Line2D([], [], marker="o", color=frame_colors[i], lw=.8, label=str(f["matlab_bscan"])) for i, f in enumerate(frames)]
    ax.legend(handles=handles, title="B-scan (MATLAB)", loc="center left", bbox_to_anchor=(1.01, .5), fontsize=9)
    ax.set(ylabel="Frame mean μ in the common CF window (mm⁻¹)",
           title=f"{config['concentration_percent']:g}% Intralipid {config['acquisition_labels'][0]} · N={len(frames)} paired frames per method\nFixed CF-derived gain · {config['iterations']:,} iterations · final state")
    ax.grid(axis="y", alpha=.2)
    fig.get_layout_engine().set(rect=(0, .06, 1, .94))
    fig.text(.45, .02, "Box: median and IQR; whiskers: 1.5×IQR. All frame means shown. CF and DiffOCT share CF-derived gain.", ha="center", fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(run / f"boxplots.{suffix}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, len(results), figsize=(6 * len(results), 8), constrained_layout=True, squeeze=False)
    for col, (name, result) in enumerate(results.items()):
        history = result["history"]
        steps = [h["iteration"] for h in history]
        for i, frame in enumerate(frames):
            axes[0, col].plot(steps, [h["mean_mu"][i] for h in history], color=frame_colors[i], label=str(frame["matlab_bscan"]))
            axes[1, col].semilogy(steps, [h["objective"][i] for h in history], color=frame_colors[i])
        axes[0, col].set(title=LABELS[name].replace("\n", ": "), ylabel="Mean μ in CF window (all A-lines)")
        axes[0, col].legend(title="B-scan", ncol=4, fontsize=8)
        axes[1, col].set(xlabel="Iteration", ylabel="Weighted data MSE + TV")
        for row in range(2):
            axes[row, col].grid(alpha=.15)
    fig.suptitle(f"{config['concentration_percent']:g}% {config['acquisition_labels'][0]} · {len(frames)} frames · fixed CF-derived gain · {config['iterations']:,} steps")
    for suffix in ("png", "pdf"):
        fig.savefig(run / f"convergence.{suffix}", dpi=180)
    plt.close(fig)
