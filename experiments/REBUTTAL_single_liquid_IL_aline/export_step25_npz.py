"""Export the completed TV-on sweep for the standalone figure notebook.

This reads frozen experiment results without rerunning optimization. The
notebook needs only the exported NPZ and JSON files, NumPy, SciPy, and Matplotlib.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
from rerun_step25 import BETA, RUN
STAGE = ROOT / "figures/data/figureS1_optimizer_scaling/liquid"
METHODS = ["Random Search", "CMA-ES", "Adam", "L-BFGS-B"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    config = json.loads((RUN / "data/config.json").read_text())
    sizes, seeds = config["sizes"], config["seeds"]
    assert sizes == list(range(25, 476, 25)) and config["tv_weights"] == [BETA]
    assert len(seeds) == 8
    with np.load(RUN / "data/aline.npz", allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    assert sha256(RUN / "data/aline.npz") == config["aline_sha256"]
    config_sha = sha256(RUN / "data/config.json")
    shape = (len(sizes), len(seeds), len(METHODS))
    depth_count = len(data["target_db"])
    cf_mask, valid = data["cf_comparison"].astype(bool), data["valid"].astype(bool)
    assert depth_count == 476 and valid.sum() == 475 and cf_mask.sum() == 100
    assert valid[cf_mask].all()
    arrays = dict(sizes=np.array(sizes), seeds=np.array(seeds), methods=np.array(METHODS),
                  beta=np.array(BETA), group_index=np.empty((len(sizes), depth_count), dtype=np.int32),
                  mu=np.empty(shape + (depth_count,), dtype=np.float32),
                  predicted_db=np.empty(shape + (depth_count,), dtype=np.float32),
                  theta=np.full(shape + (max(sizes),), np.nan, dtype=np.float32),
                  theta_initial=np.full(shape[:2] + (max(sizes),), np.nan, dtype=np.float32),
                  history_length=np.empty(shape, dtype=np.int32))
    for key in ["cf_mape_percent", "ms_per_eval", "wall_sec", "mse_db2", "rmse_db",
                "cf_window_mean_mu", "cf_relative_disagreement_percent", "best_objective"]:
        arrays[key] = np.empty(shape, dtype=np.float64)
    arrays["fwd_evals"] = np.empty(shape, dtype=np.int32)
    for key, value in data.items():
        arrays["aline_" + key] = value
    histories, terminations, records = {}, [], []
    for ni, n in enumerate(sizes):
        for si, seed in enumerate(seeds):
            stem = f"tv{BETA:g}_n{n:04d}_seed{seed}"
            json_path, npz_path = RUN / "results" / (stem + ".json"), RUN / "results" / (stem + ".npz")
            result = json.loads(json_path.read_text())
            assert (result["n"], result["seed"], result["beta"]) == (n, seed, BETA)
            provenance = result["provenance"]
            assert provenance["script_sha256"] == config["base_runner_sha256"]
            assert provenance["config_sha256"] == config_sha
            assert provenance["data_sha256"] == config["aline_sha256"]
            assert provenance["budget"] == 20000 and provenance["device"] == "cuda"
            assert sha256(npz_path) == result["arrays_sha256"]
            records.append(dict(stem=stem, json_sha256=sha256(json_path),
                                npz_sha256=result["arrays_sha256"], provenance=provenance))
            with np.load(npz_path, allow_pickle=False) as f:
                group = f["group_index"]
                np.testing.assert_array_equal(group, np.arange(depth_count) * n // depth_count)
                arrays["group_index"][ni] = group
                arrays["theta_initial"][ni, si, :n] = f["theta_initial"]
                for mi, method in enumerate(METHODS):
                    index = (ni, si, mi)
                    key = method.lower().replace(" ", "_").replace("-", "_")
                    mu, pred, theta, history = [f[key + suffix] for suffix in
                                               ["_mu", "_pred_db", "_theta", "_history"]]
                    m = result["methods"][method]
                    np.testing.assert_array_equal(mu, theta[group])
                    mse = np.mean((pred[valid].astype(float) - data["target_db"][valid]) ** 2)
                    np.testing.assert_allclose(mse, m["mse_db2"], rtol=2e-5, atol=1e-8)
                    tv = np.mean(np.abs(np.diff(mu.astype(float))))
                    np.testing.assert_allclose(mse + BETA * tv, m["best_objective"], rtol=2e-5, atol=1e-8)
                    assert np.isfinite(mu).all() and ((mu >= .01) & (mu <= 20)).all()
                    assert 0 < m["fwd_evals"] <= 20000 and m["wall_sec"] > 0
                    np.testing.assert_allclose(m["ms_per_eval"], m["wall_sec"] * 1000 / m["fwd_evals"])
                    assert history[-1, 0] == m["fwd_evals"] and np.all(np.diff(history[:, 2]) <= 1e-8)
                    reference = float(data["cf_mu"])
                    arrays["cf_mape_percent"][index] = 100 * np.mean(np.abs(mu[cf_mask].astype(float) - reference) / reference)
                    for metric in ["ms_per_eval", "wall_sec", "mse_db2", "rmse_db", "fwd_evals",
                                   "cf_window_mean_mu", "cf_relative_disagreement_percent", "best_objective"]:
                        arrays[metric][index] = m[metric]
                    arrays["mu"][index], arrays["predicted_db"][index] = mu, pred
                    arrays["theta"][index][:n] = theta
                    arrays["history_length"][index] = len(history)
                    histories[index] = history.copy()
                    terminations.append(m["termination"])
    assert len(records) == 152
    gpu_models = {r["provenance"]["gpu"] for r in records}
    assert len(gpu_models) == 1
    arrays["termination"] = np.array(terminations).reshape(shape)
    arrays["history"] = np.full(shape + (int(arrays["history_length"].max()), 3), np.nan)
    for index, history in histories.items():
        arrays["history"][index][:len(history)] = history
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    run_directory=str(RUN.relative_to(ROOT)), solver_runs=608,
                    array_axes=["problem size", "seed", "method"], methods=METHODS,
                    sizes=sizes, seeds=seeds, axial_tv=BETA,
                    objective="1000 * (mean squared corrected-intensity residual in dB + beta * mean absolute adjacent mu difference); axial TV on, beta=1",
                    mape_definition="100 * mean(abs(mu_recovered[CF window] - mu_CF) / mu_CF), calculated per seed",
                    mape_reference="CF estimate within the original 100-pixel CF window; not per-pixel ground truth",
                    error_bars="Sample standard deviation across eight numerical seeds (ddof=1)",
                    timing_definition="1000 * total solver wall seconds / actual objective-evaluation count",
                    history_columns=["objective evaluations", "elapsed seconds", "best objective without 1000 multiplier"],
                    padded_arrays="theta, theta_initial and history use trailing NaNs; valid lengths are N and history_length",
                    calibration_note="Fixed gain contains CF information; CF MAPE is conditional reference agreement",
                    gpu=next(iter(gpu_models)), input_sha256=config["aline_sha256"],
                    config_sha256=config_sha, runner_sha256=config["base_runner_sha256"],
                    exporter_sha256=sha256(__file__), checks="All 608 saved states, grouping, MSE, budgets, timing identities and input hashes verified")
    STAGE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STAGE / "results_tv_on.npz", **arrays)
    shutil.copy2(RUN / "data/aline.npz", STAGE / "aline.npz")
    shutil.copy2(RUN / "data/config.json", STAGE / "config.json")
    shutil.copy2(RUN / "data/validation.json", STAGE / "validation.json")
    write_json(STAGE / "notebook_metadata.json", metadata)
    write_json(STAGE / "source_manifest.json", dict(sources=records, export_sha256=sha256(STAGE / "results_tv_on.npz")))
    (STAGE / "README.md").write_text("""# Liquid IL single-A-line figure data

Use `figures/figureS1b_liquid_aline_scaling.ipynb` from the repository
root or the figures directory. Its plotting cells read this exported data and generate
`figureS1b.{pdf,jpeg,png}` in this data folder. Only the assembled S1 figure
is saved in `figures/output/`. The joint/independent comparison is
in `figureS1c_joint_independent.ipynb`, using the sibling `coupling/` export.
The S1 experiment builds all three panels and the section preview with
`bash scripts/run_REBUTTAL_single_liquid_IL_aline.sh figures`.

- `results_tv_on.npz`: all 608 results, metrics, native recovered profiles,
  grouped parameters, convergence histories, and frozen A-line arrays.
  Metric axes are (19 problem sizes, 8 seeds, 4 methods).
- `aline.npz`: the source A-line and calibration.
- `notebook_metadata.json`: metric definitions and array layout.
- `config.json`, `validation.json`: protocol and numerical validation.
- `source_manifest.json`: checksums and provenance for all 152 configurations.

N = 25, 50, 75, ..., 475. All runs optimize intensity MSE plus axial TV (beta=1).
The figure's left axis is pixelwise attenuation MAPE against CF within its
100-pixel fit window. CF is a reference estimate and contributes to the fixed
gain; this metric is not independent ground-truth attenuation accuracy.
The right axis is measured milliseconds per objective evaluation. Error bars
are sample SD across seeds on one fixed measured signal.
""")
    print(f"Verified and exported 608 solver runs to {STAGE / 'results_tv_on.npz'}")


if __name__ == "__main__":
    main()
