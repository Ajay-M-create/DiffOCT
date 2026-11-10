"""Prepare or execute a fixed-gain H or L phantom pilot in an isolated run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

HERE = next(p for p in Path(__file__).resolve().parents if p.name == "REBUTTAL_liquid_intralipid_cf_comparison")
SOURCE_FILES = ("pilot.py", "model.py", "gain.py", "preprocessing.py", "validate.py", "summarize_pilot.py", "plot_alignment.py")


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def checked_run(path):
    path = Path(path).resolve()
    if HERE not in path.parents:
        raise ValueError("Run path must be inside this experiment")
    return path


def prepare(iterations=15000, concentration=0.5, domains=("full_depth", "cf_window"), python_frames=None, acquisition="H", run_directory=None, source_directory=None, validation_result=None):
    import h5py
    from preprocessing import correct_and_mask
    from plot_alignment import load_frames, DATA
    from validate import validate
    from gain import fixed_cf_gain, FORMULA

    if iterations < 1:
        raise ValueError("Iterations must be positive")
    if acquisition not in ("H", "L"):
        raise ValueError("Acquisition must be H or L")
    if not domains or any(d not in ("full_depth", "cf_window") for d in domains):
        raise ValueError("Unsupported fitting domains")
    python_frames = np.linspace(0, 511, 8, dtype=int).tolist() if python_frames is None else list(python_frames)
    if not python_frames or len(set(python_frames)) != len(python_frames) or any(f < 0 or f >= 512 for f in python_frames):
        raise ValueError("Frame indices must be unique and within 0..511")
    config = dict(concentration_percent=float(concentration), python_frames=python_frames,
                  acquisition_labels=[acquisition], seed_repeats=1, base_seed=42,
                  iterations=iterations, learning_rate=0.001, final_lr_factor=0.01,
                  optimizer="Adam", adam_betas=[0.9, 0.999], adam_eps=1e-8,
                  mse_weight=1000.0, tv_weight=1000.0, mu_bounds=[1e-6, 20.0],
                  initialization="NumPy PCG64 Uniform(0,5), full 500x500 frame before crop",
                  seed_mapping="SeedSequence([42, python_volume, python_frame, repeat])",
                  alpha=1.0, noise_domain="measured", correction="(I-N)/(S*H)",
                  gain="Fixed per A-line from CF natural-log intercept; converted and never optimized",
                  gain_variant="per_aline_cf",
                  gain_formula=FORMULA, gain_log_base="CF G_prime is natural log; DiffOCT gain is dB",
                  gain_origin="Absolute zero-based first original zFitMask pixel per A-line",
                  gain_fallback="Median valid gain of the same B-scan for nonfinite G/mu, nonpositive mu, or empty original fit band; exclude fallback A-lines from paired summaries",
                  gain_disclosure="CF and DiffOCT share a CF-derived gain; the comparison is conditional on that gain",
                  fitting_domains=list(domains), primary_domain="full_depth",
                  loss_support={"cf_window": "zFitMask intersect finite positive corrected pixels",
                                "full_depth": "Finite positive corrected pixels from each CF-band start to last usable depth; IntMask unused"},
                  transport_support="Continuous modeled path within each domain; missing observations retain latent optical-path mu",
                  tv_support="Neighbor pairs within transport support",
                  mu_summary_support="Same valid pixels inside original CF mask for BOTH fitting domains; mean per A-line then paired A-line mean",
                  geometry="TIFF(z,x), HDF5 mask(v,y,x,z).T; zFP evaluated at z=1..500; no additional shifts",
                  mapping="Exported concentrations order, H then L per concentration; H MATLAB odd volumes, L MATLAB even volumes",
                  primary_seed_repeat=0, dtype="float32", batch_size=len(python_frames),
                  summary_state="Final iteration; no early stopping or checkpoint selection for the comparison",
                  purpose=f"{acquisition} phantom comparison with fixed CF-derived gain and a specified iteration budget")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    concentration_tag = f"{concentration:g}".replace(".", "p")
    run = checked_run(run_directory if run_directory is not None else HERE / "runs" / f"fixed_cf_gain_{acquisition}_{concentration_tag}_{run_id}_{config_hash}")
    run.mkdir(parents=True, exist_ok=False)
    (run / "code").mkdir()
    (run / "logs").mkdir()
    (run / "results").mkdir()
    for domain in config["fitting_domains"]:
        (run / "results" / domain).mkdir()
    source_directory = HERE if source_directory is None else Path(source_directory)
    for name in SOURCE_FILES:
        shutil.copy2(source_directory / name, run / "code" / name)
    print(f"RUN_DIR={run}", flush=True)
    checks = validate() if validation_result is None else validation_result
    write_json(run / "validation.json", checks)
    frames, calibration, sources = load_frames(config["python_frames"], config["acquisition_labels"], config["concentration_percent"])
    config["dz_mm"] = calibration["dz_water_um"] / 1000.0
    arrays = {name: [] for name in ("target_db", "loss_mask", "comparison_mask", "transport_mask", "full_depth_loss_mask", "full_depth_transport_mask", "excluded_mask", "original_fit_mask", "measured_db", "noise", "sensitivity", "confocal", "cf_mu", "cf_mu_filtered", "cf_G_prime", "gain_db", "gain_fallback")}
    frame_records = []
    with h5py.File(DATA / "CF_Method_Outputs.mat", "r") as cf:
        for frame in frames:
            r = frame["report"].copy()
            corrected = correct_and_mask(frame["db"], frame["noise"], frame["sensitivity"], frame["confocal"], frame["fit"])
            for key in ("target_db", "loss_mask", "comparison_mask", "transport_mask", "full_depth_loss_mask", "full_depth_transport_mask", "excluded_mask", "original_fit_mask"):
                arrays[key].append(corrected[key])
            arrays["measured_db"].append(frame["db"])
            for key in ("noise", "sensitivity", "confocal"):
                arrays[key].append(frame[key])
            for key, source in (("cf_mu", "mu"), ("cf_mu_filtered", "muFilt")):
                arrays[key].append(cf[source][r["python_volume"], r["python_bscan"], :, 0])
            g_prime = cf["G"][r["python_volume"], r["python_bscan"], :, 0].astype(np.float64)
            fixed_gain, fallback = fixed_cf_gain(g_prime, arrays["cf_mu"][-1], frame["fit"], config["dz_mm"])
            arrays["cf_G_prime"].append(g_prime)
            arrays["gain_db"].append(fixed_gain)
            arrays["gain_fallback"].append(fallback)
            r["gain_fallback_alines"] = int(fallback.sum())
            r["gain_median_min_max_db"] = [float(np.median(fixed_gain)), float(fixed_gain.min()), float(fixed_gain.max())]
            valid_per_aline = corrected["loss_mask"].sum(axis=0)
            r.update(valid_pixels=int(corrected["loss_mask"].sum()), excluded_pixels=int(corrected["excluded_mask"].sum()),
                     valid_alines=int((valid_per_aline > 0).sum()), min_valid_pixels_per_aline=int(valid_per_aline.min()),
                     max_valid_pixels_per_aline=int(valid_per_aline.max()))
            r["full_depth_valid_pixels"] = int(corrected["full_depth_loss_mask"].sum())
            r["full_depth_excluded_pixels"] = int((corrected["full_depth_transport_mask"] & ~corrected["full_depth_loss_mask"]).sum())
            if r["noncontiguous_alines"]:
                raise ValueError(f"Unexpected original noncontiguous transport band: {r}")
            frame_records.append(r)
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    for key in ("target_db", "measured_db"):
        arrays[key] = arrays[key].astype(np.float32)
    assert np.all(arrays["loss_mask"] <= arrays["original_fit_mask"])
    assert not np.any(arrays["loss_mask"] & arrays["excluded_mask"])
    assert np.isfinite(arrays["target_db"]).all()
    assert np.all(arrays["comparison_mask"] <= arrays["full_depth_loss_mask"])
    config["crop_z_python_start_stop"] = {}
    for domain, key in (("cf_window", "transport_mask"), ("full_depth", "full_depth_transport_mask")):
        occupied = np.any(arrays[key], axis=(0, 2))
        config["crop_z_python_start_stop"][domain] = [int(np.where(occupied)[0][0]), int(np.where(occupied)[0][-1] + 1)]
    np.savez_compressed(run / "prepared.npz", **arrays)
    write_json(run / "config.json", config)
    for source in sources:
        path = Path(source["path"])
        if path.stat().st_mtime_ns != source["mtime_ns"]:
            raise RuntimeError(f"Input changed during preparation: {path}")
        source["sha256"] = sha256(path)
        if path.stat().st_mtime_ns != source["mtime_ns"]:
            raise RuntimeError(f"Input changed during hashing: {path}")
    def git_output(*args):
        return subprocess.run(["git", *args], cwd=HERE, text=True, capture_output=True, check=True).stdout.strip()
    provenance = dict(created_utc=run_id, source_files=sources, frames=frame_records,
                      calibration=calibration, git_revision=git_output("rev-parse", "HEAD"),
                      git_status=git_output("status", "--short"),
                      code_sha256={name: sha256(run / "code" / name) for name in SOURCE_FILES},
                      prepared_sha256=sha256(run / "prepared.npz"), config_sha256=sha256(run / "config.json"),
                      exclusion_policy="User requested pixel-level exclusion on 2026-09-28",
                      noise_confirmation="User confirmed Sebastian also uses measured-space noise subtraction before S/H")
    write_json(run / "provenance.json", provenance)
    write_json(run / "status.json", {"status": "prepared", "expected_optimizations": len(frames) * config["seed_repeats"] * len(config["fitting_domains"]), "fitting_domains": config["fitting_domains"]})
    print(f"Prepared {len(frames)} B-scans; excluded {int(arrays['excluded_mask'].sum())} pixels; all usable A-lines retained.", flush=True)
    return run


def fit(run, domain):
    import platform
    import torch
    from model import objective

    run = checked_run(run)
    config = json.loads((run / "config.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    for name, expected in provenance["code_sha256"].items():
        if sha256(run / "code" / name) != expected:
            raise RuntimeError(f"Snapshot changed: {name}")
    if sha256(run / "prepared.npz") != provenance["prepared_sha256"] or sha256(run / "config.json") != provenance["config_sha256"]:
        raise RuntimeError("Prepared data or configuration changed")
    result_dir = run / "results" / domain
    if list(result_dir.glob("seed_*")):
        raise FileExistsError("Result files exist; create a new run instead of overwriting")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU pilot requires an allocated CUDA GPU")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    with np.load(run / "prepared.npz") as data:
        start, stop = config["crop_z_python_start_stop"][domain]
        target = torch.as_tensor(data["target_db"][:, start:stop], device=device)
        prefix = "full_depth_" if domain == "full_depth" else ""
        valid = torch.as_tensor(data[prefix + "loss_mask"][:, start:stop], device=device)
        transport = torch.as_tensor(data[prefix + "transport_mask"][:, start:stop], device=device)
        comparison = torch.as_tensor(data["comparison_mask"][:, start:stop], device=device)
        fixed_gain = torch.as_tensor(data["gain_db"], dtype=target.dtype, device=device)
    if not torch.all(valid.sum(dim=(1, 2)) > 0):
        raise ValueError("Empty frame requires explicit skip handling before optimization")
    env = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               slurm_job_id=os.environ.get("SLURM_JOB_ID"), hostname=platform.node())
    write_json(run / f"environment_{domain}.json", env)
    write_json(run / f"status_{domain}.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "environment": env})
    iterations = config["iterations"]
    for repeat in range(config["seed_repeats"]):
        initial = []
        stabilized = []
        for r in provenance["frames"]:
            rng = np.random.default_rng(np.random.SeedSequence([config["base_seed"], r["python_volume"], r["python_bscan"], repeat]))
            full = rng.uniform(0.0, 5.0, size=(500, 500)).astype(np.float32)
            stabilized.append(int((full[start:stop] < config["mu_bounds"][0]).sum()))
            initial.append(np.maximum(full[start:stop], config["mu_bounds"][0]))
        mu = torch.nn.Parameter(torch.as_tensor(np.stack(initial), device=device))
        optimizer = torch.optim.Adam([mu], lr=config["learning_rate"], betas=tuple(config["adam_betas"]), eps=config["adam_eps"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations,
                    eta_min=config["learning_rate"] * config["final_lr_factor"])
        best_mu = mu.detach().clone()
        n = mu.shape[0]
        best_objective = torch.full((n,), float("inf"), device=device)
        best_mse = best_objective.clone()
        best_iteration = torch.zeros(n, dtype=torch.int64, device=device)
        history = []
        begin = time.monotonic()
        for step in range(iterations + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, mse, gain, tv = objective(mu, target, valid, transport, config["dz_mm"], fixed_gain, config["mse_weight"], config["tv_weight"])
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Nonfinite loss at seed {repeat}, iteration {step}")
            with torch.no_grad():
                improved = loss < best_objective
                best_mu.copy_(torch.where(improved[:, None, None], mu, best_mu))
                best_objective.copy_(torch.minimum(best_objective, loss))
                best_mse.copy_(torch.where(improved, mse, best_mse))
                best_iteration.copy_(torch.where(improved, step, best_iteration))
            if step % 100 == 0 or step == iterations:
                means = (mu.detach() * comparison).sum(dim=(1, 2)) / comparison.sum(dim=(1, 2))
                history.append(dict(iteration=step, mse=mse.detach().cpu().tolist(), objective=loss.detach().cpu().tolist(),
                                    tv=tv.detach().cpu().tolist(), mean_mu=means.cpu().tolist()))
            if step % 1000 == 0 or step == iterations:
                print(f"domain={domain} seed={repeat} iteration={step}/{iterations} mean_mse={float(mse.mean()):.6g} "
                      f"comparison_mean_mu={float(means.mean()):.6g} elapsed_s={time.monotonic()-begin:.1f}", flush=True)
            if step == iterations:
                break
            loss.sum().backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                mu.clamp_(*config["mu_bounds"])
        with torch.no_grad():
            check_loss, check_mse, check_gain, check_tv = objective(best_mu, target, valid, transport, config["dz_mm"], fixed_gain, config["mse_weight"], config["tv_weight"])
            if not (torch.allclose(check_loss, best_objective, rtol=2e-5, atol=1e-4)
                    and torch.allclose(check_mse, best_mse, rtol=2e-5, atol=1e-5)
                    and torch.equal(check_gain, fixed_gain)):
                raise AssertionError("Best-state mu/gain/objective mismatch")
        temporary = result_dir / f"seed_{repeat}.tmp.npz"
        result = result_dir / f"seed_{repeat}.npz"
        np.savez_compressed(temporary, mu_best=best_mu.detach().cpu().numpy(), mu_final=mu.detach().cpu().numpy(),
                            gain_db=fixed_gain.cpu().numpy(), mse_final=mse.detach().cpu().numpy(),
                            objective_final=loss.detach().cpu().numpy(), tv_final=tv.detach().cpu().numpy(),
                            mse_best=best_mse.cpu().numpy(),
                            objective_best=best_objective.cpu().numpy(), iteration_best=best_iteration.cpu().numpy(),
                            crop_start=start, crop_stop=stop)
        temporary.replace(result)
        write_json(result_dir / f"seed_{repeat}_history.json", history)
        write_json(result_dir / f"seed_{repeat}_complete.json",
                   dict(seed_repeat=repeat, frames=n, elapsed_s=time.monotonic()-begin, result_sha256=sha256(result),
                        initialization_stabilized_counts=stabilized,
                        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(), saved_state_verified=True))
        del mu, optimizer, scheduler, best_mu
    from summarize_pilot import summarize
    summarize(run, domain)
    write_json(run / f"status_{domain}.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(), "optimizations": len(provenance["frames"]) * config["seed_repeats"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "fit", "summarize"])
    parser.add_argument("--iterations", type=int, default=15000, help="Preparation only: optimizer steps and cosine schedule length")
    parser.add_argument("--concentration", type=float, default=0.5, help="Preparation only: concentration in percent")
    parser.add_argument("--frames", type=int, nargs="+", help="Preparation only: explicit Python B-scan indices")
    parser.add_argument("--acquisition", choices=["H", "L"], default="H")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--domain", choices=["full_depth", "cf_window", "both"], default="both")
    args = parser.parse_args()
    if args.mode == "prepare":
        domains = ("full_depth", "cf_window") if args.domain == "both" else (args.domain,)
        prepare(args.iterations, args.concentration, domains, args.frames, args.acquisition)
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required")
        run = checked_run(args.run_dir)
        if args.mode == "fit":
            configured = json.loads((run / "config.json").read_text())["fitting_domains"]
            domains = configured if args.domain == "both" else [args.domain]
            if any(d not in configured for d in domains):
                raise ValueError("Requested fitting domain was not prepared")
            for domain in domains:
                try:
                    fit(run, domain)
                except Exception as error:
                    write_json(run / f"status_{domain}.json", {"status": "failed", "error": repr(error)})
                    raise
        else:
            from summarize_pilot import summarize
            summarize(run, args.domain)


if __name__ == "__main__":
    main()
