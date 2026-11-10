"""Matched solver comparison on grouped attenuation parameters of one real A-line.

Commands: prepare, validate, run --seed S, summarize (see README.md).
All fitting uses the same native measurements, physical depth range and gain.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import cma
import numpy as np
import scipy
from scipy.optimize import minimize
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
RESULTS = HERE / "results"
SOURCE = ROOT / "experiments/REBUTTAL_liquid_intralipid_cf_comparison/runs/production_HL_full_depth_cf_gain_20260928T175833Z/shards/H_1_192_255/plot_data.npz"
METHODS = ["Adam", "L-BFGS-B", "Random Search", "CMA-ES"]
LOW, HIGH, SCALE = 0.01, 20.0, 1000.0


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    """Freeze one prespecified central A-line, before looking at solver results."""
    DATA.mkdir(parents=True, exist_ok=True)
    if (DATA / "config.json").exists():
        raise FileExistsError("Prepared data already exist; preserve this benchmark's frozen inputs")
    frame, aline = 255, 250
    with np.load(SOURCE) as f:
        row = int(np.flatnonzero(f["python_bscan"] == frame)[0])
        def line(key):
            return f[key][row, :, aline].copy()
        transport = line("full_depth_transport_mask").astype(bool)
        original = line("original_fit_mask").astype(bool)
        start = int(f["original_cf_window_start_python"][row, aline])
        end = int(np.flatnonzero(transport)[-1]) + 1
        assert original.argmax() == start and transport[start:end].all()
        assert not bool(f["gain_fallback"][row, aline])
        arrays = {k: line(k)[start:end] for k in
                  ["target_db", "measured_db", "full_depth_loss_mask", "comparison_mask", "mu_final"]}
        arrays.update(noise=f["noise"][row, start:end],
                      sensitivity=f["sensitivity"][row, start:end],
                      confocal=f["confocal"][row, start:end],
                      depth_python=np.arange(start, end),
                      dz_mm=np.array(float(f["dz_mm"])),
                      gain_db=np.array(float(f["gain_db"][row, aline])),
                      cf_mu=np.array(float(f["cf_mu"][row, aline])),
                      cf_G_prime=np.array(float(f["cf_G_prime"][row, aline])))
    arrays["valid"] = arrays.pop("full_depth_loss_mask").astype(bool)
    arrays["cf_comparison"] = arrays.pop("comparison_mask").astype(bool)
    assert arrays["cf_comparison"].sum() == 100
    assert np.isfinite(arrays["target_db"][arrays["valid"]]).all()
    np.savez_compressed(DATA / "aline.npz", **arrays)
    length = end - start
    config = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(SOURCE.relative_to(ROOT)), source_sha256=sha256(SOURCE),
        aline_sha256=sha256(DATA / "aline.npz"),
        selection="Prespecified spatial center: 1% H, B-scan 256, A-line 251 (MATLAB indices); no selection by solver outcome",
        concentration_percent=1.0, acquisition="H", bscan_python=frame, aline_python=aline,
        depth_start_python=start, depth_stop_exclusive_python=end, native_depth_samples=length,
        valid_observations=int(arrays["valid"].sum()),
        sizes=[n for n in [25, 50, 75, 100, 150, 200, 300, 400] if n < length] + [length],
        seeds=list(range(8)), methods=METHODS, max_objective_evaluations=20000,
        tv_weights=[0.0, 1.0], bounds=[LOW, HIGH], init_range=[LOW, 10.0],
        init_seed_base=42, adam_lr=0.01, loss_scale=SCALE, dtype="float32",
        grouping="Contiguous nonempty depth bins: group[z] = floor(z*N/L); one constant mu per bin; expand to the unchanged native grid",
        forward="gain_db + 10*log10(mu[z]) - (20/ln(10))*dz*sum(mu[:z])",
        objective="1000 * (mean valid squared dB residual + beta * mean absolute adjacent mu differences on the expanded native grid)",
        tv_normalization="Fixed native-grid denominator L-1 for every N; beta=1 matches production's axial TV/MSE weight ratio",
        alpha=1.0, gain="Fixed from the same CF intercept conversion as liquid phantom production; contains CF mu; never optimized",
        target="Existing (I-N)/(S*H) corrected log-intensity; preserve individual invalid-observation exclusions and continuous optical transport",
        primary_metrics=["best common objective", "total wall time", "ms per objective evaluation"],
        secondary_metrics=["intensity RMSE in dB", "CF-window mean mu relative disagreement (reference, not ground truth)"],
        budget_note="Common maximum number of objective evaluations, including the initial point; each L-BFGS-B line-search evaluation and each CMA-ES candidate counted. Native convergence may stop a solver early.",
        original_budget_note="Original synthetic Figure 1 used optimizer-specific iteration/evaluation budgets; the new experiment uses a common evaluation cap and records actual counts.",
        preprocessing_note="Already despeckled; no additional smoothing. Current user describes 7x15 averaging; historical liquid acquisition notes list [7,11,11] in z,x,y. Exact kernel description remains to be reconciled.",
        interpretation="One fixed real A-line tests solver ranking on this processed signal. TV on/off isolates adding axial regularization. It does not isolate intrinsic correlation from despeckling or test lateral TV coupling across A-lines.",
    )
    write_json(DATA / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)


def load_data():
    config = json.loads((DATA / "config.json").read_text())
    assert sha256(DATA / "aline.npz") == config["aline_sha256"]
    with np.load(DATA / "aline.npz") as f:
        data = {k: f[k] for k in f.files}
    return config, data


def groups(length, n):
    if not 1 <= n <= length:
        raise ValueError("N must not exceed the number of native depth samples")
    index = np.arange(length) * n // length
    assert np.array_equal(np.unique(index), np.arange(n))
    return index


class Objective:
    def __init__(self, data, n, beta, device="cpu", dtype=torch.float32):
        self.group = torch.tensor(groups(len(data["target_db"]), n), device=device)
        self.target = torch.as_tensor(data["target_db"], dtype=dtype, device=device)
        self.valid = torch.as_tensor(data["valid"], dtype=torch.bool, device=device)
        self.gain = torch.as_tensor(data["gain_db"], dtype=dtype, device=device)
        self.dz = float(data["dz_mm"])
        self.beta = beta

    def parts(self, theta):
        mu = theta[self.group]
        optical = torch.cumsum(mu * self.dz, dim=0)
        tau = torch.cat([optical.new_zeros(1), optical[:-1]])
        pred = self.gain + 10 * torch.log10(mu) - (20 / math.log(10)) * tau
        mse = (pred[self.valid] - self.target[self.valid]).square().mean()
        tv = (mu[1:] - mu[:-1]).abs().mean()
        return mse + self.beta * tv, mse, tv, mu, pred

    def __call__(self, theta):
        return SCALE * self.parts(theta)[0]


class BudgetReached(Exception):
    pass


class Evaluator:
    """Count every objective evaluation; track the best evaluated state."""
    def __init__(self, objective, budget, device):
        self.objective, self.budget, self.device = objective, budget, device
        self.count = 0
        self.best_value = math.inf
        self.best_theta = None
        self.history = []
        self.start = time.perf_counter()

    def evaluate(self, theta, grad=False):
        if self.count >= self.budget:
            raise BudgetReached
        if grad:
            loss = self.objective(theta)
            gradient, = torch.autograd.grad(loss, theta)
        else:
            with torch.no_grad():
                loss = self.objective(theta)
            gradient = None
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite objective")
        if gradient is not None and not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite gradient")
        self.count += 1
        if value < self.best_value:
            self.best_value = value
            self.best_theta = theta.detach().clone()
        if self.count == 1 or self.count % 100 == 0 or self.count == self.budget:
            self.record()
        return value, gradient

    def record(self):
        self.history.append([self.count, time.perf_counter() - self.start, self.best_value / SCALE])


def solve(method, objective, x0, budget, seed, device):
    # Warm up forward/backward at x0 without updating any parameter.
    warm = torch.tensor(x0, device=device, requires_grad=True)
    torch.autograd.grad(objective(warm), warm)
    if method == "Adam":
        theta = torch.nn.Parameter(torch.tensor(x0, device=device))
        optimizer = torch.optim.Adam([theta], lr=0.01)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ev = Evaluator(objective, budget, device)
    stop = "evaluation_budget"
    details = {}
    try:
        if method == "Adam":
            while ev.count < budget:
                _, gradient = ev.evaluate(theta, grad=True)
                theta.grad = gradient
                optimizer.step()
                with torch.no_grad():
                    theta.clamp_(LOW, HIGH)
        elif method == "L-BFGS-B":
            def fg(x):
                theta = torch.tensor(x, device=device, dtype=torch.float32, requires_grad=True)
                value, gradient = ev.evaluate(theta, grad=True)
                return value, gradient.detach().cpu().numpy().astype(np.float64)
            result = minimize(fg, x0.astype(np.float64), jac=True, method="L-BFGS-B",
                              bounds=[(LOW, HIGH)] * len(x0),
                              options=dict(maxiter=budget, maxfun=budget, maxcor=10, ftol=0., gtol=0., maxls=20))
            stop = str(result.message)
            details = dict(scipy_status=int(result.status), outer_iterations=int(result.nit))
        elif method == "Random Search":
            theta = torch.tensor(x0, device=device)
            current, _ = ev.evaluate(theta)
            bias = torch.zeros_like(theta)
            rho, successes, failures = 1.0, 0, 0
            rng = torch.Generator(device=device).manual_seed(1000003 + seed)
            while ev.count < budget:
                if successes >= 5:
                    rho *= 2
                elif failures >= 3:
                    rho *= 0.5
                if rho == 0:
                    stop = "step_size_underflow"
                    break
                proposal = theta + bias + rho * torch.randn(theta.shape, generator=rng, device=device)
                candidate = proposal.clamp(LOW, HIGH)
                value, _ = ev.evaluate(candidate)
                if value < current:
                    bias = 0.4 * (proposal - theta) + 0.2 * bias
                    theta, current = candidate, value
                    successes, failures = successes + 1, 0
                else:
                    reverse = (2 * theta - proposal).clamp(LOW, HIGH)
                    value, _ = ev.evaluate(reverse)
                    if value < current:
                        bias -= 0.4 * (proposal - theta)
                        theta, current = reverse, value
                        successes, failures = successes + 1, 0
                    else:
                        bias *= 0.5
                        successes, failures = 0, failures + 1
            details = dict(final_step_size=rho)
        elif method == "CMA-ES":
            ev.evaluate(torch.tensor(x0, device=device))
            opts = dict(bounds=[LOW, HIGH], seed=seed + 1, verbose=-9, maxfevals=budget,
                        tolfun=0., tolx=0., CMA_active=False,
                        updatecovwait=max(1, math.ceil(math.sqrt(len(x0)))))
            es = cma.CMAEvolutionStrategy(x0.astype(float).tolist(), 1., opts)
            while ev.count < budget and not es.stop():
                candidates = es.ask()
                fitness = []
                for candidate in candidates:
                    value, _ = ev.evaluate(torch.tensor(candidate, device=device, dtype=torch.float32))
                    fitness.append(value)
                es.tell(candidates, fitness)
            if es.stop():
                stop = str(es.stop())
            details = dict(population_size=int(es.popsize), complete_generations=int(es.countiter))
        else:
            raise ValueError(method)
    except BudgetReached:
        stop = "evaluation_budget"
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - ev.start
    if not ev.history or ev.history[-1][0] != ev.count:
        ev.record()
    assert ev.count <= budget and ev.best_theta is not None
    result = dict(method=method, seed=seed, fwd_evals=ev.count, wall_sec=wall,
                  ms_per_eval=wall * 1000 / ev.count, termination=stop,
                  best_objective=ev.best_value / SCALE, solver_details=details,
                  peak_gpu_mb=(torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.))
    return ev.best_theta, np.asarray(ev.history), result


def validate():
    """Check physical invariance, production algebra, gradients and hard budgets."""
    config, data = load_data()
    length = len(data["target_db"])
    checks = {}
    for n in config["sizes"]:
        index = groups(length, n)
        assert len(index) == length and index[0] == 0 and index[-1] == n - 1
    # Constant CF mu and converted fixed gain reproduce the supplied CF line.
    obj = Objective(data, 25, 0, dtype=torch.float64)
    theta = torch.full((25,), float(data["cf_mu"]), dtype=torch.float64)
    pred = obj.parts(theta)[4].numpy()
    expected = (10 / np.log(10)) * (float(data["cf_G_prime"]) -
               2 * float(data["cf_mu"]) * float(data["dz_mm"]) * data["depth_python"])
    np.testing.assert_allclose(pred, expected, rtol=0, atol=1e-10)
    checks["cf_constant_round_trip_max_db"] = float(np.max(abs(pred - expected)))
    # Check against the actual production objective on an arbitrary native map.
    import importlib.util
    source = ROOT / "experiments/REBUTTAL_liquid_intralipid_cf_comparison/model.py"
    spec = importlib.util.spec_from_file_location("liquid_production_model", source)
    production = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(production)
    rng = np.random.default_rng(1827)
    native = torch.tensor(rng.uniform(.5, 4, length), dtype=torch.float64)
    obj = Objective(data, length, 1, dtype=torch.float64)
    prod = production.objective(native[None, :, None], obj.target[None, :, None],
                                obj.valid[None, :, None], torch.ones((1, length, 1), dtype=torch.bool),
                                obj.dz, obj.gain.reshape(1, 1))[0]
    torch.testing.assert_close(obj(native), prod[0], rtol=1e-12, atol=1e-9)
    checks["production_objective_difference"] = float(abs(obj(native) - prod[0]))
    for beta in config["tv_weights"]:
        obj = Objective(data, 25, beta, dtype=torch.float64)
        theta = torch.tensor(rng.uniform(.5, 4, 25), dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(obj, (theta,), eps=1e-6, atol=1e-3, rtol=1e-4)
        checks[f"gradient_beta_{beta:g}"] = "passed"
    # The missing observation must remain in optical transport.
    missing = np.flatnonzero(~data["valid"])
    assert len(missing) == length - config["valid_observations"]
    checks["missing_observations_preserved"] = missing.tolist()
    x0 = np.random.default_rng(42).uniform(LOW, 10., 25).astype(np.float32)
    obj = Objective(data, 25, 1)
    initial = float(obj(torch.tensor(x0))) / SCALE
    for method in METHODS:
        theta, history, result = solve(method, obj, x0, 31, 0, torch.device("cpu"))
        assert result["fwd_evals"] <= 31
        np.testing.assert_allclose(float(obj(theta)) / SCALE, result["best_objective"], rtol=1e-6)
        assert result["best_objective"] <= initial + 1e-6
        assert torch.all((theta >= LOW) & (theta <= HIGH))
        checks[method] = dict(evaluations=result["fwd_evals"], best_objective=result["best_objective"])
    write_json(DATA / "validation.json", checks)
    print(json.dumps(checks, indent=2), flush=True)


def run(seed, sizes=None, budget=None, cpu=False):
    config, data = load_data()
    if not (DATA / "validation.json").exists():
        raise RuntimeError("Run validate before fitting")
    budget = budget or config["max_objective_evaluations"]
    sizes = sizes or config["sizes"]
    assert seed in config["seeds"]
    device = torch.device("cpu" if cpu else "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Production timing requires a CUDA GPU")
    torch.set_num_threads(1)
    provenance = dict(host=platform.node(), python=platform.python_version(),
                      torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, cma=cma.__version__,
                      cuda=torch.version.cuda, torch_cpu_threads=torch.get_num_threads(),
                      device=str(device), gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                      job_id=os.environ.get("SLURM_JOB_ID"), script_sha256=sha256(__file__),
                      config_sha256=sha256(DATA / "config.json"), data_sha256=config["aline_sha256"],
                      budget=budget, clock="perf_counter; CUDA synchronized before/after run; one untimed forward/backward warmup; no warmup updates")
    out = RESULTS if budget == config["max_objective_evaluations"] and not cpu else HERE / "checks" / f"budget{budget}_{device.type}"
    out.mkdir(parents=True, exist_ok=True)
    for beta in config["tv_weights"]:
        for n in sizes:
            stem = f"tv{beta:g}_n{n:04d}_seed{seed}"
            path = out / f"{stem}.json"
            if path.exists():
                old = json.loads(path.read_text())
                stable = ["script_sha256", "config_sha256", "data_sha256", "budget", "device", "gpu"]
                if any(old["provenance"][key] != provenance[key] for key in stable):
                    raise RuntimeError(f"Existing result has different provenance: {path}")
                print("already complete", stem, flush=True)
                continue
            obj = Objective(data, n, beta, device=device)
            x0 = np.random.default_rng(42 + seed).uniform(LOW, 10., n).astype(np.float32)
            methods, saved = {}, {"theta_initial": x0, "group_index": groups(len(data["target_db"]), n)}
            # Rotate order across seeds to reduce systematic timing-order effects.
            order = METHODS[seed % 4:] + METHODS[:seed % 4]
            for method in order:
                theta, history, record = solve(method, obj, x0, budget, seed, device)
                with torch.no_grad():
                    value, mse, tv, mu, pred = obj.parts(theta)
                np.testing.assert_allclose(float(value.item()), record["best_objective"], rtol=1e-6)
                mu_np = mu.cpu().numpy()
                mean_mu = float(mu_np[data["cf_comparison"]].mean(dtype=np.float64))
                record.update(mse_db2=float(mse.item()), rmse_db=float(mse.sqrt().item()),
                              tv_native=float(tv.item()), cf_window_mean_mu=mean_mu,
                              cf_relative_disagreement_percent=100 * abs(mean_mu / float(data["cf_mu"]) - 1))
                methods[method] = record
                key = method.lower().replace(" ", "_").replace("-", "_")
                saved.update({f"{key}_theta": theta.cpu().numpy(), f"{key}_mu": mu_np,
                              f"{key}_pred_db": pred.cpu().numpy(), f"{key}_history": history})
                print(f"{stem} {method}: objective={record['best_objective']:.6g}, "
                      f"RMSE={record['rmse_db']:.4g} dB, evals={record['fwd_evals']}, "
                      f"wall={record['wall_sec']:.2f}s", flush=True)
            np.savez_compressed(out / f"{stem}.npz", **saved)
            write_json(path, dict(n=n, beta=beta, seed=seed, methods=methods, provenance=provenance,
                                 arrays_sha256=sha256(out / f"{stem}.npz")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "validate", "run"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "validate":
        validate()
    else:
        run(args.seed, args.sizes, args.budget, args.cpu)
