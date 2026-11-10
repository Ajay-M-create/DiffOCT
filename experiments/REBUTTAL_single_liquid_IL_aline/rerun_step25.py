"""Rerun the unchanged solvers with axial TV on (beta=1) and N=25,50,...,475."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
BETA = 1.0
RUN = BASE / "runs/tv_on_step25_20260929"
EXPECTED_SOLVER_SHA = "c88778d390a7b63096f48ffb9fc367b6adfe3b8501a019a210766b150987c8cc"


def backend():
    source = RUN / "benchmark.py"
    if not source.exists():
        source = BASE / "benchmark.py"
    spec = importlib.util.spec_from_file_location("frozen_liquid_benchmark", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.sha256(source) == EXPECTED_SOLVER_SHA
    module.ROOT, module.HERE = ROOT, RUN
    module.DATA, module.RESULTS = RUN / "data", RUN / "results"
    return module


def prepare():
    b = backend()
    if (RUN / "data/config.json").exists():
        raise FileExistsError("Preserve the frozen step-25 run; configuration already exists")
    (RUN / "data").mkdir(parents=True)
    (RUN / "results").mkdir()
    (RUN / "logs").mkdir()
    shutil.copy2(BASE / "benchmark.py", RUN / "benchmark.py")
    shutil.copy2(BASE / "data/aline.npz", RUN / "data/aline.npz")
    config = json.loads((BASE / "data/config.json").read_text())
    config.update(created_utc=datetime.now(timezone.utc).isoformat(),
                  sizes=list(range(25, 476, 25)), tv_weights=[BETA],
                  run_name=RUN.name, base_runner_sha256=EXPECTED_SOLVER_SHA,
                  driver_sha256=b.sha256(__file__),
                  rerun_note="All sizes rerun with unchanged solvers, signal, bounds, seeds and evaluation cap; axial TV on (beta=1, equal to the data term as in the phantom inversions) only; N=25,50,...,475",
                  interpretation="One fixed processed real A-line; axial TV on. Solver comparisons do not isolate intrinsic correlation from despeckling or test lateral TV.")
    b.write_json(RUN / "data/config.json", config)
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "validate", "run"])
    p.add_argument("--task", type=int, default=0)
    args = p.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        b = backend()
        config, _ = b.load_data()
        assert b.sha256(__file__) == config["driver_sha256"]
        if args.command == "validate":
            b.validate()
        else:
            assert 0 <= args.task < 32
            seed, subset = divmod(args.task, 4)
            b.run(seed, sizes=config["sizes"][subset::4])
