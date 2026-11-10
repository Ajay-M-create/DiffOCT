"""S1(a) ownership, validation and reruns using the original solver backend."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=HERE/"runs/bscan_scaling"
STAGE=ROOT/"figures/data/figureS1_optimizer_scaling/bscan"
SIZES=[1,2,3,5,10,15,25,50,75,100,150,200,250,300,350,400,450,500]
METHODS=["Adam","L-BFGS","Random Search","CMA-ES"]
METRICS=["time_ms","mape","fwd_evals","wall_sec","ms_per_fwd","peak_mem_mb"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(path,n):
    payload=json.loads(path.read_text())
    assert {k for k in payload if not k.startswith('_')}=={str(n)}, path
    assert set(payload[str(n)])==set(METHODS), path
    for method in METHODS:
        r=payload[str(n)][method]
        assert r["seeds_present"]==list(range(8)), (path,method)
        for metric in METRICS:
            values=np.asarray(r["seed_"+metric])
            assert values.shape==(8,) and np.isfinite(values).all(), (path,method,metric)
            np.testing.assert_allclose(r["avg_"+metric],values.mean(),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(r["std_"+metric],values.std(),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(r["seed_ms_per_fwd"],
            1000*np.array(r["seed_wall_sec"])/r["seed_fwd_evals"],rtol=1e-10)
    return payload


def prepare():
    """Import the completed original experiment once, preserving its metadata."""
    (RUN/"results").mkdir(parents=True,exist_ok=True)
    manifest_path=RUN/"import_manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "description":"Original S1(a) results imported with all eight per-seed values. Hashes record the imported artifacts, not the historical numerical source code.",
        "sources":{}}
    for n in SIZES:
        name=f"results_scale{n:04d}.json";target=RUN/"results"/name;source=STAGE/name
        if not target.exists() and source.exists():
            check(source,n);shutil.copy2(source,target)
            manifest["sources"][name]=dict(path=str(source.relative_to(ROOT)),sha256=sha(source))
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")


def backend():
    spec=importlib.util.spec_from_file_location("s1_bscan_backend",ROOT/"experiments/optimizer_scaling.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.OUT_DIR=str(RUN/"results");mod.NUM_ITERS=20000;mod.NUM_SEEDS=8
    mod.SCALE_INDEX="";mod.SEED_INDEX=""
    assert mod.ALINE_SCALES==SIZES
    return mod


def validate():
    prepare()
    completed=0
    for n in SIZES:
        p=RUN/"results"/f"results_scale{n:04d}.json"
        if p.exists():check(p,n);completed+=1
    if completed<len(SIZES):
        import torch
        b=backend();b.load_phantom_data(torch.device("cpu"))
    print(f"S1(a): {completed}/18 complete scales; eight seeds and four solvers per scale.")


def run(task):
    assert 0<=task<len(SIZES)
    n=SIZES[task];p=RUN/"results"/f"results_scale{n:04d}.json"
    if p.exists():
        check(p,n);print(f"S1(a) N={n}: verified completed results; reused.");return
    b=backend()
    import torch
    if not torch.cuda.is_available():raise RuntimeError("S1(a) timing requires CUDA")
    b.SCALE_INDEX=str(task)
    # Record actual dependencies for new runs without changing old provenance.
    sources=[ROOT/"experiments/optimizer_scaling.py",ROOT/"configs/config.yaml"]
    sources += sorted((ROOT/"src").rglob("*.py"))
    sources += [Path(b.PHANTOM_DATA_DIR)/name for name in
                ["mu_gt_mm.npy","alpha_gt.npy","target_dB.npy","results.json"]]
    dependencies={str(p):sha(p) for p in sources}
    b.main();check(p,n)
    assert dependencies=={str(p):sha(p) for p in sources}, "Numerical inputs changed during execution"
    (RUN/"results"/f"provenance_scale{n:04d}.json").write_text(json.dumps(dependencies,indent=2)+"\n")


def export():
    prepare();STAGE.mkdir(parents=True,exist_ok=True)
    sources={}
    for n in SIZES:
        p=RUN/"results"/f"results_scale{n:04d}.json"
        check(p,n);shutil.copy2(p,STAGE/p.name);sources[p.name]=sha(p)
    (STAGE/"source_manifest.json").write_text(json.dumps(dict(
        sizes=SIZES,seeds=list(range(8)),methods=METHODS,solver_runs=576,
        source_directory=str((RUN/"results").relative_to(ROOT)),sha256=sources,
        sd_ddof=0,provenance="Original completed sweep; each file retains per-seed values and its original metadata."),indent=2)+"\n")
    (STAGE/"README.md").write_text("""# S1(a): original synthetic B-scan scaling

The S1 experiment owns these 18 scales × eight seeds × four solvers (576 fits).
Rebuild all S1 plots with `bash scripts/run_REBUTTAL_single_liquid_IL_aline.sh figures`.
Use the same launcher with `bscan` to resume or recompute missing S1(a) scales.
The shared numerical backend is `experiments/optimizer_scaling.py`; main Figure 1
has a separate axial-scaling experiment and launcher.

JSON files retain all per-seed MAPE, timing and evaluation counts. Export verifies
complete coverage, means, population SD (the original plot convention), and
wall-time/evaluation ratios. Original runs use solver-specific budgets and stopping
rules; they do not impose the liquid study's shared 20,000-evaluation cap.
""")
    print("Exported 576 verified S1(a) solver results.")
