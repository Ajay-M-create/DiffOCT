#!/bin/bash
# Liquid Intralipid rebuttal pipeline: DiffOCT vs conventional curve fitting (CF)
# on 5 concentrations (0.5, 1, 1.5, 2, 3%) x 2 acquisitions (H, L) x 512 B-scans
# = 5,120 frames, full-depth single-scattering fit, fixed per-A-line CF gain,
# 15,000 iterations. One call submits the whole chain as dependent Slurm jobs:
#
#   1 create      CPU        production.py create      new series folder, code snapshot,
#                                                      manifest, input audit
#   2 preflight   1 GPU      production.py preflight   64-frame memory / batch check
#   3 workers     8 GPUs     production.py worker      array 0-7, 64 frames per GPU per volume
#   4 diagnostic  1 GPU      production.py diagnostic  60k-iteration spot check (3% H, frame 220)
#   5 report      CPU        production_report.py      regressions, violins, offsets, summaries
#   6 post        CPU        production_diagnostics.py residuals, attenuation breakdown,
#                                                      artifact flags, example A-lines
#   7 figure      CPU        (only with STAGE_FIGURE=1) plot_selected_alines.py, raw 3D volume
#                            (render_3d_liquid.py) and staging of the figure inputs into
#                            figures/data/figure3_liquid_phantom/ (overwrites them)
#
# Steps 2-6 run from the series' frozen code snapshot (SERIES_DIR/code). Step 4 runs
# alongside step 3; step 5 waits for both (afterany); step 6 needs step 5 to succeed.
#
# Usage (from anywhere; submits jobs and exits):
#   bash scripts/run_REBUTTAL_liquid_intralipid_production.sh [SERIES_DIR]
#   STAGE_FIGURE=1 bash scripts/run_REBUTTAL_liquid_intralipid_production.sh
# SERIES_DIR defaults to a new timestamped folder under the experiment's runs/.
# It must not exist yet (create refuses to overwrite).
#
# After step 7, rebuild the 3D panel renders by running
# figures/figure3_liquid_phantom.ipynb with RERENDER_3D=1 on a compute node
# and copying the render_*.jpg/.pdf files it writes into the figure data folder.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EXP="$REPO/experiments/REBUTTAL_liquid_intralipid_cf_comparison"
PY=/home/gridsan/amanicka/.conda/envs/diff-oct/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SERIES_DIR="${1:-$EXP/runs/production_HL_full_depth_cf_gain_$STAMP}"
SERIES_DIR=$(realpath -m "$SERIES_DIR")
STAGE_FIGURE="${STAGE_FIGURE:-0}"
FIGDATA="$REPO/figures/data/figure3_liquid_phantom"
LOG="$EXP/runs/logs_$(basename "$SERIES_DIR")"

if [[ -e "$SERIES_DIR" ]]; then
    echo "Series folder already exists: $SERIES_DIR" >&2
    exit 1
fi
mkdir -p "$LOG"
echo "Series: $SERIES_DIR"
echo "Logs:   $LOG"

# Common environment for every job; MPLCONFIGDIR is set per job below.
ENV_SETUP="set -euo pipefail
export OMP_NUM_THREADS=4
unset DISPLAY"

submit() {  # submit <name> <dependency or ''> ; job script on stdin
    local name=$1 dep=$2
    sbatch --parsable ${dep:+--dependency=$dep} -o "$LOG/$name-%j.out" -J "liquid_$name"
}

# 1 create: new series, frozen code snapshot, manifest (CPU; hashes the input TIFFs)
CREATE=$(submit create '' <<EOF
#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
$ENV_SETUP
export MPLCONFIGDIR="$LOG/mpl_create"; mkdir -p "\$MPLCONFIGDIR"
cd "$EXP"
"$PY" -u production.py create "$SERIES_DIR"
EOF
)

# 2 preflight: 64-frame GPU memory and batch-independence check
PREFLIGHT=$(submit preflight "afterok:$CREATE" <<EOF
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 00:20:00
$ENV_SETUP
export MPLCONFIGDIR="$SERIES_DIR/cache/mpl_preflight"; mkdir -p "\$MPLCONFIGDIR"
"$PY" -u "$SERIES_DIR/code/production.py" preflight "$SERIES_DIR"
EOF
)

# 3 workers: eight GPUs, each fits one 64-frame block of every volume
WORKERS=$(submit workers "afterok:$PREFLIGHT" <<EOF
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 04:00:00
#SBATCH --array=0-7%8
$ENV_SETUP
export MPLCONFIGDIR="$SERIES_DIR/cache/mpl_worker_\${SLURM_ARRAY_TASK_ID}"; mkdir -p "\$MPLCONFIGDIR"
"$PY" -u "$SERIES_DIR/code/production.py" worker "$SERIES_DIR" --worker "\$SLURM_ARRAY_TASK_ID"
EOF
)

# 4 diagnostic: 60,000-iteration convergence spot check, in parallel with the workers
DIAGNOSTIC=$(submit diagnostic "afterok:$PREFLIGHT" <<EOF
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 00:30:00
$ENV_SETUP
export MPLCONFIGDIR="$SERIES_DIR/cache/mpl_diagnostic"; mkdir -p "\$MPLCONFIGDIR"
"$PY" -u "$SERIES_DIR/code/production.py" diagnostic "$SERIES_DIR"
EOF
)

# 5 report: regressions, violins and summaries (a failed diagnostic does not block it)
REPORT=$(submit report "afterany:$WORKERS:$DIAGNOSTIC" <<EOF
#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
$ENV_SETUP
export MPLCONFIGDIR="$SERIES_DIR/cache/mpl_report"; mkdir -p "\$MPLCONFIGDIR"
"$PY" -u "$SERIES_DIR/code/production_report.py" "$SERIES_DIR"
EOF
)

# 6 post: residuals, attenuation breakdown, artifact flags, example A-lines
POST=$(submit post "afterok:$REPORT" <<EOF
#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
$ENV_SETUP
export MPLCONFIGDIR="$SERIES_DIR/cache/mpl_post"; mkdir -p "\$MPLCONFIGDIR"
"$PY" -u "$SERIES_DIR/code/production_diagnostics.py" "$SERIES_DIR"
EOF
)

echo "create $CREATE -> preflight $PREFLIGHT -> workers $WORKERS + diagnostic $DIAGNOSTIC -> report $REPORT -> post $POST"

# 7 figure inputs (opt-in): selected A-lines, raw 3D volume, staging into figures/data
if [[ "$STAGE_FIGURE" == "1" ]]; then
    FIGURE=$(submit figure "afterok:$POST" <<EOF
#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:45:00
$ENV_SETUP
export MPLCONFIGDIR="$SERIES_DIR/cache/mpl_figure"; mkdir -p "\$MPLCONFIGDIR"
export PYVISTA_OFF_SCREEN=true VTK_SMP_MAX_THREADS=8
cd "$EXP"
# Panel (g): the four hand-picked A-lines (1% H 23, 3% H 491, 2% L 398, 3% L 179; frame 256)
"$PY" -u plot_selected_alines.py "$SERIES_DIR"
# Panels (d)/(e): raw 3D volume for 3% H, 225-pixel band below the CF-window start
SERIES="$SERIES_DIR" ACQ=H CONC=3 DEPTH_PIXELS=225 RENDER_IMAGES=0 "$PY" -u render_3d_liquid.py
# Panels (f)/(g): stage the violin and A-line inputs used by the figure notebook
"$PY" - <<'PYEOF'
import hashlib, json, shutil
from pathlib import Path
series, figdata = Path("$SERIES_DIR"), Path("$FIGDATA")
pairs = {"violin_frame_means.npz": series / "display_membership.npz",
         "violin_regressions.csv": series / "regressions.csv",
         "aline_signals_selected.npz": series / "post_diagnostics" / "selected_aline_signals.npz",
         "aline_metrics_selected.csv": series / "post_diagnostics" / "selected_alines.csv"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
record = json.loads((figdata / "staged_sources.json").read_text())
for name, src in pairs.items():
    shutil.copy2(src, figdata / name)
    record["files"][name] = dict(source=str(src), sha256=sha(figdata / name), source_sha256=sha(src))
record["note"] = f"Copies staged for figures/figure3_liquid_phantom.ipynb from {series}."
(figdata / "staged_sources.json").write_text(json.dumps(record, indent=2) + "\n")
print("staged", list(pairs))
PYEOF
EOF
    )
    echo "figure $FIGURE (after post): selected A-lines, raw 3D volume, staging into $FIGDATA"
fi
