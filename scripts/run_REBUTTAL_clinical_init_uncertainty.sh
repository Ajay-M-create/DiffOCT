#!/bin/bash
# Five Figure 5 frames: 20 seeds plus ten initialization conditions x five seeds.
# Submit a two-task array, followed by CPU staging and the single S8 composite.
# Usage: bash scripts/run_REBUTTAL_clinical_init_uncertainty.sh
#        bash scripts/run_REBUTTAL_clinical_init_uncertainty.sh figures
# Env: RUN_NAME, NUM_ITERS (25000), REPEATS_PER_BATCH (10).
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J reb_init_unc
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/reb_init_unc-%A_%a.out
set -euo pipefail
REPO=/home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
PY=/home/gridsan/amanicka/.conda/envs/diff-oct/bin/python
mkdir -p "$REPO/scripts/logs"
cd "$REPO"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=8
export RUN_NAME=${RUN_NAME:-five_frames_$(date +%Y%m%dT%H%M%S)}
RUN_DIR=$REPO/experiments/REBUTTAL_clinical_init_uncertainty/runs/$RUN_NAME
if [ "${1:-}" = "figures" ]; then
    "$PY" scripts/build_rebuttal_notebooks.py s8
elif [ "${1:-}" = "stage" ]; then
    "$PY" experiments/REBUTTAL_clinical_init_uncertainty/init_uncertainty.py stage "$RUN_DIR"
    "$PY" scripts/build_rebuttal_notebooks.py s8
elif [ -z "${SLURM_JOB_ID:-}" ]; then
    SWEEP_JOB=$(sbatch --parsable --array=0-1 --export=ALL "$0")
    echo "Sweep job: $SWEEP_JOB; RUN_NAME=$RUN_NAME"
    sbatch --dependency="afterok:$SWEEP_JOB" --export=ALL --job-name=s8_stage "$0" stage
else
    MODES=(seed init)
    "$PY" -u experiments/REBUTTAL_clinical_init_uncertainty/init_uncertainty.py "${MODES[$SLURM_ARRAY_TASK_ID]}" "$RUN_DIR"
fi
