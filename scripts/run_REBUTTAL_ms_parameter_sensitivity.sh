#!/bin/bash
# Sensitivity of clinical mu to the assumed EHF parameters g and n_sample (reviewer R3.8, Figure S6).
#
# One GPU job: re-invert the five Figure 5 showcase frames (patient 02008Pre, frames
# 0, 42, 85, 127, 170) with the MS (EHF) forward model, sweeping g (n_sample = 1.35) and
# n_sample (g = 0.7), plus one SS reference per frame, using the published clinical
# pipeline (stage 1, 25,000 iterations). Stages figures/data/figureS6_ms_parameter_sensitivity/
# and executes figures/figureS5_ss_vs_ms.ipynb (figureS5a + figureS6 in figures/output/).
#
# Usage (from anywhere):
#   bash scripts/run_REBUTTAL_ms_parameter_sensitivity.sh           # submits itself
#   bash scripts/run_REBUTTAL_ms_parameter_sensitivity.sh figures   # rebuild the notebook only
# Env: FRAMES ("0,42,85,127,170"), NUM_ITERS (25000), RUN_NAME.
#
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J reb_ms_sens
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/reb_ms_sens-%j.out

set -euo pipefail
REPO=/home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
PY=/home/gridsan/amanicka/.conda/envs/diff-oct/bin/python
mkdir -p "$REPO/scripts/logs"
cd "$REPO"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=8

if [ "${1:-}" = "figures" ]; then
    "$PY" scripts/build_rebuttal_notebooks.py s5
    exit 0
fi
if [ -z "${SLURM_JOB_ID:-}" ]; then
    sbatch --export=ALL "$0"
    exit 0
fi

RUN_DIR=$REPO/experiments/REBUTTAL_ms_parameter_sensitivity/runs/${RUN_NAME:-sweep_$(date +%Y%m%dT%H%M%S)}
"$PY" -u experiments/REBUTTAL_ms_parameter_sensitivity/ms_sensitivity.py all "$RUN_DIR"
"$PY" scripts/build_rebuttal_notebooks.py s5
