#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 00:15:00
#SBATCH -J real_aline_plot

set -euo pipefail
TASK_DIR="$SLURM_SUBMIT_DIR/experiments/REBUTTAL_single_liquid_IL_aline"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$TASK_DIR/logs/mpl_summary"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset DISPLAY
mkdir -p "$MPLCONFIGDIR"
/home/gridsan/amanicka/.conda/envs/diff-oct/bin/python -u "$TASK_DIR/summarize.py"
