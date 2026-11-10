#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 00:15:00
#SBATCH -J liquid_step25_plot

set -euo pipefail
task_dir="$SLURM_SUBMIT_DIR/experiments/REBUTTAL_single_liquid_IL_aline"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$task_dir/runs/tv_off_step25_20260929/logs/mpl_summary"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset DISPLAY
mkdir -p "$MPLCONFIGDIR"
/home/gridsan/amanicka/.conda/envs/diff-oct/bin/python -u "$task_dir/plot_step25.py"
