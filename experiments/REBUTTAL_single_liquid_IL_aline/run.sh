#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 01:00:00
#SBATCH -J real_aline
#SBATCH --array=0-7%8

set -euo pipefail
TASK_DIR="$SLURM_SUBMIT_DIR/experiments/REBUTTAL_single_liquid_IL_aline"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$TASK_DIR/logs/mpl_${SLURM_ARRAY_TASK_ID}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset DISPLAY
mkdir -p "$MPLCONFIGDIR"
/home/gridsan/amanicka/.conda/envs/diff-oct/bin/python -u "$TASK_DIR/benchmark.py" run --seed "$SLURM_ARRAY_TASK_ID"
