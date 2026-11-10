#!/bin/bash
# Forward-model mismatch table (reviewer: robustness beyond same-model inversion).
#
# One GPU job: simulate 10 mismatch scenarios x 5 realizations on a digital twin of
# the 1% liquid Intralipid B-scan 256, invert all 50 B-scans in one batch with the
# nominal single-scattering model (15k iterations, gain re-estimated each iteration),
# write the table and stage figures/data/figureS7_model_mismatch/.
#
# Usage (from anywhere):
#   bash scripts/run_REBUTTAL_model_mismatch.sh      # submits itself
#   sbatch scripts/run_REBUTTAL_model_mismatch.sh    # equivalent
#
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J reb_mismatch
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/reb_mismatch-%j.out

set -euo pipefail
REPO=/home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
mkdir -p "$REPO/scripts/logs"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    sbatch "$0"
    exit 0
fi

module load anaconda/2023b
source activate diff-oct
cd "$REPO"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=8
RUN_DIR=$REPO/experiments/REBUTTAL_model_mismatch/runs/${RUN_NAME:-table_$(date +%Y%m%d)}
python -u experiments/REBUTTAL_model_mismatch/mismatch.py all "$RUN_DIR"
