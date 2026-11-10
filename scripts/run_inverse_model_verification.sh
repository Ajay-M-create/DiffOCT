#!/bin/bash
# 5-GPU array: each task handles 100 phantoms (500 total / 5 GPUs).
# Followed by a single dependent summarize job that aggregates snapshots.
#
# Usage:
#   bash scripts/run_inverse_model_verification.sh   # submits array + dependent summarize
#   sbatch scripts/run_inverse_model_verification.sh # equivalent (script self-resubmits)
#
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 96:00:00
#SBATCH -J inv_verify
#SBATCH --array=0-4%5
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/inv_verify-%A_%a.out

set -e

REPO=/home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
LOGDIR=$REPO/scripts/logs
mkdir -p "$LOGDIR"

# ── Submission mode: invoked outside SLURM → just sbatch this script ────────
# Task 0 of the resulting array will submit the dependent summarize job.
if [ -z "$SLURM_ARRAY_TASK_ID" ] && [ -z "$SLURM_JOB_ID" ]; then
    ARRAY_JID=$(sbatch --parsable "$0")
    echo "Submitted array job: $ARRAY_JID"
    echo "(Task 0 will submit the dependent summarize job once it starts.)"
    exit 0
fi

# ── Worker mode: this is a SLURM array task ─────────────────────────────────
# Task 0 submits the summarize job with afterok dependency on the whole array.
# This fires whether the array was launched via `bash run_*.sh` (which sbatch's
# this script) or directly via `sbatch run_*.sh`.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    SUMM_JID=$(sbatch --parsable --dependency=afterok:$SLURM_ARRAY_JOB_ID <<EOF
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -J inv_verify_summ
#SBATCH -o $LOGDIR/inv_verify_summ-%j.out
set -e
module load anaconda/2023b
source activate diff-oct
cd $REPO
SUMMARIZE=1 python experiments/inverse_model_verification.py
EOF
)
    echo "[task 0] Submitted summarize job (afterok:$SLURM_ARRAY_JOB_ID): $SUMM_JID"
fi

module load anaconda/2023b
source activate diff-oct
cd "$REPO"

T=$SLURM_ARRAY_TASK_ID
START=$(( T * 100 ))
END=$(( START + 100 ))

echo "[task $T] phantoms $START..$((END-1))"
START_INDEX=$START END_INDEX=$END python experiments/inverse_model_verification.py
