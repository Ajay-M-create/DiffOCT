#!/bin/bash
# 8-GPU array, one seed per task. Each task runs the full 18-scale sweep
# (ALINE_SCALES = [1, 2, 3, 5, 10, 15, 25, 50, 75, 100, 150, 200, 250, 300,
# 350, 400, 450, 500]) for its single seed at NUM_ITERS=20000. After all
# 8 tasks finish, the dependent merge job aggregates per-seed shards into
# per-scale results_scale{N:04d}.json files and stages figureS1_optimizer_scaling/bscan.
#
# Usage:
#   bash scripts/run_optimizer_scaling.sh    # submits array + dependent merge
#   sbatch scripts/run_optimizer_scaling.sh  # equivalent (script self-resubmits)
#
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 96:00:00
#SBATCH -J opt_scale
#SBATCH --array=0-7%8
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/opt_scale-%A_%a.out

set -e

REPO=/home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
LOGDIR=$REPO/scripts/logs
mkdir -p "$LOGDIR"

# ── Submission mode: invoked outside SLURM → sbatch this script ─────────────
if [ -z "$SLURM_ARRAY_TASK_ID" ] && [ -z "$SLURM_JOB_ID" ]; then
    ARRAY_JID=$(sbatch --parsable "$0")
    echo "Submitted array job: $ARRAY_JID"
    echo "(Task 0 will submit the dependent merge job once it starts.)"
    exit 0
fi

# ── Worker mode: task 0 submits the per-seed merge follow-up ────────────────
if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    # Heredoc rather than --wrap so the post-job runs under bash, not sh.
    MERGE_JID=$(sbatch --parsable --dependency=afterok:$SLURM_ARRAY_JOB_ID <<EOF
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -J opt_scale_merge
#SBATCH -o $LOGDIR/opt_scale_merge-%j.out
set -e
module load anaconda/2023b
source activate diff-oct
cd $REPO
MERGE_PER_SEED=1 python experiments/optimizer_scaling.py
EOF
)
    echo "[task 0] Submitted merge job (afterok:$SLURM_ARRAY_JOB_ID): $MERGE_JID"
fi

module load anaconda/2023b
source activate diff-oct
cd "$REPO"

# Unbuffered so log lines flush in real time (otherwise stdout/stderr stay
# in libc buffers until the process exits → log file looks frozen for hours).
export PYTHONUNBUFFERED=1

T=$SLURM_ARRAY_TASK_ID
echo "[task $T] SEED_INDEX=$T sweeping all 18 ALINE_SCALES at 20k iters"
SEED_INDEX=$T NUM_ITERS=20000 python -u experiments/optimizer_scaling.py
