#!/bin/bash
# 8-GPU array, all 10 phantoms split by frame range. Each task processes a
# 64-frame window across every phantom, so total work divides evenly:
# 8 tasks × 64 frames × 10 phantoms = 5120 frames (= 10 × 512).
# Followed by one dependent post-processing job (SUMMARIZE + VISUALIZE_3D)
# that builds the IntMask boxplot, renders the showcase 3D cross-section,
# and stages everything into figures/data/figureS2_agar_phantom/.
#
# Usage:
#   bash scripts/run_phantom_validation_ss.sh    # submits array + dependent post job
#   sbatch scripts/run_phantom_validation_ss.sh  # equivalent (script self-resubmits)
#
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 96:00:00
#SBATCH -J pv_ss
#SBATCH --array=0-7%8
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/pv_ss-%A_%a.out

set -e

REPO=/home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
LOGDIR=$REPO/scripts/logs
mkdir -p "$LOGDIR"

# ── Submission mode: invoked outside SLURM → sbatch this script ─────────────
if [ -z "$SLURM_ARRAY_TASK_ID" ] && [ -z "$SLURM_JOB_ID" ]; then
    ARRAY_JID=$(sbatch --parsable "$0")
    echo "Submitted array job: $ARRAY_JID"
    echo "(Task 0 will submit the dependent post-processing job once it starts.)"
    exit 0
fi

# ── Worker mode: task 0 submits the SUMMARIZE + VISUALIZE_3D follow-up ──────
if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    POST_JID=$(sbatch --parsable --dependency=afterok:$SLURM_ARRAY_JOB_ID <<EOF
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -J pv_ss_post
#SBATCH -o $LOGDIR/pv_ss_post-%j.out
set -e
module load anaconda/2023b
source activate diff-oct
cd $REPO
MODEL=ss SUMMARIZE=1 python experiments/phantom_validation_ss_or_ms.py
MODEL=ss VISUALIZE_3D=1 PHANTOM_INDEX=9 python experiments/phantom_validation_ss_or_ms.py
EOF
)
    echo "[task 0] Submitted post-processing job (afterok:$SLURM_ARRAY_JOB_ID): $POST_JID"
fi

module load anaconda/2023b
source activate diff-oct
cd "$REPO"

T=$SLURM_ARRAY_TASK_ID
TOTAL_FRAMES=512
N_TASKS=8
PER_TASK=$(( TOTAL_FRAMES / N_TASKS ))   # 64
START=$(( T * PER_TASK ))
END=$(( START + PER_TASK ))

echo "[task $T] all 10 phantoms, frames $START..$((END-1))"
MODEL=ss START_FRAME=$START END_FRAME=$END \
    python experiments/phantom_validation_ss_or_ms.py
