#!/bin/bash
#SBATCH --job-name=22_all_pre_psnr
#SBATCH --output=scripts/logs/all_pre_psnr_%A_%a.out
#SBATCH --time=96:00:00
#SBATCH --array=0-7
#SBATCH -n 1
#SBATCH --gres=gpu:volta:1
#SBATCH --signal=B:USR1@120

# ============================================================================
# Paper-config PSNR across all 128 Pre-IVL pullbacks, 8 GPUs in parallel.
#
# PATIENT-SPLIT: each array task (= 1 GPU) processes its own disjoint slice of
# patients end-to-end.  A whole ~170-frame pullback fits on one 32 GB Volta
# (~5 GB per batch at 30 MB/frame), so there is no benefit to splitting frames
# within a patient — we just send the entire pullback through
# shockwave_arteries_batch_gpu.BatchedSSForward + batched_optimize in one shot.
# No barriers, no task-0 assembly gate.
#
# Config (matches test/20_render_IVOCT.py):
#   NOISE_FLOOR_DB       0.0
#   AXIAL_PIXEL_SIZE_MM  0.01932
#   NUM_ITERS_S1         25000  (mu, alpha=1.0)
#   NUM_ITERS_S2         25000  (alpha, mu frozen)
#   mu_tv_weight         0
#   mu_l1_weight         0
#   CONFOCAL_ZFP / ZR    55 / 31 (Gargesha 2015 measured values: z_0=1.06 mm, z_R=0.59 mm)
#
# Auto-resubmits on SIGUSR1 (120s before time limit).  Individual frames with
# outputs/22_all_pre_render_psnr/{PID}/frame_NNN/psnr.npz are skipped and
# per-patient psnr_summary.npz gates patient-level skipping, so resumption is
# safe.
#
# Usage:
#   sbatch scripts/22_AllPreIVLRenderPsnr.sh
#
# After all patients are done the last task to finish calls SUMMARIZE=1; or
# run it manually:
#   SUMMARIZE=1 python experiments/all_pre_ivl_render_psnr.py
# ============================================================================

source /etc/profile 2>/dev/null || true
module unload anaconda 2>/dev/null || true
module load anaconda/2023b 2>/dev/null || module load anaconda/2023a-pytorch 2>/dev/null || true

if command -v conda &> /dev/null; then
    conda activate diff-oct 2>/dev/null || source activate diff-oct 2>/dev/null || true
else
    source activate diff-oct 2>/dev/null || true
fi

# Project root
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/src" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT" || { echo "ERROR: cannot cd to $PROJECT_ROOT"; exit 1; }
mkdir -p scripts/logs

export HYDRA_FULL_ERROR=1

# ---- Config ----
NUM_TASKS=8
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

export NOISE_FLOOR_DB="${NOISE_FLOOR_DB:-0.0}"
export AXIAL_PIXEL_SIZE_MM="${AXIAL_PIXEL_SIZE_MM:-0.01932}"
export NUM_ITERS_S1="${NUM_ITERS_S1:-25000}"
export NUM_ITERS_S2="${NUM_ITERS_S2:-25000}"
export DB_SCALE="${DB_SCALE:-130.0}"
export CONFOCAL_ZFP_PX="${CONFOCAL_ZFP_PX:-55}"
export CONFOCAL_ZR_PX="${CONFOCAL_ZR_PX:-31}"

PATIENT_DIR="/home/gridsan/amanicka/erelab_shared/Ajay/DiffOCT-IVLDatasetsAndSegsPrePost/Pre_Post"
OUT_ROOT="$PROJECT_ROOT/outputs/22_all_pre_render_psnr"
export OUT_ROOT
mkdir -p "$OUT_ROOT"

# ---- Resubmit on SIGUSR1 (task 0 only, to avoid 8-way duplication) ----
resubmit_handler() {
    echo ""
    echo "=================================================================="
    echo "TIMEOUT: SIGUSR1 received on task $TASK_ID"
    echo "=================================================================="
    if [ "$TASK_ID" -eq 0 ]; then
        NEW_JOB=$(sbatch --parsable "$PROJECT_ROOT/scripts/22_AllPreIVLRenderPsnr.sh")
        echo "Resubmitted as job $NEW_JOB"
    fi
    exit 0
}
trap 'resubmit_handler' USR1

python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null \
    || echo "WARNING: no GPU on task $TASK_ID"

# ---- Discover Pre patient IDs ----
mapfile -t PATIENTS < <(python3 -c "
import glob, os, re
files = sorted(glob.glob(os.path.join('${PATIENT_DIR}', '*Pre*.nii.gz')))
seen = set()
for f in files:
    base = os.path.basename(f)
    idx = base.find('Pre')
    if idx < 0: continue
    pid = base[:idx+3]
    if pid not in seen:
        seen.add(pid); print(pid)
")

NUM_PATIENTS=${#PATIENTS[@]}
if [ "$NUM_PATIENTS" -eq 0 ]; then
    echo "ERROR: no Pre patients found under $PATIENT_DIR"
    exit 1
fi

# ---- Compute this task's patient slice (patient-split) ----
PATIENTS_PER_TASK=$(( (NUM_PATIENTS + NUM_TASKS - 1) / NUM_TASKS ))
SLICE_START=$(( TASK_ID * PATIENTS_PER_TASK ))
SLICE_END=$(( SLICE_START + PATIENTS_PER_TASK ))
[ "$SLICE_END" -gt "$NUM_PATIENTS" ] && SLICE_END="$NUM_PATIENTS"

echo ""
echo "=================================================================="
echo "22_AllPreIVLRenderPsnr  (PATIENT-SPLIT)  —  Task $TASK_ID / $NUM_TASKS"
echo "=================================================================="
echo "Patients total:       $NUM_PATIENTS"
echo "Patients this task:   [$SLICE_START, $SLICE_END)  ($((SLICE_END - SLICE_START)) patients)"
echo "Noise floor:          $NOISE_FLOOR_DB dB"
echo "Axial pixel size:     $AXIAL_PIXEL_SIZE_MM mm"
echo "Iterations s1 / s2:   $NUM_ITERS_S1 / $NUM_ITERS_S2"
echo "Confocal ZFP / ZR:    $CONFOCAL_ZFP_PX / $CONFOCAL_ZR_PX px"
echo "Output root:          $OUT_ROOT"
echo "Patient dir:          $PATIENT_DIR"
echo "=================================================================="

if [ "$SLICE_START" -ge "$NUM_PATIENTS" ]; then
    echo "Task $TASK_ID: no patients to process."
    exit 0
fi

COMPLETED=0
SKIPPED=0

for P_IDX in $(seq $SLICE_START $((SLICE_END - 1))); do
    PATIENT_ID="${PATIENTS[$P_IDX]}"
    PATIENT_OUT="$OUT_ROOT/$PATIENT_ID"

    if [ -f "$PATIENT_OUT/psnr_summary.npz" ]; then
        echo "[$((P_IDX + 1))/$NUM_PATIENTS] SKIP (done): $PATIENT_ID"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo ""
    echo "[$((P_IDX + 1))/$NUM_PATIENTS] ============= $PATIENT_ID ============="
    T_PATIENT_START=$(date +%s)

    # Whole-patient worker: no START_FRAME/END_FRAME means process everything.
    # Worker internally batches frames via compute_batch_size + BatchedSSForward.
    unset START_FRAME END_FRAME
    export PATIENT_ID
    python3 experiments/all_pre_ivl_render_psnr.py
    if [ $? -ne 0 ]; then
        echo "  ERROR: worker failed on $PATIENT_ID"
        continue
    fi

    # Assemble per-patient NIfTIs + summary
    export PATIENT_ID ASSEMBLE=1
    python3 experiments/all_pre_ivl_render_psnr.py
    unset ASSEMBLE

    COMPLETED=$((COMPLETED + 1))
    T_ELAPSED=$(( $(date +%s) - T_PATIENT_START ))
    echo "  $PATIENT_ID done in ${T_ELAPSED}s"
done

echo ""
echo "=================================================================="
echo "Task $TASK_ID DONE: $COMPLETED completed, $SKIPPED skipped"
echo "=================================================================="

# ---- Global summary: only run when this task sees every patient finished ----
# Simple check: count per-patient psnr_summary.npz against discovered patients.
ALL_DONE_COUNT=$(ls "$OUT_ROOT"/*/psnr_summary.npz 2>/dev/null | wc -l)
if [ "$ALL_DONE_COUNT" -ge "$NUM_PATIENTS" ]; then
    echo ""
    echo "All $NUM_PATIENTS patients have psnr_summary.npz — running global summary."
    export SUMMARIZE=1
    python3 experiments/all_pre_ivl_render_psnr.py
fi
