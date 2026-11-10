#!/bin/bash
# Optimize the 5 representative B-scans of one IV-OCT patient (two-stage
# mu → alpha) and stage the results to figures/data/figure5_clinical/scratch_<PID>/
#
# Defaults: 02008Pre, Pre_Post NIfTI cohort, Gargesha confocal params (55/31).
# Override any of: PATIENT_ID, PRE_PATIENTS_DIR, PRE_OAC_DIR,
# CONFOCAL_ZFP_PX, CONFOCAL_ZR_PX, NUM_ITERS, NOISE_FLOOR_DB, AXIAL_PIXEL_SIZE.
#
# Usage:
#   sbatch scripts/run_showcase_patient_frames.sh
#   PATIENT_ID=02008Pre sbatch --export=ALL scripts/run_showcase_patient_frames.sh
#
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -J fig5_show
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/fig5_show-%j.out

set -e

REPO=/home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
mkdir -p "$REPO/scripts/logs"

module load anaconda/2023b
source activate diff-oct
cd "$REPO"

# Live-flush Python stdout so progress shows up immediately in the SLURM log
export PYTHONUNBUFFERED=1

# Inputs / outputs (override via --export=ALL,VAR=value if needed)
: "${PATIENT_ID:=02008Pre}"
: "${PRE_PATIENTS_DIR:=/home/gridsan/amanicka/erelab_shared/Ajay/DiffOCT-IVLDatasetsAndSegsPrePost/Pre_Post}"
: "${POST_PATIENTS_DIR:=$PRE_PATIENTS_DIR}"
: "${PRE_OAC_DIR:=$REPO/outputs/figure5_showcase}"
: "${POST_OAC_DIR:=$PRE_OAC_DIR}"
: "${CONFOCAL_ZFP_PX:=55}"
: "${CONFOCAL_ZR_PX:=31}"
: "${BATCH_SIZE:=5}" # Optimize 5 frames
: "${NOISE_FLOOR_DB:=15.0}" # Noise floor in dB

PATIENT_ID="$PATIENT_ID" \
SHOWCASE_ONLY=1 \
BATCH_SIZE="$BATCH_SIZE" \
NOISE_FLOOR_DB="$NOISE_FLOOR_DB" \
PRE_PATIENTS_DIR="$PRE_PATIENTS_DIR" \
POST_PATIENTS_DIR="$POST_PATIENTS_DIR" \
PRE_OAC_DIR="$PRE_OAC_DIR" \
POST_OAC_DIR="$POST_OAC_DIR" \
CONFOCAL_ZFP_PX="$CONFOCAL_ZFP_PX" \
CONFOCAL_ZR_PX="$CONFOCAL_ZR_PX" \
python experiments/generate_patient_oac.py
