#!/bin/bash
# Single patient (02008Pre) per the paper showcase. ~270 frames split into
# 8 chunks across the array. A dependent job waits for all chunks, then
# runs ASSEMBLE to build the per-patient NIfTI and stage
# figure5_clinical showcase frames.
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 96:00:00
#SBATCH -J pat_oac
#SBATCH --array=0-7%8
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/pat_oac-%A_%a.out

set -e
module load anaconda/2023b
source activate diff-oct
cd /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal

PATIENT_ID=${PATIENT_ID:-02008Pre}
TOTAL_FRAMES=$(PATIENT_ID="$PATIENT_ID" python - <<'PYFRAMES'
import os
import nibabel as nib
from experiments.generate_patient_oac import find_patient_files
shape = nib.load(find_patient_files(os.environ["PATIENT_ID"])["volume_path"]).shape
print(shape[2] if len(shape) >= 3 else 1)
PYFRAMES
)
N_TASKS=8
PER_TASK=$(( (TOTAL_FRAMES + N_TASKS - 1) / N_TASKS ))

T=$SLURM_ARRAY_TASK_ID
START=$(( T * PER_TASK ))
END=$(( START + PER_TASK ))
if [ $END -gt $TOTAL_FRAMES ]; then END=$TOTAL_FRAMES; fi

echo "[task $T] PATIENT_ID=$PATIENT_ID frames $START..$END"
DEFER_FINALIZE=1 PATIENT_ID=$PATIENT_ID START_FRAME=$START END_FRAME=$END \
    python experiments/generate_patient_oac.py

# Finalize once, after every worker in this array succeeds.
if [ "${SLURM_ARRAY_TASK_ID:-}" = "0" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    POST_JID=$(sbatch --parsable --dependency="afterok:$SLURM_ARRAY_JOB_ID" <<EOF
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -J pat_oac_post
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/pat_oac_post-%j.out
set -e
module load anaconda/2023b
source activate diff-oct
cd /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
PATIENT_ID=$PATIENT_ID ASSEMBLE=1 python experiments/generate_patient_oac.py
EOF
)
    echo "Submitted finalization after array $SLURM_ARRAY_JOB_ID: $POST_JID"
fi

