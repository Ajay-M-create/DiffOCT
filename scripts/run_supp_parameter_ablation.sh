#!/bin/bash
# Eight phases: S3a uses 1-4 and S3b uses 5-8, one phase per array task
# (max 8 GPUs concurrent). Each phase iterates over its conditions and 10
# phantoms internally via subprocess calls. After all workers finish each
# one dependent SUMMARIZE job stages both S3a and S3b.
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 96:00:00
#SBATCH -J pab
#SBATCH --array=0-7%8
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/pab-%A_%a.out

set -e
module load anaconda/2023b
source activate diff-oct
cd /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal

declare -a PHASES_FOR_TASK
PHASES_FOR_TASK[0]="1"
PHASES_FOR_TASK[1]="2"
PHASES_FOR_TASK[2]="3"
PHASES_FOR_TASK[3]="4"
PHASES_FOR_TASK[4]="5"
PHASES_FOR_TASK[5]="6"
PHASES_FOR_TASK[6]="7"
PHASES_FOR_TASK[7]="8"

T=$SLURM_ARRAY_TASK_ID
echo "[task $T] phases: ${PHASES_FOR_TASK[$T]}"
for PHASE in ${PHASES_FOR_TASK[$T]}; do
    echo "[task $T] running phase $PHASE"
    PHASE=$PHASE python experiments/supp_parameter_ablation.py
done

# Finalize once, after every worker in this array succeeds.
if [ "${SLURM_ARRAY_TASK_ID:-}" = "0" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    POST_JID=$(sbatch --parsable --dependency="afterok:$SLURM_ARRAY_JOB_ID" <<EOF
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -J pab_post
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/pab_post-%j.out
set -e
module load anaconda/2023b
source activate diff-oct
cd /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
SUMMARIZE=1 python experiments/supp_parameter_ablation.py
EOF
)
    echo "Submitted finalization after array $SLURM_ARRAY_JOB_ID: $POST_JID"
fi

