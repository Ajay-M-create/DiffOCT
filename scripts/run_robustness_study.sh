#!/bin/bash
# 290 runs total (10 phantoms x 29 noise slices). 8 tasks; each loops through
# its share of RUN_INDEX values (stride 8). After workers finish, SUMMARIZE
# builds figure4_identifiability_robustness panels e-g.
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 96:00:00
#SBATCH -J robust
#SBATCH --array=0-7%8
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/robust-%A_%a.out

set -e
module load anaconda/2023b
source activate diff-oct
cd /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal

N_RUNS=290
T=$SLURM_ARRAY_TASK_ID
for (( r=T; r<N_RUNS; r+=8 )); do
    echo "[task $T] RUN_INDEX=$r"
    RUN_INDEX=$r python experiments/robustness_study.py
done

# Finalize once, after every worker in this array succeeds.
if [ "${SLURM_ARRAY_TASK_ID:-}" = "0" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    POST_JID=$(sbatch --parsable --dependency="afterok:$SLURM_ARRAY_JOB_ID" <<EOF
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -J robust_post
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/robust_post-%j.out
set -e
module load anaconda/2023b
source activate diff-oct
cd /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
SUMMARIZE=1 python experiments/robustness_study.py
EOF
)
    echo "Submitted finalization after array $SLURM_ARRAY_JOB_ID: $POST_JID"
fi

