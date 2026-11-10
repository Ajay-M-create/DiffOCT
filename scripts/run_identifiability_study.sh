#!/bin/bash
# Identifiability sweep, both phases:
#   Phase 1: 10 phantoms x 20 seeds = 200 runs
#   Phase 2: 10 init conditions x 10 phantoms x 5 seeds = 500 runs
# 8 tasks; each task loops through its share of RUN_INDEX values for the
# selected phase. Submit twice with PHASE=1 then PHASE=2:
#   sbatch --export=ALL,PHASE=1 scripts/run_identifiability_study.sh
#   sbatch --export=ALL,PHASE=2 scripts/run_identifiability_study.sh
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 96:00:00
#SBATCH -J ident
#SBATCH --array=0-7%8
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/ident-%A_%a.out

set -e
module load anaconda/2023b
source activate diff-oct
cd /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal

: ${PHASE:=1}
if [ "$PHASE" = "1" ]; then
    N_RUNS=200
elif [ "$PHASE" = "2" ]; then
    N_RUNS=500
else
    echo "PHASE must be 1 or 2 (got $PHASE)"; exit 2
fi

T=$SLURM_ARRAY_TASK_ID
for (( r=T; r<N_RUNS; r+=8 )); do
    echo "[task $T] PHASE=$PHASE RUN_INDEX=$r"
    PHASE=$PHASE RUN_INDEX=$r python experiments/identifiability_study.py
done

# Finalize once, after every worker in this array succeeds.
if [ "${SLURM_ARRAY_TASK_ID:-}" = "0" ] && [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    POST_JID=$(sbatch --parsable --dependency="afterok:$SLURM_ARRAY_JOB_ID" <<EOF
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -J ident_post
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/ident_post-%j.out
set -e
module load anaconda/2023b
source activate diff-oct
cd /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
SUMMARIZE=1 PHASE=$PHASE python experiments/identifiability_study.py
EOF
)
    echo "Submitted finalization after array $SLURM_ARRAY_JOB_ID: $POST_JID"
fi

