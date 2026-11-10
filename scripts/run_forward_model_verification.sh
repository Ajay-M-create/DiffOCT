#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 96:00:00
#SBATCH -J fwd_verify
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/fwd_verify-%j.out

set -e
module load anaconda/2023b
source activate diff-oct
cd /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal
python experiments/forward_model_verification.py
