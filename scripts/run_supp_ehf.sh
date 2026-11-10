#!/bin/bash
# Single Volta. Reads outputs/phantom_validation_{ss,ms}/ (must already be
# populated) plus runs the forward-model comparison.
# inverse optimization (panel c). Stages figures/data/figureS5_ss_vs_ms/.
# Total runtime is ~5-15 min — the synthetic SS+MS inversion is the bulk.
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -J supp_ehf
#SBATCH -o /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal/scripts/logs/supp_ehf-%j.out

set -e
module load anaconda/2023b
source activate diff-oct
cd /home/gridsan/groups/erelab/Ajay/DiffOCT-rebuttal

export PYTHONUNBUFFERED=1
python -u experiments/supp_ehf.py
