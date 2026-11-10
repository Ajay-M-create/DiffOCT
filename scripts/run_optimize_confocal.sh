#!/bin/bash
#SBATCH --job-name=confocal_opt
#SBATCH --output=scripts/logs/optimize_confocal_%j.log
#SBATCH --error=scripts/logs/optimize_confocal_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:volta:1

# Experiment 12: Confocal parameter optimization study
# Runs all 4 phases by default, or a single phase via PHASE env var.
#
# Usage:
#   sbatch scripts/12_OptimizeConfocal.sh              # All phases
#   PHASE=1 sbatch scripts/12_OptimizeConfocal.sh      # Phase 1 only
#   SUMMARIZE=1 sbatch scripts/12_OptimizeConfocal.sh   # Report only

source ~/.bashrc
conda activate diff-oct 2>/dev/null || true

cd /home/gridsan/groups/erelab/Ajay/DiffOCT-Phantom-Experiments

echo "========================================"
echo "Experiment 12: Confocal Optimization"
echo "PHASE=${PHASE:-0 (all)}"
echo "PHANTOM_INDEX=${PHANTOM_INDEX:-0}"
echo "BSCAN_INDEX=${BSCAN_INDEX:-256}"
echo "NUM_ITERATIONS=${NUM_ITERATIONS:-3000}"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo "========================================"

python experiments/optimize_confocal.py

echo "Exit code: $?"
echo "Done at $(date)"
