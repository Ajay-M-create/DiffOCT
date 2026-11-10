#!/bin/bash
# =============================================================================
# Modal CPU & GPU Benchmarking
# =============================================================================
# Runs the DiffOCT optimization routine across different Modal GPU types and
# CPU configurations.  Requires a Modal token to be configured.
#
# Usage:
#   bash scripts/15_BenchmarkingCPUandGPU.sh
#
# Environment variables:
#   NUM_ITERATIONS  -- iterations per run (default: 1000)
#   PHANTOM_INDEX   -- phantom 0-9 (default: 0)
#   BSCAN_INDEX     -- B-scan frame (default: 256)
# =============================================================================

set -euo pipefail

# Load conda environment
module load anaconda/2023b
source activate diff-oct

# Ensure modal is available
if ! command -v modal &> /dev/null; then
    echo "ERROR: 'modal' CLI not found. Install with: pip install modal"
    exit 1
fi

# Defaults (override via env vars)
export NUM_ITERATIONS="${NUM_ITERATIONS:-1000}"
export PHANTOM_INDEX="${PHANTOM_INDEX:-0}"
export BSCAN_INDEX="${BSCAN_INDEX:-256}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== DiffOCT Modal Benchmark ==="
echo "  NUM_ITERATIONS : ${NUM_ITERATIONS}"
echo "  PHANTOM_INDEX  : ${PHANTOM_INDEX}"
echo "  BSCAN_INDEX    : ${BSCAN_INDEX}"
echo ""

modal run experiments/hardware_benchmark.py
