#!/usr/bin/env bash
#SBATCH -J synthetic_aline
#SBATCH -t 12:00:00
set -euo pipefail
script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
phase=${SYNTHETIC_PHASE:-submit}
if [[ "$phase" == submit && -z "${SLURM_JOB_ID:-}" ]]; then
  repo=$(cd -- "$(dirname -- "$script_path")/.." && pwd)
else
  repo=${DIFFOCT_REPO:-${SLURM_SUBMIT_DIR:?Repository location required}}
fi
python_bin=${DIFFOCT_PYTHON:-/home/gridsan/amanicka/.conda/envs/diff-oct/bin/python}
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/diffoct-synthetic-mpl-${SLURM_JOB_ID:-local}"
unset DISPLAY
cd "$repo"
if [[ "$phase" == submit ]]; then
  case ${1:-run} in
    --help|-h) printf 'Usage: bash %s [run|figures|--dry-run]\n' "$script_path"; exit 0;;
    --dry-run) printf 'Synthetic single-A-line experiment only: 8 seed tasks, then figures/figure1_single_aline_scaling.ipynb\n'; exit 0;;
    figures) exec "$python_bin" "$repo/scripts/build_rebuttal_notebooks.py" figure1;;
    run) ;;
    *) exit 2;;
  esac
  logdir="$repo/scripts/logs";mkdir -p "$logdir"
  job=$(sbatch --parsable -p xeon-g6-volta --gres=gpu:volta:1 -c 8 --mem=16G --array=0-7%8 \
    --output="$logdir/reb_1aline-%A_%a.out" --export="ALL,SYNTHETIC_PHASE=run,DIFFOCT_REPO=$repo" "$script_path")
  job=${job%%;*}
  next=$(sbatch --parsable -p xeon-p8 -c 2 --mem=12G --time=00:40:00 --dependency="afterok:$job" \
    --output="$logdir/reb_1aline_notebook-%j.out" --export="ALL,SYNTHETIC_PHASE=figures,DIFFOCT_REPO=$repo" "$script_path")
  printf 'Submitted synthetic array %s and Figure 1 notebook %s\n' "$job" "$next"
elif [[ "$phase" == run ]]; then
  SEED_INDEX=${SLURM_ARRAY_TASK_ID:?} NUM_ITERS=20000 "$python_bin" -u experiments/REBUTTAL_single_aline_scaling.py
elif [[ "$phase" == figures ]]; then
  exec "$python_bin" scripts/build_rebuttal_notebooks.py figure1
else
  printf 'Unknown phase: %s\n' "$phase" >&2;exit 2
fi
