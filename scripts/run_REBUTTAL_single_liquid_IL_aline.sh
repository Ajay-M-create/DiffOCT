#!/usr/bin/env bash
#SBATCH -J supplement_s1
#SBATCH -t 00:40:00
set -euo pipefail
script_path=$(readlink -f -- "${BASH_SOURCE[0]}")
phase=${LIQUID_PHASE:-submit}
if [[ "$phase" == submit && -z "${SLURM_JOB_ID:-}" ]]; then
  repo=$(cd -- "$(dirname -- "$script_path")/.." && pwd)
else
  repo=${DIFFOCT_REPO:-${SLURM_SUBMIT_DIR:?Repository location required}}
fi
python_bin=${DIFFOCT_PYTHON:-/home/gridsan/amanicka/.conda/envs/diff-oct/bin/python}
task_dir="$repo/experiments/REBUTTAL_single_liquid_IL_aline"
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/diffoct-liquid-mpl-${SLURM_JOB_ID:-local}"
unset DISPLAY
cd "$repo"
if [[ "$phase" == submit ]]; then
  study=${1:-all}
  case "$study" in
    --help|-h) printf 'Usage: bash %s [all|bscan|scaling|coupling|figures|validate] [--dry-run]\n' "$script_path"; exit 0;;
    all|bscan|scaling|coupling|figures|validate) ;;
    *) printf 'Unknown study: %s\n' "$study" >&2; exit 2;;
  esac
  if [[ ${2:-} == --dry-run ]]; then
    printf 'S1 experiment: %s; completed matching results are reused.\n' "$study"
    for part in bscan scaling coupling; do
      if [[ "$study" != all && "$study" != "$part" ]]; then continue; fi
      case "$part" in
        bscan) printf 'S1(a): 18 tasks (one per scale), 8 seeds x 4 solvers each; 96h/task.\n';;
        scaling) printf 'S1(b): 32 tasks, 608 liquid A-line fits; 40min/task.\n';;
        coupling) printf 'S1(c): 23 tasks, 15 B-scan fits + 8 separate A-line controls; 40min/task.\n';;
      esac
    done
    printf 'Final build: S1a, S1b, S1c notebooks, composite figure, S1 text and PDF preview.\n'
    exit 0
  fi
  if [[ "$study" == figures ]]; then
    exec "$python_bin" "$task_dir/experiment.py" figures
  fi
  if [[ "$study" == validate ]]; then
    "$python_bin" "$task_dir/experiment.py" prepare
    exec "$python_bin" "$task_dir/experiment.py" validate
  fi
  "$python_bin" "$task_dir/experiment.py" prepare --study "$study"
  "$python_bin" "$task_dir/experiment.py" validate --study "$study"
  mkdir -p "$task_dir/logs"
  dependencies=()
  for part in bscan scaling coupling; do
    if [[ "$study" != all && "$study" != "$part" ]]; then continue; fi
    case "$part" in
      bscan) task_array=0-17%8; task_time=96:00:00; task_mem=32G;;
      scaling) task_array=0-31%8; task_time=00:40:00; task_mem=12G;;
      coupling) task_array=0-22%8; task_time=00:40:00; task_mem=12G;;
    esac
    job=$(sbatch --parsable -p xeon-g6-volta --gres=gpu:volta:1 -c 4 --mem="$task_mem" --time="$task_time" \
      --array="$task_array" --output="$task_dir/logs/${part}-%A_%a.out" \
      --export="ALL,LIQUID_PHASE=$part,DIFFOCT_REPO=$repo" "$script_path")
    dependencies+=("${job%%;*}")
    printf 'Submitted S1 %s job %s\n' "$part" "$job"
  done
  dependency=$(IFS=:; echo "${dependencies[*]}")
  job=$(sbatch --parsable -p xeon-p8 -c 2 --mem=12G --dependency="afterok:$dependency" \
    --output="$task_dir/logs/notebook-%j.out" \
    --export="ALL,LIQUID_PHASE=figures,DIFFOCT_REPO=$repo" "$script_path")
  printf 'Submitted supplementary notebook job %s\n' "$job"
else
  case "$phase" in
    bscan|scaling|coupling) exec "$python_bin" "$task_dir/experiment.py" run --study "$phase" --task "${SLURM_ARRAY_TASK_ID:?}";;
    figures) exec "$python_bin" "$task_dir/experiment.py" figures;;
    *) printf 'Unknown phase: %s\n' "$phase" >&2; exit 2;;
  esac
fi
