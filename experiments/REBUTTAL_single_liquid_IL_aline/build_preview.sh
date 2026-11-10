#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$experiment_dir/../.." && pwd)
paper_dir="$project_dir/paper/DiffOCT_CommsEngRevisions"
preview_dir="$experiment_dir/paper_preview"
mkdir -p "$preview_dir"
cd "$paper_dir"
for pass in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$preview_dir" \
    "$experiment_dir/supplementary_S1_preview.tex" > "$preview_dir/pass$pass.stdout"
done
printf 'Compiled %s\n' "$preview_dir/supplementary_S1_preview.pdf"
