#!/usr/bin/env bash
# Compile the standalone S7 preview and copy it next to the staged table data.
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$experiment_dir/../.." && pwd)
preview_dir="$experiment_dir/paper_preview"
mkdir -p "$preview_dir"
cd "$experiment_dir"
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$preview_dir" \
    supplementary_S7_preview.tex > "$preview_dir/pass$pass.stdout"
done
cp "$preview_dir/supplementary_S7_preview.pdf" "$project_dir/figures/data/figureS7_model_mismatch/"
printf 'Compiled %s\n' "$preview_dir/supplementary_S7_preview.pdf"
