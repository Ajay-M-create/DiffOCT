# Main Figure 1: synthetic single-A-line axial scaling

This is a standalone experiment, separate from the liquid-phantom supplementary study.

- Runner: `experiments/REBUTTAL_single_aline_scaling.py`
- Launcher: `scripts/run_REBUTTAL_single_aline_scaling.sh`
- Notebook: `figures/figure1_single_aline_scaling.ipynb`
- Raw results: `outputs/rebuttal_single_aline_scaling/`
- Staged data: `figures/data/figure1_single_aline_scaling/`
- Final figure: `figures/output/figure1.pdf` (also JPEG and PNG)

The experiment increases N = 25, 50, 100, 200, 500, 1000, 2000, 5000 attenuation
parameters within one A-line, retaining a physical tissue thickness of 0.16 mm.
Four solvers use eight paired initializations per size. MAPE compares recovered
attenuation with known synthetic ground truth. The solver implementations and
original solver-specific iteration/evaluation budgets are shared with the original
B-scan scaling experiment; the data and launch process are separate.

```bash
# Submit the eight synthetic seed tasks, then the Figure 1 notebook.
bash scripts/run_REBUTTAL_single_aline_scaling.sh
# Regenerate the figure from completed shards without rerunning optimization.
bash scripts/run_REBUTTAL_single_aline_scaling.sh figures
```

The original joint B-scan sweep now supplies S1(a). The liquid A-line and matched
joint/independent B-scan experiments are owned by `REBUTTAL_single_liquid_IL_aline/`.
