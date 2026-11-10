# Install and runtime guide

## Environment

The project uses a single conda environment defined in `setup/environment.yml`.

### On the MIT SuperCloud cluster

```bash
module load anaconda/2023b
conda env create -f setup/environment.yml      # first time only
source activate diff-oct
```

Note: `conda activate` requires `conda init`, which is not configured for the cluster shells; use `source activate` instead.

### On a local workstation

```bash
conda env create -f setup/environment.yml
conda activate diff-oct
```

Verify the install:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import h5py, pyvista, omegaconf, scipy, pandas, matplotlib; print('OK')"
```

Optional dependencies for figure rendering:

- `pyvista` (3D volume rendering for Figure 3 panels j–k). Off-screen mode is used; no display server required.
- `pdflatex` from TeX Live (Figure 4 table panels).

---

## Data layout

Two top-level directories are gitignored and supplied by the user.

### `inputs/`

Raw acquisitions, organized by date:

```
inputs/
  IntralipidPhantoms_20250206/        Phantom TIFF stacks (5 IL conc × 2 focal positions × 512 B-scans)
  IntralipidPhantoms_20260317/        Despeckled variants
  IntralipidPhantoms_20250206_Noised/ Noise-injected for robustness study
  Metadata.txt
```

Phantom filenames follow `AgarInIL_<conc_pct>..<replicate>.tif` and are auto-discovered by experiment scripts.

Clinical IV-OCT data (used by `experiments/generate_patient_oac.py`) is referenced by patient ID and is not redistributed; the expected layout is described in the paper Methods §4.6.

### `outputs/`

Per-experiment artifacts (gitignored):

```
outputs/
  forward_model_verification/         experiments/forward_model_verification.py
  inverse_model_verification/         experiments/inverse_model_verification.py
  phantom_validation_ss/              experiments/phantom_validation_ss_or_ms.py (SS run)
  phantom_validation_ms/              experiments/phantom_validation_ss_or_ms.py (MS run)
  identifiability_study/              experiments/identifiability_study.py
  robustness_study/                   experiments/robustness_study.py
  parameter_ablation/                 experiments/supp_parameter_ablation.py (Fig S1 + S2)
  optimizer_scaling/                  experiments/optimizer_scaling.py (Fig 1)
  supp_ehf/                           experiments/supp_ehf.py (Fig S3)
```

---

## Data flow for figure reproduction

```
experiments/X.py  ─── writes ───▶  outputs/X/             # raw, gitignored
                                          │
                                          └── extract ──▶  figures/data/X/   # small, tracked
                                                                  │
                                                  read by  ─▶  figures/X.ipynb
                                                                  │
                                                  saves   ─▶  figures/output/X.{pdf,png}
```

After running an experiment that produces data for a figure, copy the minimal subset needed for plotting into `figures/data/`. The notebooks read from `figures/data/` so figures can be regenerated without re-running compute.

The cached subsets committed to `figures/data/` are sufficient to reproduce main Figures 1 to 4. Figure 5 uses restricted clinical data and is provided as a rendered figure only.

---

## Cluster execution (SLURM)

Single-GPU job:

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -t 04:00:00
#SBATCH -o slurm-%j.out

module load anaconda/2023b
source activate diff-oct
cd /path/to/DiffOCT
python experiments/phantom_validation_ss_or_ms.py
EOF
```

Multi-GPU array (one task per phantom or per phase):

```bash
sbatch --array=0-9 <<'EOF'
#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH -t 06:00:00
#SBATCH -o slurm-%A_%a.out

module load anaconda/2023b
source activate diff-oct
cd /path/to/DiffOCT
PHANTOM_INDEX=$SLURM_ARRAY_TASK_ID python experiments/phantom_validation_ss_or_ms.py
EOF
```

Each experiment script reads environment variables for indexing (`PHANTOM_INDEX`, `BSCAN_INDEX`, `START_FRAME`, `END_FRAME`, `PHASE`, `RUN_INDEX`, `SUMMARIZE`); see the `experiments/<script>.py` headers for the supported variables.

---

## Curve-fitting tissue mask

`matlab_CF_scripts_results/IntMask.mat` is the tissue mask from the conventional MATLAB curve-fitting pipeline. Phantom experiments use it to report tissue-only μ.

---

## Common gotchas

- N(z) noise-floor subtraction in linear space requires eager mode and disables CUDA graphs.
- The `recompute_gain_mse_optimal_each_iter` flag also disables CUDA graphs but improves convergence.
- IntMask-filtered tissue μ is the correct comparison metric versus the curve-fitting reference; all-pixel averages are biased low by ~30% due to noise-floor masking.
- Phantom figures use the `plasma` colormap; clinical IV-OCT figures use `viridis`. Stay consistent with paper conventions.
