# Figure notebooks

Each notebook renders one main figure of the paper from the cached arrays in `data/`
and writes `output/figureN.{pdf,jpeg,svg}`. Run them from this directory (or the
repository root) in the `diff-oct` environment; none of them re-runs an optimization.

| Figure | Notebook | Data folder under `data/` |
|---|---|---|
| 1, single-A-line optimizer scaling | [figure1_single_aline_scaling.ipynb](figure1_single_aline_scaling.ipynb) | `figure1_single_aline_scaling/` |
| 2, forward and inverse verification | [figure2_verification.ipynb](figure2_verification.ipynb) | `figure2_verification/` |
| 3, liquid Intralipid phantom validation | [figure3_liquid_phantom.ipynb](figure3_liquid_phantom.ipynb) | `figure3_liquid_phantom/` |
| 4, identifiability and noise robustness | [figure4_identifiability_robustness.ipynb](figure4_identifiability_robustness.ipynb) | `figure4_identifiability_robustness/` |
| 5, clinical IV-OCT μ and α | Rendered figure only, `output/figure5.*` | Restricted clinical data, not included |
| 6, framework overview | Externally drawn diagram | n/a |

Figure 1 merges the per-seed result shards in `data/figure1_single_aline_scaling/`
(or fresh shards in `outputs/rebuttal_single_aline_scaling/` after re-running
`experiments/REBUTTAL_single_aline_scaling.py`). Figure 4 typesets its tables with
`pdflatex`. Figure 5 is built from clinical data restricted by a data use agreement,
so only the rendered figure is provided.

`output/Supplementary_Data_1_Source_Data.xlsx` holds the source data for all main
figures. It was assembled by `scripts/build_source_data.py`, which also reads raw
experiment results that are not part of this repository.
