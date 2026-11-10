#!/usr/bin/env python3
"""Validate the staged S7 results and render four compact LaTeX tables."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile

import fitz
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "figures/data/figureS7_model_mismatch"
OUT = ROOT / "figures/output"

# Each table repeats the same matched reference. No combined condition is shown.
PANELS = [
    ("a", r"$\alpha$", r"SD (dB)",
     [("matched", "$0$"), ("alpha_1dB", "$1$"), ("alpha_3dB", "$3$")]),
    ("b", r"Gain", "Perturbation",
     [("matched", "$0$ dB"), ("gain_m3", "$-3$ dB"),
      ("gain_p3", "$+3$ dB")]),
    ("c", r"$z_f$", "Error (px)",
     [("zf_m20", "$-20$"), ("matched", "$0$"), ("zf_p20", "$+20$")]),
    ("d", r"$z_R$", "Factor",
     [("zr_x0p8", "$0.8$"), ("matched", "$1.0$"), ("zr_x1p2", "$1.2$")]),
]


def validate():
    manifest = json.loads((DATA / "source_manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
    validation = json.loads((DATA / "validation.json").read_text())
    assert validation["noiseless_status"] == validation["texture_status"] == "PASS"
    runs = pd.read_csv(DATA / "metrics_per_run.csv")
    summary = pd.read_csv(DATA / "summary_table.csv").set_index("scenario")
    expected = {key for _, _, _, rows in PANELS for key, _ in rows}
    assert set(summary.index) == set(runs.scenario)
    assert expected.issubset(summary.index)
    assert runs.groupby("scenario").seed.apply(lambda x: set(x) == {0, 1, 2, 3, 4}).all()
    assert len(runs) == 5 * len(summary)
    np.testing.assert_allclose(runs.bias_percent, 100 * (runs.mu_hat / runs.mu_true - 1), atol=1e-12)
    for metric in ("mu_hat", "bias_percent", "mape_percent", "delta_mape_pp"):
        grouped = runs.groupby("scenario")[metric]
        for suffix, values in (("mean", grouped.mean()), ("sd", grouped.std(ddof=1))):
            np.testing.assert_allclose(summary[f"{metric}_{suffix}"], values.reindex(summary.index),
                                       rtol=1e-9, atol=1e-12)
    # An earlier simulator ignored gain_error_db. Refuse to plot those stale results.
    for key in ("gain_m3", "gain_p3"):
        assert not np.array_equal(summary.loc[key].values, summary.loc["matched"].values)
        a = runs[runs.scenario == key].sort_values("seed").mu_hat.to_numpy()
        b = runs[runs.scenario == "matched"].sort_values("seed").mu_hat.to_numpy()
        assert not np.array_equal(a, b), "Gain offset was not applied to the simulated signal"
    return summary.loc[summary.index.isin(expected)], manifest, validation


def build():
    summary, manifest, validation = validate()
    OUT.mkdir(exist_ok=True)
    panels, plotted = [], []
    for letter, title, parameter, rows in PANELS:
        lines = [r"\begin{minipage}[t]{228pt}",
                 rf"\textbf{{({letter})}}\enspace {title}\par\vspace{{3pt}}",
                 r"\fontsize{9}{11}\selectfont",
                 r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcc@{}}",
                 r"\toprule",
                 parameter + r" & Bias (\%) & $\Delta$MAPE (pp) \\", r"\midrule"]
        for key, label in rows:
            row = summary.loc[key]
            values = []
            for metric in ("bias_percent", "delta_mape_pp"):      # MAPE change vs the paired matched run
                mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
                mean = mean if round(mean, 1) != 0 else 0.0          # no "-0.0" after rounding
                if metric == "delta_mape_pp" and key == "matched":
                    values.append(r"$0$")                           # reference row
                else:
                    values.append(rf"${mean:.1f} \pm {sd:.1f}$")
            lines.append(label + " & " + " & ".join(values) + r" \\")
            plotted.append(dict(panel=letter, parameter=title, scenario=key, condition=label,
                                **{k: row[k] for k in row.index if k.endswith(("_mean", "_sd"))}))
        lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{minipage}"])
        panels.append("\n".join(lines))
    body = (r"\begin{minipage}{464pt}" + "\n" + panels[0] + r"\hfill" + "\n" + panels[1]
            + r"\par\vspace{10pt}" + "\n" + panels[2] + r"\hfill" + "\n" + panels[3]
            + "\n" + r"\end{minipage}")
    source = (r"\documentclass[border=3pt]{standalone}" + "\n"
              + r"\usepackage{booktabs,amsmath}" + "\n"
              + r"\renewcommand{\arraystretch}{1.3}" + "\n"
              + r"\begin{document}" + "\n" + body + "\n" + r"\end{document}" + "\n")
    (OUT / "figureS7.tex").write_text(source)
    (OUT / "figureS7_table.tex").write_text(body + "\n")
    pd.DataFrame(plotted).to_csv(OUT / "figureS7_table.csv", index=False)
    with tempfile.TemporaryDirectory(prefix="diffoct-s7-") as tmp:
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                 f"-output-directory={tmp}", str(OUT / "figureS7.tex")],
                                capture_output=True, text=True)
        if result.returncode or "Overfull" in result.stdout:
            raise RuntimeError(result.stdout[-5000:])
        (OUT / "figureS7.pdf").write_bytes((Path(tmp) / "figureS7.pdf").read_bytes())
    with fitz.open(OUT / "figureS7.pdf") as doc:
        doc[0].get_pixmap(dpi=300).save(OUT / "figureS7.jpeg")
        (OUT / "figureS7.svg").write_text(doc[0].get_svg_image())
    report = dict(source_run=manifest["run"], rows=len(plotted), unique_conditions=len(summary),
                  combined_included=False, sd_denominator="N-1", realizations=5,
                  metrics=["bias_percent", "delta_mape_pp"], input_validation=validation)
    (OUT / "figureS7_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    # Keep the displayed table data beside the raw inputs for the notebook.
    # The raw simulation summaries retain every completed condition.
    export_hashes = {}
    for name in ("figureS7_table.csv", "figureS7_table.tex"):
        payload = (OUT / name).read_bytes()
        (DATA / name).write_bytes(payload)
        export_hashes[name] = hashlib.sha256(payload).hexdigest()
    (DATA / "figure_exports_manifest.json").write_text(json.dumps(dict(
        source_run=manifest["run"], source_sha256=manifest["sha256"],
        displayed_scenarios=list(dict.fromkeys(row["scenario"] for row in plotted)),
        excluded_from_figure=["gain_drift", "combined"], sha256=export_hashes,
        builder="experiments/REBUTTAL_model_mismatch/build_figure.py",
    ), indent=2) + "\n")
    print(OUT / "figureS7.pdf")
    return pd.DataFrame(plotted)


if __name__ == "__main__":
    build()
