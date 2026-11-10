"""Update the combined S1 figure and text from the verified step-25 rerun."""
import difflib
import json
from pathlib import Path
import shutil

from rerun_step25 import BASE, ROOT, RUN


def main():
    summary = json.loads((RUN/"summary.json").read_text())
    assert summary["runs"] == 608 and summary["verified_saved_objectives"]
    r = summary["aggregate"]["0.0"]["475"]
    source = (BASE/"supplementary_S1.template.tex").read_text()
    coupling = BASE/"runs/r3_6_joint_independent/supplementary_coupling.tex"
    if not coupling.exists():
        raise RuntimeError("Run the matched R3.6 comparison and coupling_report.py before building S1")
    source += "\n\\clearpage\n" + coupling.read_text()
    for tag, name in [("ADAM","Adam"),("LBFGS","L-BFGS-B"),("CMA","CMA-ES"),("RS","Random Search")]:
        for suffix, metric, fmt in [("MAPE","cf_pixel_mape_percent",".2f"),
                                    ("MSE","mse_db2",".4g"),("MS","ms_per_eval",".2f"),
                                    ("SEC","wall_sec",".2f"),("EVALS","fwd_evals",".0f")]:
            source = source.replace(f"@{tag}_{suffix}@", format(r[name][metric]["mean"],fmt))
    assert "@" not in source
    for metric in ["cf_pixel_mape_percent","mse_db2"]:
        assert max(r[m][metric]["mean"] for m in ["Adam","L-BFGS-B"]) < min(r[m][metric]["mean"] for m in ["CMA-ES","Random Search"])
    s1path = BASE/"supplementary_S1.tex"
    old_s1 = s1path.read_text()
    new_s1 = source
    paper = ROOT/"paper/DiffOCT_CommsEngRevisions"
    manuscript = paper/"sn-article.tex"
    before = manuscript.read_text()
    assert before.count(old_s1) == 1
    after = before.replace(old_s1,new_s1,1)
    after = after.replace("with axial TV off and on.","with axial TV disabled.")
    after = after.replace("both with and without axial TV (\\Cref{fig:optimizer_real_aline})", "with axial TV disabled (\\Cref{fig:optimizer_real_aline})")
    after = after.replace("both with and without axial TV.", "with axial TV disabled.")
    after = after.replace("real-data range of 25--476 parameters", "real-data range of 25--475 parameters")
    backup = RUN/"paper_integration"
    backup.mkdir(exist_ok=True)
    if not (backup/"sn-article.before_step25.tex").exists():
        (backup/"sn-article.before_step25.tex").write_text(before)
        (backup/"supplementary_S1.before_step25.tex").write_text(old_s1)
    (backup/"step25_changes.diff").write_text("".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="sn-article.before_step25.tex",tofile="sn-article.tex")))
    manuscript.write_text(after)
    s1path.write_text(new_s1)
    (paper/"supplementary_optimizer_scaling.tex").write_text(new_s1)
    shutil.copy2(ROOT/"figures/data/figureS1_optimizer_scaling/liquid/figureS1b.pdf",paper/"Figs/supp_optimizer_real_aline.pdf")
    shutil.copy2(ROOT/"figures/data/figureS1_optimizer_scaling/bscan/figureS1a.pdf",paper/"Figs/supp_optimizer_alines.pdf")
    shutil.copy2(ROOT/"figures/data/figureS1_optimizer_scaling/coupling/figureS1c.pdf",paper/"Figs/supp_optimizer_coupling.pdf")
    print("Updated S1(a–c), with 608 measured A-line fits and 23 joint/independent fits.")


if __name__ == "__main__":
    main()
