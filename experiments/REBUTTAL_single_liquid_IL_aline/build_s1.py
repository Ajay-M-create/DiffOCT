"""Assemble the three executed S1 panels and compile the complete S1 section."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/"figures/output"
DATA=ROOT/"figures/data/figureS1_optimizer_scaling"
PANELS={"a":DATA/"bscan", "b":DATA/"liquid", "c":DATA/"coupling"}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Keep compatibility copies inside the experiment; canonical plots are S1a–c.
    for panel,directory,alias in [
        ("a","bscan",None),
        ("b","liquid","rebuttal_single_liquid_IL_aline_scaling"),
        ("c","coupling","coupling_comparison")]:
        for suffix in ["pdf","jpeg","png"]:
            source=PANELS[panel]/f"figureS1{panel}.{suffix}"
            if alias:
                shutil.copy2(source,DATA/directory/f"{alias}.{suffix}")
                shutil.copy2(source,HERE/"figures"/f"{alias}.{suffix}")
    # Vector PDF composition: scaling panels above the full-width map comparison.
    composite=fitz.open()
    docs=[fitz.open(PANELS[panel]/f"figureS1{panel}.pdf") for panel in "abc"]
    try:
        width,margin,gap,label_height=1000.,12.,16.,24.
        half=(width-gap)/2
        top=max(half*d[0].rect.height/d[0].rect.width for d in docs[:2])
        bottom=width*docs[2][0].rect.height/docs[2][0].rect.width
        page=composite.new_page(width=width+2*margin,
            height=2*margin+top+bottom+2*label_height+gap)
        rects=[fitz.Rect(margin,margin,margin+half,margin+top),
               fitz.Rect(margin+half+gap,margin,margin+width,margin+top),
               fitz.Rect(margin,margin+top+label_height+gap,
                         margin+width,margin+top+label_height+gap+bottom)]
        for doc,rect,label in zip(docs,rects,["(a)","(b)","(c)"]):
            page.show_pdf_page(rect,doc,0)
            x=(rect.x0+rect.x1-fitz.get_text_length(label,fontsize=16))/2
            page.insert_text((x,rect.y1+19),label,fontsize=16)
        assert all(label in page.get_text() for label in ["(a)","(b)","(c)"])
        composite.save(OUT/"figureS1.pdf",garbage=4,deflate=True)
        page.get_pixmap(matrix=fitz.Matrix(200/72,200/72)).save(OUT/"figureS1.jpeg")
    finally:
        for doc in docs:doc.close()
        composite.close()
    for name in ["figureS1.pdf","figureS1.jpeg"]:
        shutil.copy2(OUT/name,DATA/name)
    from update_step25_paper import main as update_paper
    update_paper()
    subprocess.run(["bash",str(HERE/"build_preview.sh")],cwd=ROOT,check=True)
    preview=HERE/"paper_preview/supplementary_S1_preview.pdf"
    with fitz.open(preview) as doc:
        text="\n".join(page.get_text() for page in doc)
        assert "Figure S1" in text and "Figure S2" not in text
        assert all(label in text for label in ["(a)","(b)","(c)"])
        assert "Matched joint and independent fits" in text
        coupling=json.loads((DATA/"coupling/summary.json").read_text())["cf_window"]
        for items in coupling["matched_differences"].values():
            mean=sum(item["mae"] for item in items)/len(items)
            assert f"{mean:.6f}" in text
        assert "CF-window comparisons" in text
        assert "N = 3 runs" in text
        pages=len(doc)
    for directory in [DATA,DATA/"liquid"]:
        shutil.copy2(preview,directory/preview.name)
    # Record exactly which inputs and notebook sources produced this build.
    manifest=dict(panels=["S1(a): synthetic B-scan scaling","S1(b): liquid A-line scaling",
                         "S1(c): recovered attenuation and runtime"],preview_pages=pages,
                  verified_runs=dict(bscan=576,liquid=608,coupling=23),sha256={})
    sources=[DATA/part/"source_manifest.json" for part in ["bscan","liquid","coupling"]]
    sources += [ROOT/"figures"/name for name in ["figureS1a_bscan_scaling.ipynb",
                "figureS1b_liquid_aline_scaling.ipynb","figureS1c_joint_independent.ipynb"]]
    sources += [PANELS[panel]/f"figureS1{panel}.pdf" for panel in "abc"]+[OUT/"figureS1.pdf",preview]
    for path in sources:
        manifest["sha256"][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (DATA/"build_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Built all three S1 panels, composite figure, and {pages}-page section preview.")


if __name__=="__main__":main()
