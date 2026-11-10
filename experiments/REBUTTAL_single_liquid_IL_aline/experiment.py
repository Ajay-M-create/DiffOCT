"""Complete S1 experiment: B-scan scaling, liquid scaling and R3.6 fits."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["prepare","validate","run","summarize","export","assemble","figures"])
    p.add_argument("--study",choices=["bscan","scaling","coupling","all"],default="all")
    p.add_argument("--task",type=int,default=0)
    a=p.parse_args()
    if a.command=="figures":
        subprocess.run([sys.executable,str(ROOT/"scripts/build_rebuttal_notebooks.py"),"s1"],cwd=ROOT,check=True)
        return
    if a.command=="assemble":
        from build_s1 import main as assemble
        assemble();return
    if a.command=="run" and a.study=="all":p.error("run requires a single --study")
    if a.study in ["all","bscan"]:
        import bscan_scaling
        if a.command=="prepare":bscan_scaling.prepare()
        elif a.command=="validate":bscan_scaling.validate()
        elif a.command=="run":bscan_scaling.run(a.task)
        elif a.command in ["summarize","export"]:bscan_scaling.export()
    if a.study in ["all","scaling"]:
        from rerun_step25 import backend,prepare,RUN
        if a.command=="prepare":
            if not (RUN/"data/config.json").exists(): prepare()
            backend().load_data()
        elif a.command=="validate":backend().validate()
        elif a.command=="run":
            assert a.study=="scaling" and 0<=a.task<32
            b=backend();config,_=b.load_data();seed,subset=divmod(a.task,4)
            b.run(seed,sizes=config["sizes"][subset::4])
        elif a.command=="summarize":
            from plot_step25 import main as summarize
            summarize()
        elif a.command=="export":
            from export_step25_npz import main as export
            export()
    if a.study in ["all","coupling"]:
        import coupling
        if a.command=="prepare":coupling.prepare()
        elif a.command=="validate":coupling.validate()
        elif a.command=="run":coupling.run(a.task)
        elif a.command in ["summarize","export"]:
            from coupling_report import main as summarize
            summarize()


if __name__=="__main__":main()
