#!/usr/bin/env python3
"""Execute the Figure 1 and/or S1 notebooks using a real local Jupyter kernel.

Uses the installed jupyter_client protocol directly; no nbconvert dependency.
Writes execution counts, stdout, rich outputs, and errors into each notebook.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from queue import Empty
import sys
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS={"figure1":["figure1_single_aline_scaling.ipynb"],
           "s1":["figureS1a_bscan_scaling.ipynb","figureS1b_liquid_aline_scaling.ipynb",
                 "figureS1c_joint_independent.ipynb"],
           "s5":["figureS5_ss_vs_ms.ipynb"],
           "s8":["figureS8_init_uncertainty.ipynb"]}
S1=ROOT/"experiments/REBUTTAL_single_liquid_IL_aline/experiment.py"


def execute(path):
    from jupyter_client import KernelManager
    nb=json.loads(path.read_text())
    def source_signature(value):
        return [(c["cell_type"],"".join(c["source"])) for c in value["cells"]]
    initial_sources=source_signature(nb)
    def save():
        current=json.loads(path.read_text())
        if source_signature(current)!=initial_sources:
            raise RuntimeError(f"{path.name} was edited during execution; preserved the edited file")
        path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+"\n")
    env=os.environ.copy()
    env.update(MPLBACKEND="Agg",MPLCONFIGDIR="/tmp/diffoct-notebook-mpl",
               IPYTHONDIR="/tmp/diffoct-notebook-ipython",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
               PATH=str(Path(sys.executable).parent)+os.pathsep+env.get("PATH",""))
    km=KernelManager(kernel_name="python3")
    km.kernel_spec.argv[0]=sys.executable
    started=time.perf_counter();count=0;success=False
    km.start_kernel(cwd=str(ROOT/"figures"),env=env)
    client=km.client();client.start_channels()
    try:
        client.wait_for_ready(timeout=90)
        for idx,cell in enumerate(nb["cells"]):
            if cell["cell_type"]!="code":continue
            source=cell["source"]
            if isinstance(source,list):source="".join(source)
            cell["outputs"]=[];cell["execution_count"]=None
            msg_id=client.execute(source,allow_stdin=False,stop_on_error=True)
            deadline=time.monotonic()+900;error=None
            while True:
                if time.monotonic()>deadline:raise TimeoutError(f"{path.name} cell {idx+1}")
                try:message=client.get_iopub_msg(timeout=5)
                except Empty:continue
                if message.get("parent_header",{}).get("msg_id")!=msg_id:continue
                kind=message["header"]["msg_type"];c=message["content"]
                if kind=="execute_input":cell["execution_count"]=c["execution_count"]
                elif kind=="stream":cell["outputs"].append(dict(output_type="stream",name=c["name"],text=c["text"]))
                elif kind in ["display_data","execute_result"]:
                    value=dict(output_type=kind,data=c["data"],metadata=c.get("metadata",{}))
                    if kind=="execute_result":value["execution_count"]=c["execution_count"]
                    cell["outputs"].append(value)
                elif kind=="clear_output":cell["outputs"]=[]
                elif kind=="error":
                    error=c
                    cell["outputs"].append(dict(output_type="error",ename=c["ename"],evalue=c["evalue"],traceback=c["traceback"]))
                elif kind=="status" and c["execution_state"]=="idle":break
            count+=1
            save()
            if error:raise RuntimeError(f"{path.name}, cell {idx+1}: {error['ename']}: {error['evalue']}")
            print(f"Executed {path.name}: cell {idx+1}",flush=True)
        success=True
    finally:
        client.stop_channels();km.shutdown_kernel(now=True)
        save()
    return dict(notebook=str(path.relative_to(ROOT)),code_cells=count,success=success,
                seconds=time.perf_counter()-started,python=sys.executable,
                source_sha256=hashlib.sha256(json.dumps(initial_sources).encode()).hexdigest(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("target",choices=["figure1","s1","s5","s8","all","remapped"])
    a=p.parse_args()
    if a.target in ["s1","all","remapped"]:
        subprocess.run([sys.executable,str(S1),"export"],
                       cwd=ROOT,check=True)
    if a.target in ["all","remapped"]:
        names=NOTEBOOKS["figure1"]+NOTEBOOKS["s1"]
        if a.target=="remapped":
            names += [p.name for p in sorted((ROOT/"figures").glob("*.ipynb")) if p.name not in names]
    else:
        names=NOTEBOOKS[a.target]
    records=[execute(ROOT/"figures"/name) for name in names]
    if a.target in ["s1","all","remapped"]:
        subprocess.run([sys.executable,str(S1),"assemble"],cwd=ROOT,check=True)
    output=ROOT/("figures/data/figureS1_optimizer_scaling" if a.target=="s1" else "figures/output")
    output.mkdir(parents=True,exist_ok=True)
    (output/f"notebook_build_{a.target}.json").write_text(json.dumps(records,indent=2)+"\n")
    print(f"Successfully executed {len(records)} notebooks for {a.target}.",flush=True)


if __name__=="__main__":main()
