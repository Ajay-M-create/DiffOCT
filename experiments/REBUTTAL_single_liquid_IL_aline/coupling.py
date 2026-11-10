"""R3.6: matched joint B-scan and independent A-line optimization."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
from benchmark import ROOT, HERE, SOURCE, sha256, write_json

RUN = HERE / "runs/r3_6_joint_independent"
CASES = [("joint_no_tv", False, 0., 0.), ("independent_no_tv", True, 0., 0.),
         ("joint_axial_tv", False, 1., 0.), ("independent_axial_tv", True, 1., 0.),
         ("joint_axial_lateral_tv", False, 1., 1.)]
SERIAL_COLUMNS = [0, 166, 333, 499]


def prepare():
    if (RUN / "config.json").exists():
        load()
        print("Existing frozen R3.6 input verified")
        return
    RUN.mkdir(parents=True, exist_ok=True)
    for name in ["results", "logs", "figures"]:
        (RUN / name).mkdir(exist_ok=True)
    with np.load(SOURCE) as f:
        row = int(np.flatnonzero(f["python_bscan"] == 255)[0])
        d = {k: f[k][row].copy() for k in ["target_db", "gain_db", "cf_mu", "gain_fallback"]}
        d.update(valid=f["full_depth_loss_mask"][row].astype(bool),
                 transport=f["full_depth_transport_mask"][row].astype(bool),
                 cf_mask=f["comparison_mask"][row].astype(bool),
                 dz_mm=np.array(float(f["dz_mm"])))
    assert d["target_db"].shape == (500,500)
    assert np.isfinite(d["target_db"]).all() and np.isfinite(d["gain_db"]).all()
    assert d["valid"].sum(axis=0).min() > 0
    np.savez_compressed(RUN/"bscan.npz", **d)
    write_json(RUN/"config.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(SOURCE.relative_to(ROOT)), source_sha256=sha256(SOURCE),
        data_sha256=sha256(RUN/"bscan.npz"), code_sha256=sha256(__file__),
        selection="1% H, MATLAB B-scan 256; all 500 A-lines",
        seeds=[0,1,2], cases=[x[0] for x in CASES], steps=20000,
        valid_pixels=int(d["valid"].sum()), transport_pixels=int(d["transport"].sum()),
        gain_fallback_alines=int(d["gain_fallback"].sum()),
        dtype="float64", lr=.01, bounds=[.01,20.], loss_scale=1000., adam_eps=1e-8,
        initialization="Uniform(.01,10), seed 42+seed, same native matrix for every case",
        objective="1000*(sum valid squared dB residual / K + beta_z*sum masked axial differences / Z + beta_x*sum masked lateral differences / X)",
        normalization="K,Z,X are the fixed full-B-scan valid observation/edge counts. Independent A-lines minimize their own weighted contributions; sum equals joint loss without lateral TV. Scaling retained to match Adam epsilon effects.",
        independent_execution="500 separate A-line objectives and elementwise Adam states evaluated in a batch, with a common fixed update count; no cross-A-line term or shared optimized parameter",
        serial_columns_python=SERIAL_COLUMNS, serial_seed=0,
        timing="Full-B-scan wall time includes all 20000 updates and final evaluation; independent full-frame timings refer to batched execution, not 500 serial fits",
        precision_note="Float64 used in every matched case to reduce reduction-order roundoff",
        scope="This comparison uses Adam, fixed per-A-line gain, and one processed frame. It tests axial versus lateral coupling; it does not isolate acquisition/despeckling correlations."))
    print("Prepared", RUN)


def load():
    config = json.loads((RUN/"config.json").read_text())
    assert sha256(RUN/"bscan.npz") == config["data_sha256"]
    assert sha256(__file__) == config["code_sha256"], "R3.6 runner changed after inputs were frozen"
    with np.load(RUN/"bscan.npz") as f:
        data = {k:f[k] for k in f.files}
    return config, data


class Problem:
    def __init__(self, d, independent, bz, bx, device="cpu", columns=None, dtype=torch.float64):
        self.independent, self.bz, self.bx = independent, bz, bx
        self.zdim = 1 if independent else 0
        self.dz = float(d["dz_mm"])
        self.K = int(d["valid"].sum())
        self.Z = int((d["transport"][1:] & d["transport"][:-1]).sum())
        self.X = int((d["transport"][:,1:] & d["transport"][:,:-1]).sum())
        cols = np.arange(d["target_db"].shape[1]) if columns is None else np.asarray(columns)
        self.cols = cols
        def convert(key, boolean=False):
            a=d[key][:,cols]
            if independent: a=a.T
            return torch.tensor(a.copy(), device=device, dtype=torch.bool if boolean else dtype)
        self.target=convert("target_db")
        self.valid=convert("valid",True)
        self.transport=convert("transport",True)
        gain=torch.tensor(d["gain_db"][cols],device=device,dtype=dtype)
        self.gain=gain[:,None] if independent else gain[None,:]
        self.zedges=self.transport.narrow(self.zdim,1,self.transport.shape[self.zdim]-1) & self.transport.narrow(self.zdim,0,self.transport.shape[self.zdim]-1)
        if bx:
            assert not independent and columns is None
            self.xedges=self.transport[:,1:] & self.transport[:,:-1]

    def forward(self,mu):
        active=torch.where(self.transport,mu,0.)
        cumulative=torch.cumsum(active*self.dz,dim=self.zdim)
        zeros=torch.zeros_like(cumulative.narrow(self.zdim,0,1))
        tau=torch.cat([zeros,cumulative.narrow(self.zdim,0,cumulative.shape[self.zdim]-1)],dim=self.zdim)
        local=torch.where(self.transport,mu,1.)
        return self.gain+10*torch.log10(local)-(20/math.log(10))*tau

    def parts(self,mu):
        pred=self.forward(mu)
        mse_a=torch.where(self.valid,(pred-self.target).square(),0.).sum(dim=self.zdim)/self.K
        z_a=(torch.diff(mu,dim=self.zdim).abs()*self.zedges).sum(dim=self.zdim)/max(self.Z,1)
        x=(torch.diff(mu,dim=1).abs()*self.xedges).sum()/max(self.X,1) if self.bx else mu.new_zeros(())
        contributions=mse_a+self.bz*z_a
        return contributions, mse_a.sum(), z_a.sum(), x, pred

    def loss(self,mu):
        a,_,_,x,_=self.parts(mu)
        return 1000*(a.sum()+self.bx*x)


def validate():
    config,d=load()
    # Use a small crop to test the exact mathematical decomposition in float64.
    toy=dict(target_db=d["target_db"][30:42,:4],gain_db=d["gain_db"][:4],
             valid=np.ones((12,4),bool),transport=np.ones((12,4),bool),dz_mm=d["dz_mm"])
    rng=np.random.default_rng(817)
    init=torch.tensor(rng.uniform(.3,3,(12,4)),dtype=torch.float64)
    checks={}
    for bz in [0.,1.]:
        joint=Problem(toy,False,bz,0)
        independent=Problem(toy,True,bz,0)
        x=init.clone().requires_grad_(); y=init.T.contiguous().requires_grad_()
        lj=joint.loss(x); li=independent.loss(y)
        gj,=torch.autograd.grad(lj,x); gi,=torch.autograd.grad(li,y)
        torch.testing.assert_close(lj,li,rtol=1e-13,atol=1e-10)
        torch.testing.assert_close(gj,gi.T,rtol=1e-13,atol=1e-10)
        a=torch.nn.Parameter(init.clone()); b=torch.nn.Parameter(init.T.contiguous())
        oa=torch.optim.Adam([a],lr=.01); ob=torch.optim.Adam([b],lr=.01)
        for _ in range(50):
            for q,o,p in [(a,oa,joint),(b,ob,independent)]:
                o.zero_grad(set_to_none=True); p.loss(q).backward();o.step()
        torch.testing.assert_close(a,b.T,rtol=1e-10,atol=1e-10)
        checks[f"bz_{bz:g}"]=dict(loss_abs_difference=float(abs(lj-li)),gradient_max_abs=float((gj-gi.T).abs().max()),adam_50_step_max_abs=float((a-b.T).abs().max()))
    # A-line perturbation changes its own downstream predictions only.
    p=Problem(toy,False,0,0); x=init.clone(); y=x.clone(); y[3,1]+=.7
    diff=p.forward(y)-p.forward(x)
    assert torch.all(diff[:,[0,2,3]]==0) and torch.all(diff[:3,1]==0) and torch.all(diff[3:,1]!=0)
    checks["transport_cross_aline_max_abs"]=float(diff[:,[0,2,3]].abs().max())
    # Lateral L1 TV has genuine cross-line dependence even though its Hessian
    # is zero away from kinks. Force an ordering reversal to test the gradient.
    a=torch.ones((12,4),dtype=torch.float64); a[:,1]=2
    b=a.clone(); b[:,1]=.5
    def grad(v,bx):
        v=v.clone().requires_grad_(); loss=Problem(toy,False,0,bx).loss(v)
        return torch.autograd.grad(loss,v)[0]
    off=(grad(a,0)-grad(b,0))[:,0].abs().max()
    on=(grad(a,1)-grad(b,1))[:,0].abs().max()
    assert off==0 and on>0
    checks["neighbor_gradient_change_tv_off"]=float(off)
    checks["neighbor_gradient_change_lateral_tv_on"]=float(on)
    # Match the actual production objective, including transport and TV masks.
    import importlib.util
    source=ROOT/"experiments/REBUTTAL_liquid_intralipid_cf_comparison/model.py"
    spec=importlib.util.spec_from_file_location("production_liquid_model",source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    p=Problem(d,False,1,1)
    mu=torch.tensor(rng.uniform(.3,3,d["target_db"].shape),dtype=torch.float64)
    actual=mod.objective(mu[None],p.target[None],p.valid[None],p.transport[None],p.dz,p.gain)[0][0]
    torch.testing.assert_close(p.loss(mu),actual,rtol=1e-12,atol=1e-8)
    checks["production_objective_abs_difference"]=float(abs(p.loss(mu)-actual))
    write_json(RUN/"validation.json",dict(status="passed",checks=checks))
    print(json.dumps(checks,indent=2))


def run(task, steps_override=None, cpu=False):
    config,d=load()
    assert (RUN/"validation.json").exists()
    torch.set_num_threads(1)
    device=torch.device("cpu" if cpu else "cuda")
    if not cpu and not torch.cuda.is_available(): raise RuntimeError("R3.6 timing requires CUDA")
    assert 0<=task<23
    if task<15:
        seed,case=divmod(task,5); name,ind,bz,bx=CASES[case]; columns=None
        stem=f"{name}_seed{seed}"
    else:
        case,col=divmod(task-15,4); seed=0; columns=[SERIAL_COLUMNS[col]]
        name=f"serial_{'no_tv' if case==0 else 'axial_tv'}"; ind=True; bz=float(case);bx=0.
        stem=f"{name}_col{columns[0]:03d}"
    out=RUN/("checks" if cpu or steps_override else "results");out.mkdir(exist_ok=True)
    path=out/f"{stem}.json"
    steps=steps_override or config["steps"]
    if path.exists():
        old=json.loads(path.read_text())
        assert old["code_sha256"]==config["code_sha256"] and old["steps"]==steps
        assert old["arrays_sha256"]==sha256(out/f"{stem}.npz")
        print("Already complete",stem);return
    p=Problem(d,ind,bz,bx,device=device,columns=columns)
    full=np.random.default_rng(42+seed).uniform(.01,10,d["target_db"].shape)
    init=full[:,p.cols];init=init.T.copy() if ind else init.copy()
    mu=torch.nn.Parameter(torch.tensor(init,device=device,dtype=torch.float64))
    with torch.no_grad():mu[~p.transport]=1.
    optimizer=torch.optim.Adam([mu],lr=.01,eps=1e-8)
    torch.autograd.grad(p.loss(mu),mu) # untimed kernel warmup, no update
    if not cpu: torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        a,mse,tvz,tvx,_=p.parts(mu)
        value=1000*(a.sum()+bx*tvx)
        if step%100==0:
            history.append([step,time.perf_counter()-started,float(value.detach()/1000),float(mse.detach()),float(tvz.detach()),float(tvx.detach())])
        if ind:
            torch.autograd.backward(1000*a,torch.ones_like(a))
        else:
            value.backward()
        optimizer.step()
        with torch.no_grad():mu.clamp_(.01,20.);mu[~p.transport]=1.
    with torch.no_grad():
        a,mse,tvz,tvx,pred=p.parts(mu);value=a.sum()+bx*tvx
    if not cpu:torch.cuda.synchronize()
    wall=time.perf_counter()-started
    history.append([steps,wall,float(value),float(mse),float(tvz),float(tvx)])
    result_mu=mu.detach().cpu().numpy();result_pred=pred.cpu().numpy()
    if ind:result_mu=result_mu.T;result_pred=result_pred.T
    assert np.isfinite(result_mu).all() and np.isfinite(history).all()
    cm=d["cf_mask"][:,p.cols] & np.isfinite(d["cf_mu"][p.cols])[None,:] & (d["cf_mu"][p.cols]>0)[None,:]
    ref=np.broadcast_to(d["cf_mu"][p.cols],result_mu.shape)
    cf_mape=float(np.mean(100*np.abs(result_mu[cm]-ref[cm])/ref[cm])) if cm.any() else None
    np.savez_compressed(out/f"{stem}.npz",mu=result_mu,pred_db=result_pred,history=np.asarray(history),columns=p.cols)
    record=dict(case=name,seed=seed,columns_python=p.cols.tolist(),steps=steps,
        objective=float(value),mse_weighted=float(mse),tvz_weighted=float(tvz),tvx_weighted=float(tvx),
        actual_column_mse=float(np.mean((result_pred-d["target_db"][:,p.cols])[d["valid"][:,p.cols]]**2)),
        cf_mape_percent=cf_mape,wall_sec=wall,ms_per_update=1000*wall/steps,
        code_sha256=config["code_sha256"],config_sha256=sha256(RUN/"config.json"),data_sha256=config["data_sha256"],
        arrays_sha256=sha256(out/f"{stem}.npz"),gpu=torch.cuda.get_device_name() if not cpu else None,
        torch=torch.__version__,host=platform.node(),job_id=os.environ.get("SLURM_JOB_ID"))
    write_json(path,record)
    print(stem,json.dumps({k:record[k] for k in ["objective","actual_column_mse","cf_mape_percent","wall_sec"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["prepare","validate","run"])
    p.add_argument("--task",type=int,default=0)
    p.add_argument("--steps",type=int)
    p.add_argument("--cpu",action="store_true")
    a=p.parse_args()
    if a.command=="prepare":prepare()
    elif a.command=="validate":validate()
    else:run(a.task,a.steps,a.cpu)
