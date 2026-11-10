"""Validate R3.6 saved states and generate the joint/independent comparison."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from coupling import RUN,CASES,SERIAL_COLUMNS,load
from benchmark import HERE,ROOT,sha256,write_json

STAGE=ROOT/"figures/data/figureS1_optimizer_scaling/coupling"


def differences(x,y,mask):
    delta=np.abs(x-y)[mask]
    return dict(mae=float(delta.mean()),p99=float(np.quantile(delta,.99)),max=float(delta.max()),
                rmse=float(np.sqrt(np.mean(delta**2))))


def main():
    config,data=load();records={};states={};rows=[];gpus=set()
    cf_mask=data["cf_mask"]
    assert np.all(cf_mask <= data["transport"]) and np.all(cf_mask <= data["valid"])
    assert np.all(cf_mask.sum(axis=0)==100)
    expected=[f"{case[0]}_seed{s}" for s in config["seeds"] for case in CASES]
    expected += [f"serial_{tv}_col{c:03d}" for tv in ["no_tv","axial_tv"] for c in SERIAL_COLUMNS]
    for stem in expected:
        r=json.loads((RUN/"results"/f"{stem}.json").read_text())
        path=RUN/"results"/f"{stem}.npz"
        assert r["code_sha256"]==config["code_sha256"] and r["data_sha256"]==config["data_sha256"]
        assert r["config_sha256"]==sha256(RUN/"config.json") and r["arrays_sha256"]==sha256(path)
        assert r["steps"]==20000
        with np.load(path) as f:state={k:f[k] for k in f.files}
        mu=state["mu"];cols=state["columns"]
        transport=data["transport"][:,cols];valid=data["valid"][:,cols]
        optical=np.cumsum(np.where(transport,mu,0)*float(data["dz_mm"]),axis=0)
        tau=np.vstack([np.zeros_like(optical[:1]),optical[:-1]])
        pred=data["gain_db"][cols][None,:]+10*np.log10(np.where(transport,mu,1))-(20/np.log(10))*tau
        np.testing.assert_allclose(pred,state["pred_db"],rtol=1e-11,atol=1e-9)
        mse=np.mean((pred-data["target_db"][:,cols])[valid]**2)
        np.testing.assert_allclose(mse,r["actual_column_mse"],rtol=1e-9,atol=1e-10)
        assert state["history"][-1,0]==20000 and np.all(np.isfinite(state["history"]))
        # Recompute all regularizer terms independently from the saved map.
        zmask=transport[1:]&transport[:-1]
        tvz=np.sum(abs(np.diff(mu,axis=0))*zmask)/max(int((data["transport"][1:]&data["transport"][:-1]).sum()),1)
        tvx=0.
        if r["case"]=="joint_axial_lateral_tv":
            xm=transport[:,1:]&transport[:,:-1]
            tvx=float(np.mean(abs(np.diff(mu,axis=1))[xm]))
        weighted_mse=float(np.sum((pred-data["target_db"][:,cols])[valid]**2)/data["valid"].sum())
        bz=1. if "axial" in r["case"] else 0.
        np.testing.assert_allclose(weighted_mse+bz*tvz+tvx,r["objective"],rtol=1e-9,atol=1e-10)
        r["lateral_tv_native"]=(float(np.mean(abs(np.diff(mu,axis=1))[transport[:,1:]&transport[:,:-1]])) if len(cols)>1 else None)
        cm=cf_mask[:,cols]
        r["cf_window_mse_db2"]=float(np.mean((pred-data["target_db"][:,cols])[cm]**2))
        r["cf_window_neighbor_difference"]=(float(np.mean(abs(np.diff(mu,axis=1))[cm[:,1:]&cm[:,:-1]])) if len(cols)>1 else None)
        ref=data["cf_mu"][cols]
        positive=cm & (np.isfinite(ref)&(ref>0))[None,:]
        computed_mape=float(np.mean((100*abs(mu-np.broadcast_to(ref,mu.shape))/np.where(ref>0,ref,1))[positive]))
        np.testing.assert_allclose(computed_mape,r["cf_mape_percent"],rtol=1e-12)
        records[stem]=r;states[stem]=state;gpus.add(r["gpu"])
        rows.append({k:r[k] for k in ["case","seed","cf_window_mse_db2","cf_mape_percent","cf_window_neighbor_difference","wall_sec","ms_per_update"]})
    assert len(gpus)==1 and None not in gpus
    pairs={}
    for tv in ["no_tv","axial_tv"]:
        pairs[tv]=[]
        for seed in config["seeds"]:
            j=f"joint_{tv}_seed{seed}";i=f"independent_{tv}_seed{seed}"
            d=differences(states[j]["mu"],states[i]["mu"],data["transport"])
            d.update(seed=seed,mse_abs_difference=abs(records[j]["actual_column_mse"]-records[i]["actual_column_mse"]),
                     joint_wall_sec=records[j]["wall_sec"],independent_batched_wall_sec=records[i]["wall_sec"])
            pairs[tv].append(d)
    serial=[]
    for tv in ["no_tv","axial_tv"]:
        for col in SERIAL_COLUMNS:
            a=states[f"serial_{tv}_col{col:03d}"]["mu"]
            b=states[f"independent_{tv}_seed0"]["mu"][:,col:col+1]
            d=differences(a,b,data["transport"][:,col:col+1]);d.update(tv=tv,column_python=col)
            serial.append(d)
    aggregate={}
    for case in CASES:
        vals=[records[f"{case[0]}_seed{s}"] for s in config["seeds"]]
        aggregate[case[0]]={metric:dict(mean=float(np.mean([x[metric] for x in vals])),sd=float(np.std([x[metric] for x in vals],ddof=1)))
                            for metric in ["objective","actual_column_mse","cf_mape_percent","lateral_tv_native","wall_sec","ms_per_update"]}
    lateral=[differences(states[f"joint_axial_lateral_tv_seed{s}"]["mu"],states[f"independent_axial_tv_seed{s}"]["mu"],data["transport"]) for s in config["seeds"]]
    summary=dict(config=config,verified_runs=len(records),gpu=next(iter(gpus)),aggregate=aggregate,
                 matched_differences=pairs,serial_checks=serial,lateral_tv_map_changes=lateral)
    # Retain full-depth diagnostics above for provenance; S1(c) reports only CF-window metrics.
    cf_pairs={};cf_serial=[]
    for tv in ["no_tv","axial_tv"]:
        cf_pairs[tv]=[]
        for seed in config["seeds"]:
            j=f"joint_{tv}_seed{seed}";i=f"independent_{tv}_seed{seed}"
            d=differences(states[j]["mu"],states[i]["mu"],cf_mask)
            d.update(seed=seed,mse_abs_difference=abs(records[j]["cf_window_mse_db2"]-records[i]["cf_window_mse_db2"]))
            cf_pairs[tv].append(d)
        for col in SERIAL_COLUMNS:
            d=differences(states[f"serial_{tv}_col{col:03d}"]["mu"],
                states[f"independent_{tv}_seed0"]["mu"][:,col:col+1],cf_mask[:,col:col+1])
            d.update(tv=tv,column_python=col);cf_serial.append(d)
    cf_aggregate={}
    for case in CASES:
        vals=[records[f"{case[0]}_seed{s}"] for s in config["seeds"]]
        cf_aggregate[case[0]]={metric:dict(mean=float(np.mean([x[metric] for x in vals])),sd=float(np.std([x[metric] for x in vals],ddof=1)))
            for metric in ["cf_window_mse_db2","cf_mape_percent","cf_window_neighbor_difference","wall_sec","ms_per_update"]}
    summary["cf_window"]=dict(aggregate=cf_aggregate,matched_differences=cf_pairs,serial_checks=cf_serial,
        pixels=int(cf_mask.sum()),depth_pixels_per_aline=100,alines=500,
        depth_rows_python=[int(np.where(cf_mask)[0].min()),int(np.where(cf_mask)[0].max())],
        positive_cf_reference_pixels=int(cf_mask[:,np.isfinite(data["cf_mu"])&(data["cf_mu"]>0)].sum()),
        timing_scope="Whole B-scan optimization; metrics and displayed maps restricted to the supplied CF window",
        lateral_tv_map_changes=[differences(states[f"joint_axial_lateral_tv_seed{s}"]["mu"],
            states[f"independent_axial_tv_seed{s}"]["mu"],cf_mask) for s in config["seeds"]])
    write_json(RUN/"summary.json",summary)
    with (RUN/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report(summary)
    supplement(summary)
    export(config,data,records,states,expected)
    print(f"Verified {len(records)} R3.6 fits, including eight separately executed A-lines.")


def export(config,data,records,states,expected):
    """Export everything S1(c) needs; its notebook needs no raw run access."""
    STAGE.mkdir(parents=True,exist_ok=True)
    keys=[c[0] for c in CASES];seeds=config["seeds"]
    arrays=dict(data,cases=np.array(keys),seeds=np.array(seeds))
    for field in ["mu","history"]:
        arrays[field]=np.stack([np.stack([states[f"{case}_seed{s}"][field] for s in seeds]) for case in keys])
    for field in ["wall_sec","actual_column_mse","cf_mape_percent","cf_window_mse_db2","cf_window_neighbor_difference"]:
        arrays[field]=np.array([[records[f"{case}_seed{s}"][field] for s in seeds] for case in keys])
    arrays["serial_columns"]=np.array(SERIAL_COLUMNS)
    arrays["serial_mu"]=np.stack([np.stack([states[f"serial_{tv}_col{c:03d}"]["mu"][:,0]
        for c in SERIAL_COLUMNS]) for tv in ["no_tv","axial_tv"]])
    np.savez_compressed(STAGE/"results.npz",**arrays)
    for name in ["config.json","summary.json","metrics.csv","RESULTS.md","validation.json"]:
        shutil.copy2(RUN/name,STAGE/name)
    sources=[RUN/"config.json",RUN/"bscan.npz",HERE/"coupling.py",Path(__file__)]
    sources += [RUN/"results"/f"{stem}.{ext}" for stem in expected for ext in ["json","npz"]]
    write_json(STAGE/"source_manifest.json",dict(
        export_sha256=sha256(STAGE/"results.npz"),
        sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        staged={name:sha256(STAGE/name) for name in ["config.json","summary.json","validation.json"]},
        map_seed=0,display_mask="cf_mask",reported_metrics_region="cf_window",
        timing="Whole-frame batch, mean and sample SD across three initializations"))
    (STAGE/"README.md").write_text("""# S1(c): recovered attenuation and runtime

Rebuild with `bash scripts/run_REBUTTAL_single_liquid_IL_aline.sh figures`.
`figureS1c_joint_independent.ipynb` reads only this export and writes S1(c).
`results.npz` holds all 15 full-frame maps, timings, masks, CF references,
histories and eight separately executed A-line controls. The export validates
the saved predictions, MSE and regularizers against all 23 frozen runs first.

Maps and reported attenuation/residual comparisons use only the supplied
100-pixel CF window per A-line (50,000 pixels total). Maps show seed 0 with one
common 0–3 mm^-1 color scale (values above 3 saturate). The five maps appear
in one row with a shared rectangular vertical color bar. Upper-right annotations
give full B-scan runtime as mean ± sample SD; N = 3 runs denotes three different
initializations, each with 20,000 Adam updates. Independent
fits execute in a batch; these are not 500 sequential-fit timings.
The lateral-TV fit changes the objective. CF is a reference estimate and
also contributes to the fixed gain. See RESULTS.md for numerical comparisons.
""")


def report(s):
    s=s["cf_window"]
    lines=["# R3.6: matched joint versus independent A-line optimization","",
           "One measured 1% H B-scan (MATLAB frame 256), all 500 A-lines; Adam, 20,000 updates, three seeds, float64. The signal, masks, gain, bounds, initialization and global normalization counts are matched. Independent A-lines use separate objectives and elementwise optimizer states, evaluated in parallel as a batch. Reported whole-frame timing is not a serial 500-A-line timing.","",
           "## Degree of coupling","",
           "- Beer–Lambert transport couples depths within an A-line; it does not transport light across A-lines.",
           "- Fixed preprocessing and calibration can correlate data across A-lines without introducing a cross-A-line term in the fitted objective.",
           "- Axial TV stays within an A-line; lateral TV couples neighboring A-lines.",
           "- Optimizing or profiling one shared frame gain can couple A-lines; this experiment holds supplied per-A-line gains fixed.",
           "- With no lateral term or shared optimized parameter, the joint objective is a sum of weighted per-A-line contributions. The minimizers agree mathematically. Finite-step algorithms and floating-point execution need not produce bitwise-identical states.","",
           "## Measured comparison within the CF range","",
           "Maps, MSE, map differences and neighboring-pixel differences use the supplied 100-pixel CF window in each A-line (50,000 pixels). CF MAPE additionally requires a positive finite CF reference (49,300 pixels). Runtime still measures optimization of the full B-scan.","",
           "| Case | CF-window MSE (dB²) | CF MAPE (%) | CF-window neighbor difference | Whole-frame seconds, mean ± SD |","|---|---:|---:|---:|---:|"]
    for key,r in s["aggregate"].items():
        lines.append(f"| {key} | {r['cf_window_mse_db2']['mean']:.6g} | {r['cf_mape_percent']['mean']:.3f} | {r['cf_window_neighbor_difference']['mean']:.5g} | {r['wall_sec']['mean']:.2f} ± {r['wall_sec']['sd']:.2f} |")
    lines += ["","| Matched setting | Mean map MAE across seeds (mm⁻¹) | Largest pixel difference (mm⁻¹) | Largest MSE difference (dB²) |","|---|---:|---:|---:|"]
    for tv,items in s["matched_differences"].items():
        lines.append(f"| {tv} | {np.mean([x['mae'] for x in items]):.6g} | {max(x['max'] for x in items):.6g} | {max(x['mse_abs_difference'] for x in items):.6g} |")
    lines += ["", "Eight additional fits ran A-lines 1, 167, 334, and 500 separately at seed 0, with TV off and axial TV only. Their map comparisons with the independent batch are recorded in summary.json; they verify actual separate execution without treating eight lines as a measured 500-line serial runtime.", "",
              "CF MAPE uses only pixels with a valid positive CF reference and the supplied CF comparison mask. CF also contributes to fixed gain. This is conditional reference agreement, not independent ground truth.","",
              "The lateral-TV case changes the objective. Its difference from independent axial-TV fits measures the effect of adding that coupling term; it is not an optimization-error comparison at a common objective. S1(c) shows recovered maps at seed 0 within the CF window, with a common 0–3 mm⁻¹ color scale (larger values saturate), and whole-frame runtimes.","",
              "This test concerns Adam on one processed real frame. L-BFGS-B history, CMA-ES covariance, global line searches, shared stopping rules, or shared nuisance-parameter estimation can produce further differences between joint and independent implementations. The single-A-line scaling plots alone did not test these effects.",""]
    (RUN/"RESULTS.md").write_text("\n".join(lines))


def supplement(s):
    s=s["cf_window"]
    no=s["matched_differences"]["no_tv"];z=s["matched_differences"]["axial_tv"]
    text=r'''\subsection*{Matched joint and independent fits}
\label{app:optimizer_coupling}
The forward model has no cross-A-line transport. With fixed calibration,
gain and masks, its data term is a sum of per-A-line losses with fixed
valid-pixel weights. Axial TV also decomposes by A-line. Lateral TV adds
terms $|\mu_{z,a+1}-\mu_{z,a}|$ that couple neighboring A-lines. A shared
gain that is optimized or re-estimated from the whole B-scan can couple
all A-lines. Correlation in a fixed despeckled target alone does not create
such a cross-A-line optimization term.
Without lateral TV or shared optimized parameters,
$\mathcal L_{\mathrm{joint}}=\sum_a\mathcal L_a(\mu_a)$, where each
contribution retains the global normalizers described below. Joint and
independent fits therefore share the same minimizers.

Figure~\ref{fig:optimizer_coupling} compares recovered attenuation and runtime.
We compared joint and independent fits on the same complete 1\% H liquid
phantom B-scan (frame 256, 500 A-lines), using Adam for 20,000 updates,
three initializations, and float64 arithmetic. The measured signal, masks,
fixed per-A-line gain, bounds and initialization were identical. We tested
no TV, axial TV only, and a joint fit with axial plus lateral TV. Independent
fits used separate A-line objectives and elementwise Adam states evaluated
in a batch. Each contribution retained the full-frame MSE and axial-edge
normalizers, so its sum exactly matched the joint objective without lateral
TV and preserved Adam's sensitivity to objective scaling. Four prespecified
A-lines were also fitted separately, with TV off and axial TV only, to check
the batched implementation. Whole-frame timing refers to the joint or
independent batch, not to 500 sequential fits.

Displayed maps and all reported attenuation and residual comparisons use only
the supplied 100-pixel CF window per A-line (50,000 pixels); runtime remains
the measured whole-frame optimization time.
Within this window, the no-TV joint and independent fits differed by a mean absolute
@NO_MAE@~mm$^{-1}$. With axial TV this difference was @Z_MAE@~mm$^{-1}$,
averaged across seeds. The largest individual pixel differences were
@NO_MAX@ and @Z_MAX@~mm$^{-1}$, respectively; the largest MSE differences
were @NO_MSE@ and @Z_MSE@~dB$^2$. Thus the mathematical decomposition
does not imply bitwise-identical finite-step fits. Adding lateral TV changed
the objective and reduced the mean absolute difference between neighboring
A-lines, as quantified below.
These results concern Adam on this frame; other solvers' global search,
covariance or quasi-Newton updates can depend on how parameters are grouped.

\begin{center}
\small
\textbf{CF-window comparisons and whole-frame runtime}\\[5pt]
\begin{tabular}{lrrrr}
\hline
Fit & CF-window MSE & CF MAPE & Neighbor difference & Time (s) \\
 & (dB$^2$) & (\%) & (mm$^{-1}$) & mean $\pm$ SD \\
\hline
@ROWS@
\hline
\end{tabular}
\end{center}
MSE, CF MAPE and neighbor differences are means over $n=3$ initializations;
time is the mean and sample standard deviation on NVIDIA V100 GPUs for
20,000 Adam updates. CF MAPE uses the 49,300 pixels with positive finite CF references and is
conditional on the fixed CF-derived gain. The neighbor difference is the
mean absolute attenuation difference across lateral edges with both endpoints
inside the CF window. The
axial and lateral TV weights were each 1 when enabled; the full objective
was multiplied by 1,000 and Adam used a learning rate of 0.01.
The separate one-A-line controls differed from the independent batch by
at most @SERIAL_MAE@~mm$^{-1}$ in mean absolute map difference and
@SERIAL_MAX@~mm$^{-1}$ at an individual pixel across the eight controls.
'''
    for prefix,items in [("NO",no),("Z",z)]:
        text=text.replace(f"@{prefix}_MAE@",f"{np.mean([x['mae'] for x in items]):.6f}")
        text=text.replace(f"@{prefix}_MAX@",f"{max(x['max'] for x in items):.4f}")
        text=text.replace(f"@{prefix}_MSE@",f"{max(x['mse_abs_difference'] for x in items):.6f}")
    rows=[]
    labels=["Joint, no TV", "Independent, no TV", "Joint, axial TV",
            "Independent, axial TV", "Joint, axial + lateral TV"]
    for case,label in zip(CASES,labels):
        r=s["aggregate"][case[0]]
        rows.append(f"{label} & {r['cf_window_mse_db2']['mean']:.6f} & "
                    f"{r['cf_mape_percent']['mean']:.3f} & {r['cf_window_neighbor_difference']['mean']:.5f} & "
                    f"${r['wall_sec']['mean']:.2f} \\pm {r['wall_sec']['sd']:.2f}$ " + r"\\")
    text=text.replace("@ROWS@","\n".join(rows))
    text=text.replace("@SERIAL_MAE@",f"{max(x['mae'] for x in s['serial_checks']):.6f}")
    text=text.replace("@SERIAL_MAX@",f"{max(x['max'] for x in s['serial_checks']):.5f}")
    assert "@" not in text
    (RUN/"supplementary_coupling.tex").write_text(text)


if __name__=="__main__":main()
