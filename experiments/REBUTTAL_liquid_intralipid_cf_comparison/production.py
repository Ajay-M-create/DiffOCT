"""Prepare, check and run 5,120 full-depth phantom frames with fixed CF gain."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import h5py
import numpy as np
import tifffile
from pilot import HERE, SOURCE_FILES, checked_run, prepare, fit, sha256, write_json
from plot_alignment import DATA

CONCENTRATIONS = [.5, 1., 1.5, 2., 3.]
EXTRA_SOURCE = ['production.py', 'production_report.py', 'audit_round2_alignment.py',
                'production_diagnostics.py']
# The single pipeline launcher lives in the repository's scripts/ folder; it is snapshotted too.
LAUNCHER = HERE.parents[1] / 'scripts' / 'run_REBUTTAL_liquid_intralipid_production.sh'


def read(path):
    return json.loads(Path(path).read_text())


def atomic_npz(path, **arrays):
    path=Path(path);temporary=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def verify_snapshot(series):
    manifest=read(series/'manifest.json')
    for name,h in manifest['code_sha256'].items():
        if sha256(series/'code'/name)!=h:raise RuntimeError('Frozen code changed: '+name)
    for source in manifest['sources']:
        p=Path(source['path'])
        if p.stat().st_size!=source['size_bytes'] or p.stat().st_mtime_ns!=source['mtime_ns']:
            raise RuntimeError('Source changed: '+str(p))
    return manifest


def create(series):
    from validate import validate
    from production_diagnostics import RULES
    series=checked_run(series);series.mkdir(exist_ok=False)
    for name in ['code','logs','shards','preflight','diagnostics']:(series/name).mkdir()
    sources={};entries=[];volumes=[];audit=[]
    with h5py.File(DATA/'CF_Method_Outputs.mat') as cf,h5py.File(DATA/'CF_Method_fit_mask.mat') as masks:
        for ci,c in enumerate(CONCENTRATIONS):
            for a in ['H','L']:
                v=2*ci+int(a=='L');token=f'{c:.1f}'.replace('.','..')
                matches=list(DATA.glob(f'*IL_{token}_LSM03_{a}*DespeckledLogInt.tif'))
                assert len(matches)==1
                image=matches[0]
                with tifffile.TiffFile(image) as tif:
                    assert len(tif.pages)==512 and tif.pages[0].shape==(500,500)
                    assert tif.pages[0].dtype==np.uint16 and 'rangeMax=[150,' in tif.pages[0].description
                    description=tif.pages[0].description
                mu=cf['mu'][v,:,:,0];g=cf['G'][v,:,:,0]
                mask=masks['zFitMask'][v].astype(bool) # y,x,z
                counts=mask.sum(axis=2);nonempty=counts>0
                first=mask.argmax(axis=2);last=499-mask[:,:,::-1].argmax(axis=2)
                assert not np.any(nonempty & (last-first+1!=counts))
                finite_gain=np.isfinite(mu)&np.isfinite(g)&(mu>0)&nonempty
                if not finite_gain.any(axis=1).all():
                    raise ValueError(f'{c}% {a} has frames without any valid CF gain: {np.where(~finite_gain.any(axis=1))[0]}')
                tag=f'{a}_{c:g}'.replace('.','p')
                volume=dict(concentration_percent=c,acquisition=a,python_volume=v,tag=tag,tiff=str(image.resolve()))
                volumes.append(volume)
                widths,freq=np.unique(counts,return_counts=True)
                audit.append(dict(**volume,tiff_db_range=[0,150],tiff_dtype='uint16',pages=512,
                       description=description,fit_width_histogram={str(k):int(n) for k,n in zip(widths,freq)},
                       cf_gain_invalid_alines_per_frame=(~finite_gain).sum(axis=1).tolist(),
                       empty_fit_alines=int((~nonempty).sum()),frames_with_no_cf_gain=0))
                sources[str(image.resolve())]=image
                for worker in range(8):
                    frames=list(range(worker*64,(worker+1)*64))
                    entries.append(dict(**volume,worker=worker,python_frames=frames,
                        run=str(series/'shards'/f'{tag}_{frames[0]:03d}_{frames[-1]:03d}')))
                del mask
    for name in ['CF_Method_Outputs.mat','CF_Method_fit_mask.mat','CF_Method_tom_predict.mat','IntMask.mat']:
        source=DATA/name;sources[str(source.resolve())]=source
    source_records=[]
    for path in sources.values():
        before=path.stat()
        source_records.append(dict(path=str(path.resolve()),size_bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=sha256(path)))
        assert path.stat().st_mtime_ns==before.st_mtime_ns
    for name in [*SOURCE_FILES,*EXTRA_SOURCE]:shutil.copy2(HERE/name,series/'code'/name)
    shutil.copy2(LAUNCHER,series/'code'/LAUNCHER.name)
    for name in ['PLAN.md','HANDOFF_fixed_cf_gain.md']:shutil.copy2(HERE/name,series/name)
    reference=HERE/'runs/series_fixed_cf_gain_H_20260928T163410Z'
    baseline=read(Path(read(reference/'manifest.json')['runs']['0.5'])/'config.json')
    # Model/regularization remain the tested pilot code, not a new implementation.
    old=Path(read(reference/'manifest.json')['runs']['0.5'])/'code'
    for name in ['model.py','gain.py','preprocessing.py']:
        assert sha256(old/name)==sha256(series/'code'/name),name
    settings={k:baseline[k] for k in ['learning_rate','final_lr_factor','optimizer','adam_betas','adam_eps','mse_weight','tv_weight','mu_bounds','initialization','seed_mapping','alpha','noise_domain','correction','gain_formula','gain_log_base','gain_origin','gain_fallback','geometry','dz_mm','summary_state','base_seed']}
    settings.update(iterations=15000,seed_repeats=1,primary_seed_repeat=0,batch_size=64,
                    fitting_domains=['full_depth'],gain_variant='per_aline_cf',gain_optimized=False,
                    intensity_encoding=dict(dtype='uint16',db_range=[0.,150.],decode='raw*150/65535',linear='10**(db/10)'),
                    aggregation='valid CF-window pixels -> per-A-line mean -> equal-weight paired A-line mean -> frame mean',
                    figure_display='Tukey 1.5 IQR per acquisition/concentration/method; retained density and retained mean marker only; no individual points; primary regressions unfiltered')
    write_json(series/'config.json',settings)
    write_json(series/'diagnostic_rules.json',RULES)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),series=str(series),volumes=volumes,entries=entries,
                  expected_frames=5120,frames_per_gpu=64,workers=8,concentrations=CONCENTRATIONS,
                  sources=source_records,code_sha256={n:sha256(series/'code'/n) for n in [*SOURCE_FILES,*EXTRA_SOURCE,LAUNCHER.name]},
                  config_sha256=sha256(series/'config.json'),reference_series=str(reference),
                  diagnostic_rules_sha256=sha256(series/'diagnostic_rules.json'),
                  all_frames_refit='All frames are fitted in 64-frame batches; eight-frame runs are retained as references, not reused',
                  diagnostic=dict(concentration_percent=3.,acquisition='H',python_frame=219,iterations=60000,run=str(series/'diagnostics/H_3_frame220_60000')),
                  gain_disclosure='Fixed CF-derived per-A-line gain; original manuscript re-profiles gain each iteration. Slope comparison conditional on shared gain; not independent absolute-mu validation.')
    write_json(series/'manifest.json',manifest)
    write_json(series/'input_audit.json',audit)
    write_json(series/'validation.json',validate())
    write_json(series/'status.json',dict(status='prepared',expected_frames=5120,production_authorized=True))
    (series/'README.md').write_text('''# Full-depth production with fixed CF gain per A-line

Five concentrations × H/L × 512 frames = 5,120 frames. Eight GPU workers each
optimize one batch of 64 consecutive frames per volume, for all ten volumes.
15,000 iterations, seed repeat 0, no early stopping; all final states retained.
A non-blocking 3% H frame-220 check runs for 60,000 iterations alongside production.

Each shard saves `prepared.npz`, `results/full_depth/seed_0.npz`, and a
self-contained `plot_data.npz`: μ maps, observed/CF/DiffOCT intensity curves,
CF μ/intercepts, gain, masks, N/S/H, coordinates, frame identities and histories.
`plot_data_index.json` and `.npz` index all shards after completion. Individual
complete shards are also immediately usable; see their `plot_data_schema.json`.
Completion records contain checksums. Files are compressed losslessly and written
atomically by one GPU worker per shard. Plots can be regenerated without fitting.
Every frame also has a loss plot in labeled PDF/PNG pages and its full logged
history in `loss_history.npz` (every 100 iterations, including initial/final).
Post-production A-line, residual, attenuation and artifact diagnostics run only
after the primary report verifies all 5,120 saved frame fits. Their rules are
fixed before fitting in `diagnostic_rules.json`.

The primary regression uses five unfiltered volume means per acquisition. The
pooled ten-volume regression is secondary. For display only, each N=512 violin
uses Tukey 1.5 IQR filtering and shows a retained mean marker without individual
points. Thresholds, membership and counts are saved. Report μ offsets in mm⁻¹.

Fixed gain uses the CF intercept conversion, making this a concentration-slope
comparison conditional on shared gain. The manuscript's original method
re-profiles gain each iteration. Manuscript files are unchanged.
''')
    print('SERIES_DIR='+str(series),flush=True)


def prepare_entry(series,entry,iterations=15000):
    run=Path(entry['run'])
    if run.exists():
        c=read(run/'config.json');p=read(run/'provenance.json')
        assert c['python_frames']==entry['python_frames'] and c['iterations']==iterations
        assert c['acquisition_labels']==[entry['acquisition']] and c['concentration_percent']==entry['concentration_percent']
        assert c['fitting_domains']==['full_depth'] and c['gain_variant']=='per_aline_cf'
        assert sha256(run/'prepared.npz')==p['prepared_sha256']
        assert sha256(run/'config.json')==p['config_sha256']
        for n,h in p['code_sha256'].items():
            assert sha256(run/'code'/n)==h==sha256(series/'code'/n)
        return run
    return prepare(iterations,entry['concentration_percent'],('full_depth',),entry['python_frames'],entry['acquisition'],
                   run_directory=run,source_directory=series/'code',validation_result=read(series/'validation.json'))


def preflight(series):
    import torch
    from model import objective
    manifest=verify_snapshot(series)
    run=prepare_entry(series,manifest['entries'][0])
    c=read(run/'config.json');p=read(run/'provenance.json')
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    device='cuda';start,stop=c['crop_z_python_start_stop']['full_depth']
    with np.load(run/'prepared.npz') as f:
        target=torch.as_tensor(f['target_db'][:,start:stop],device=device)
        valid=torch.as_tensor(f['full_depth_loss_mask'][:,start:stop],device=device)
        transport=torch.as_tensor(f['full_depth_transport_mask'][:,start:stop],device=device)
        gain=torch.as_tensor(f['gain_db'],dtype=torch.float32,device=device)
    initial=[]
    for frame in p['frames']:
        rng=np.random.default_rng(np.random.SeedSequence([42,frame['python_volume'],frame['python_bscan'],0]))
        full=rng.uniform(0,5,(500,500)).astype(np.float32)
        initial.append(np.maximum(full[start:stop],c['mu_bounds'][0]))
    mu=torch.nn.Parameter(torch.as_tensor(np.stack(initial),device=device))
    torch.cuda.reset_peak_memory_stats()
    loss=objective(mu,target,valid,transport,c['dz_mm'],gain,c['mse_weight'],c['tv_weight'])[0]
    batched_grad=torch.autograd.grad(loss.sum(),mu)[0]
    grad_error=0.;loss_error=0.
    for b in range(64):
        single=mu[b:b+1].detach().clone().requires_grad_()
        one=objective(single,target[b:b+1],valid[b:b+1],transport[b:b+1],c['dz_mm'],gain[b:b+1],c['mse_weight'],c['tv_weight'])[0]
        gradient=torch.autograd.grad(one.sum(),single)[0]
        torch.testing.assert_close(gradient,batched_grad[b:b+1],rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(one,loss[b:b+1],rtol=2e-6,atol=1e-3)
        grad_error=max(grad_error,float((gradient-batched_grad[b:b+1]).abs().max()))
        loss_error=max(loss_error,float((one-loss[b:b+1]).abs().max()))
    del batched_grad,loss,single,one,gradient
    selected=[0,31,63];singles=[torch.nn.Parameter(mu[i:i+1].detach().clone()) for i in selected]
    params=dict(lr=c['learning_rate'],betas=tuple(c['adam_betas']),eps=c['adam_eps'])
    optimizer=torch.optim.Adam([mu],**params)
    single_optimizers=[torch.optim.Adam([q],**params) for q in singles]
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=15000,eta_min=c['learning_rate']*c['final_lr_factor'])
    single_schedules=[torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=15000,eta_min=c['learning_rate']*c['final_lr_factor']) for o in single_optimizers]
    best_mu=mu.detach().clone();best=torch.full((64,),float('inf'),device=device)
    begin=time.monotonic()
    for step in range(50):
        optimizer.zero_grad(set_to_none=True)
        value=objective(mu,target,valid,transport,c['dz_mm'],gain,c['mse_weight'],c['tv_weight'])[0]
        with torch.no_grad():
            improved=value<best;best_mu.copy_(torch.where(improved[:,None,None],mu,best_mu));best.copy_(torch.minimum(best,value))
        value.sum().backward();optimizer.step();schedule.step()
        with torch.no_grad():mu.clamp_(*c['mu_bounds'])
        for i,q,o,sc in zip(selected,singles,single_optimizers,single_schedules):
            o.zero_grad(set_to_none=True)
            l=objective(q,target[i:i+1],valid[i:i+1],transport[i:i+1],c['dz_mm'],gain[i:i+1],c['mse_weight'],c['tv_weight'])[0]
            l.sum().backward();o.step();sc.step()
            with torch.no_grad():q.clamp_(*c['mu_bounds'])
    torch.cuda.synchronize()
    mu_error=0.
    for i,q in zip(selected,singles):
        torch.testing.assert_close(q,mu[i:i+1],rtol=2e-5,atol=2e-6)
        mu_error=max(mu_error,float((q-mu[i:i+1]).abs().max()))
    check=dict(status='passed',frames_simultaneous=64,shape=list(mu.shape),test_steps=50,
        independent_gradient_frames=64,gradient_max_abs_difference=grad_error,loss_max_abs_difference=loss_error,
        independent_50_step_frames=selected,mu_max_abs_difference=mu_error,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,seconds=time.monotonic()-begin,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),test_only='No estimates from this 50-step numerical check enter production',
        model_sha256=sha256(series/'code/model.py'),prepared_sha256=sha256(run/'prepared.npz'))
    assert check['peak_gpu_memory_bytes']<.85*check['gpu_total_memory_bytes']
    write_json(series/'preflight/gpu64.json',check)
    print(json.dumps(check,indent=2),flush=True)


def export_plot_data(run):
    """Save all A-line plotting data, independent of future access to raw MAT/TIFF."""
    run=Path(run);c=read(run/'config.json');p=read(run/'provenance.json');folder=run/'results/full_depth'
    if (run/'plot_data_complete.json').exists():
        saved=read(run/'plot_data_complete.json')
        assert sha256(run/'plot_data.npz')==saved['sha256']
        return saved
    assert sha256(folder/'seed_0.npz')==read(folder/'seed_0_complete.json')['result_sha256']
    with np.load(run/'prepared.npz') as f:data={k:f[k] for k in f.files}
    with np.load(folder/'seed_0.npz') as f:state={k:f[k] for k in f.files}
    np.testing.assert_array_equal(state['gain_db'],data['gain_db'].astype(np.float32))
    n=data['target_db'].shape[0]
    mu=np.full(data['target_db'].shape,np.nan,dtype=np.float32)
    start,stop=int(state['crop_start']),int(state['crop_stop']);mu[:,start:stop]=state['mu_final']
    transport=data['full_depth_transport_mask'];valid=data['full_depth_loss_mask'];comparison=data['comparison_mask']
    diff_db=np.full(mu.shape,np.nan,dtype=np.float32);diff_measured_db=np.full_like(diff_db,np.nan)
    cf_corrected=np.full_like(diff_db,np.nan);cf_analytic=np.full_like(diff_db,np.nan)
    exported=[];cf_pred_valid=np.zeros(mu.shape,dtype=bool)
    with h5py.File(DATA/'CF_Method_tom_predict.mat') as cf:
        for b,frame in enumerate(p['frames']):
            active=np.where(transport[b],mu[b],0).astype(np.float64)*c['dz_mm']
            tau=np.cumsum(active,axis=0)-active
            predicted=10*np.log10(np.where(transport[b],mu[b],1).astype(np.float64))-20/np.log(10)*tau+state['gain_db'][b][None,:]
            mse=np.mean((predicted[valid[b]]-data['target_db'][b][valid[b]])**2)
            np.testing.assert_allclose(mse,state['mse_final'][b],rtol=2e-4,atol=2e-5)
            diff_db[b][transport[b]]=predicted[transport[b]].astype(np.float32)
            measured=10*np.log10(10**(predicted/10)*(data['sensitivity'][b]*data['confocal'][b])[:,None]+data['noise'][b,:,None])
            diff_measured_db[b][transport[b]]=measured[transport[b]].astype(np.float32)
            raw=cf['tom'][frame['python_volume'],frame['python_bscan']].T.copy()
            exported.append(raw)
            j=(raw.astype(np.float64)-data['noise'][b,:,None])/(data['sensitivity'][b]*data['confocal'][b])[:,None]
            keep=np.isfinite(j)&(j>0);cf_pred_valid[b]=keep
            cf_corrected[b][keep]=(10*np.log10(j[keep])).astype(np.float32)
            cf_analytic[b]=((10/np.log(10))*(data['cf_G_prime'][b][None,:]-2*data['cf_mu'][b][None,:]*c['dz_mm']*np.arange(500)[:,None])).astype(np.float32)
    mu[~transport]=np.nan
    counts=comparison.sum(axis=1)
    aline=np.divide(np.where(comparison,mu,0).sum(axis=1,dtype=np.float64),counts,out=np.full(counts.shape,np.nan),where=counts>0)
    paired=(counts>0)&np.isfinite(data['cf_mu'])&(data['cf_mu']>0)&~data['gain_fallback']
    history=read(folder/'seed_0_history.json')
    arrays=dict(mu_final=mu,diffoct_mu_cf_window_mean=aline,paired_alines=paired,
        diffoct_prediction_corrected_db=diff_db,diffoct_prediction_measured_db=diff_measured_db,
        cf_prediction_measured_linear=np.stack(exported),cf_prediction_corrected_db=cf_corrected,
        cf_prediction_corrected_valid=cf_pred_valid,cf_analytic_line_corrected_db=cf_analytic,
        python_bscan=np.array(c['python_frames']),matlab_bscan=np.array(c['python_frames'])+1,
        python_volume=np.array([f['python_volume'] for f in p['frames']]),
        concentration_percent=np.float64(c['concentration_percent']),acquisition=np.array(c['acquisition_labels'][0]),
        z_python=np.arange(500),z_matlab=np.arange(1,501),depth_mm=np.arange(500)*c['dz_mm'],
        x_python=np.arange(500),x_matlab=np.arange(1,501),lateral_mm=np.arange(500)*p['calibration']['dx_um']/1000,
        dz_mm=np.float64(c['dz_mm']),dx_mm=np.float64(p['calibration']['dx_um']/1000),
        zFP_px=np.array([f['zFP_px'] for f in p['frames']]),zR_px=np.float64(p['calibration']['zR_px']),
        original_cf_window_start_python=data['original_fit_mask'].argmax(axis=1),
        gain_db_used=state['gain_db'],fit_mse_db2=state['mse_final'],objective_final=state['objective_final'],tv_final=state['tv_final'],
        iterations=np.int64(c['iterations']),seed_repeat=np.int64(0),base_seed=np.int64(42),alpha=np.float64(1),
        history_iteration=np.array([h['iteration'] for h in history]),
        history_mse=np.array([h['mse'] for h in history]),history_objective=np.array([h['objective'] for h in history]),
        history_tv=np.array([h['tv'] for h in history]),history_mean_mu=np.array([h['mean_mu'] for h in history]),
        **data)
    atomic_npz(run/'plot_data.npz',**arrays)
    schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()}
    write_json(run/'plot_data_schema.json',dict(arrays=schema,axes='Frame,z,x for maps; frame,x for A-line values; frame,z for calibration curves',
        mu_units='mm^-1; NaN outside modeled transport',intensity_db='10 log10(linear intensity)',
        gain_db='Prepared float64 CF-derived gain; gain_db_used is actual float32 optimization gain',
        depth_origin='Absolute zero-based axial sample coordinate; optical-depth integration starts at original_cf_window_start_python',
        prediction_support='DiffOCT predictions only on modeled transport; CF analytic line extrapolated across depth; original_fit_mask identifies CF fit support',
        cf_prediction='cf_prediction_measured_linear is the original supplied CF tom export, preserved in its stored dtype',
        example="b = np.flatnonzero(d['python_bscan'] == frame)[0]; mu = d['mu_final'][b,:,x]; cf_mu = d['cf_mu'][b,x]; measured = d['measured_db'][b,:,x]; cf = d['cf_prediction_measured_linear'][b,:,x]; diff = d['diffoct_prediction_measured_db'][b,:,x]"))
    complete=dict(sha256=sha256(run/'plot_data.npz'),frames=n,shape=list(mu.shape),bytes=(run/'plot_data.npz').stat().st_size,
                  prepared_sha256=p['prepared_sha256'],result_sha256=sha256(folder/'seed_0.npz'),intensity_mse_verified=True)
    write_json(run/'plot_data_complete.json',complete)
    return complete


def save_loss_curves(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    run=Path(run);c=read(run/'config.json')
    hist=read(run/'results/full_depth/seed_0_history.json')
    steps=np.array([h['iteration'] for h in hist]);total=np.array([h['objective'] for h in hist])
    mse=np.array([h['mse'] for h in hist]);tv=np.array([h['tv'] for h in hist])
    data_loss=mse*c['mse_weight'];tv_loss=tv*c['tv_weight']
    np.testing.assert_allclose(total,data_loss+tv_loss,rtol=2e-6,atol=1e-3)
    atomic_npz(run/'loss_history.npz',iteration=steps,python_bscan=np.array(c['python_frames']),
               matlab_bscan=np.array(c['python_frames'])+1,total_loss=total,data_loss=data_loss,
               tv_loss=tv_loss,mse_db2=mse,tv_unweighted=tv,mean_mu=np.array([h['mean_mu'] for h in hist]),
               mse_weight=np.float64(c['mse_weight']),tv_weight=np.float64(c['tv_weight']),
               acquisition=np.array(c['acquisition_labels'][0]),concentration_percent=np.float64(c['concentration_percent']))
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
    n=len(c['python_frames']);figures=[]
    temporary=run/'loss_curves.tmp.pdf'
    with PdfPages(temporary) as pdf:
        for page,first in enumerate(range(0,n,16),1):
            count=min(16,n-first);ncols=min(4,count);nrows=int(np.ceil(count/ncols))
            fig,axes=plt.subplots(nrows,ncols,figsize=(max(7.,3.3*ncols),max(3.5,2.3*nrows)),squeeze=False,layout='constrained')
            for j,ax in enumerate(axes.flat):
                b=first+j
                if b>=n:ax.set_visible(False);continue
                ax.semilogy(steps,total[:,b],color='black',lw=1,label='Total')
                ax.semilogy(steps,data_loss[:,b],color='#3487da',lw=.8,label='Data')
                ax.semilogy(steps,tv_loss[:,b],color='#e98850',lw=.8,label='TV')
                ax.set(title=f"Frame {c['python_frames'][b]+1} (MATLAB)",xlabel='Iteration',ylabel='Weighted loss')
                ax.grid(alpha=.18)
                if j==0:ax.legend(fontsize=7)
            fig.suptitle(f"{c['concentration_percent']:g}% {c['acquisition_labels'][0]} · full depth · fixed CF gain per A-line · {c['iterations']:,} iterations",fontsize=11)
            pdf.savefig(fig);filename=f'loss_curves_{page:02d}.png';fig.savefig(run/filename,dpi=140)
            figures.append(filename);plt.close(fig)
    temporary.replace(run/'loss_curves.pdf')
    write_json(run/'loss_curves_complete.json',dict(frames=n,pdf='loss_curves.pdf',png_pages=figures,
               data='loss_history.npz',history_sha256=sha256(run/'loss_history.npz'),every_frame_has_curve=True))


def worker(series,index):
    manifest=verify_snapshot(series)
    assert read(series/'preflight/gpu64.json')['status']=='passed'
    entries=[e for e in manifest['entries'] if e['worker']==index];assert len(entries)==10
    for entry in entries:
        run=Path(entry['run'])
        try:
            prepare_entry(series,entry)
            status_path=run/'status_full_depth.json'
            if not status_path.exists() or read(status_path)['status']!='complete':fit(run,'full_depth')
            saved=export_plot_data(run)
            save_loss_curves(run)
            write_json(run/'status.json',dict(status='complete',frames=64,iterations=15000,plot_data=saved,
                         slurm_job_id=os.environ.get('SLURM_JOB_ID'),worker=index,completed_utc=datetime.now(timezone.utc).isoformat()))
            print(f'COMPLETE {entry["tag"]} worker={index} frames={entry["python_frames"][0]}..{entry["python_frames"][-1]}',flush=True)
            gc.collect()
        except Exception as error:
            write_json(series/'logs'/f'worker_{index}_failure.json',dict(entry=entry,error=repr(error)))
            raise


def diagnostic(series):
    manifest=verify_snapshot(series);d=manifest['diagnostic']
    entry=dict(concentration_percent=3.,acquisition='H',python_frames=[219],run=d['run'])
    run=prepare_entry(series,entry,iterations=60000)
    fit(run,'full_depth');saved=export_plot_data(run)
    save_loss_curves(run)
    write_json(run/'status.json',dict(status='complete',frames=1,iterations=60000,plot_data=saved))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['create','preflight','worker','diagnostic']);p.add_argument('series',type=Path)
    p.add_argument('--worker',type=int,default=None);args=p.parse_args();series=checked_run(args.series)
    if args.mode=='create':create(series)
    elif args.mode=='preflight':preflight(series)
    elif args.mode=='diagnostic':diagnostic(series)
    else:worker(series,args.worker if args.worker is not None else int(os.environ['SLURM_PROCID']))
