"""Replot production from saved NPZ files; never runs optimization."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read(path):return json.loads(Path(path).read_text())


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def regression(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    slope,intercept=np.polyfit(x,y,1)
    direct=np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)
    np.testing.assert_allclose(slope,direct,rtol=0,atol=1e-12)
    return dict(slope_mm_inv_per_percentage_point=float(slope),intercept_mm_inv=float(intercept),
                r_squared=float(1-np.sum((y-(intercept+slope*x))**2)/np.sum((y-y.mean())**2)),n_volume_means=len(x))


def savefig(fig,series,name):
    for ext in ['pdf','png','svg']:fig.savefig(series/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def load_frames(series,manifest):
    pending=[]
    for entry in manifest['entries']:
        run=Path(entry['run'])
        if not (run/'status.json').exists() or read(run/'status.json')['status']!='complete':pending.append(entry)
    if pending:
        write_json(series/'status.json',dict(status='incomplete',expected_frames=5120,incomplete_shards=pending,
                                           note='No completed shard is discarded; see worker failure logs.'))
        raise RuntimeError(f'{len(pending)} production shards are incomplete')
    rows=[];index=[];verification=[]
    for entry in manifest['entries']:
        run=Path(entry['run']);config=read(run/'config.json');prov=read(run/'provenance.json')
        assert config['iterations']==15000 and config['fitting_domains']==['full_depth']
        assert config['gain_variant']=='per_aline_cf' and config['batch_size']==64
        assert config['python_frames']==entry['python_frames']
        assert sha256(run/'config.json')==prov['config_sha256']
        for name,h in prov['code_sha256'].items():assert sha256(run/'code'/name)==h==manifest['code_sha256'][name]
        completion=read(run/'plot_data_complete.json');loss_complete=read(run/'loss_curves_complete.json')
        assert sha256(run/'plot_data.npz')==completion['sha256']
        assert sha256(run/'loss_history.npz')==loss_complete['history_sha256']
        assert loss_complete['frames']==64 and loss_complete['every_frame_has_curve']
        saved=read(run/'results/full_depth/summary.json')
        with np.load(run/'plot_data.npz') as f:
            cf=f['cf_mu'].astype(np.float64);mu=f['diffoct_mu_cf_window_mean'];paired=f['paired_alines']
            np.testing.assert_array_equal(f['gain_db_used'],f['gain_db'].astype(np.float32))
            assert f['mu_final'].shape==(64,500,500)
            assert int(f['iterations'])==15000
            gain=f['gain_db_used'];fallback=f['gain_fallback'];exclude=f['excluded_mask'].sum(axis=(1,2))
            assert np.array_equal(f['python_bscan'],entry['python_frames'])
            for b,frame in enumerate(entry['python_frames']):
                if not paired[b].any():
                    rows.append(dict(acquisition=entry['acquisition'],concentration_percent=entry['concentration_percent'],python_volume=entry['python_volume'],python_bscan=frame,matlab_bscan=frame+1,status='empty_comparison',cf_mean_mm_inv=None,diffoct_mean_mm_inv=None,bias_mm_inv=None,relative_bias_percent=None,paired_alines=0,excluded_pixels=int(exclude[b]),gain_fallback_alines=int(fallback[b].sum()),gain_median_db=float(np.median(gain[b])),fit_mse_db2=float(f['fit_mse_db2'][b]),plot_data=str(run/'plot_data.npz'),batch_index=b))
                    continue
                a=float(cf[b,paired[b]].mean());d=float(mu[b,paired[b]].mean())
                np.testing.assert_allclose([a,d],[saved[b]['cf_mean_mm_inv'],saved[b]['diffoct_mean_mm_inv']],rtol=0,atol=1e-10)
                rows.append(dict(acquisition=entry['acquisition'],concentration_percent=entry['concentration_percent'],python_volume=entry['python_volume'],python_bscan=frame,matlab_bscan=frame+1,status='valid',cf_mean_mm_inv=a,diffoct_mean_mm_inv=d,bias_mm_inv=d-a,relative_bias_percent=100*(d-a)/a,paired_alines=int(paired[b].sum()),excluded_pixels=int(exclude[b]),gain_fallback_alines=int(fallback[b].sum()),gain_median_db=float(np.median(gain[b])),fit_mse_db2=float(f['fit_mse_db2'][b]),plot_data=str(run/'plot_data.npz'),batch_index=b))
        index.append(dict(**entry,plot_data=str(run/'plot_data.npz'),plot_data_sha256=completion['sha256'],
                    schema=str(run/'plot_data_schema.json'),loss_history=str(run/'loss_history.npz'),loss_pdf=str(run/'loss_curves.pdf'),
                    result_npz=str(run/'results/full_depth/seed_0.npz'),prepared_npz=str(run/'prepared.npz')))
        verification.append(dict(run=str(run),frames=64,plot_data_checksum=True,fixed_gain_verified=True,frame_means_recomputed=True,loss_curves_all_frames=True))
    assert len(rows)==5120 and len(set((r['python_volume'],r['python_bscan']) for r in rows))==5120
    return rows,index,verification


def summarize(rows,manifest):
    groups=[];members=[];thresholds=[]
    for v in manifest['volumes']:
        rr=sorted([r for r in rows if r['python_volume']==v['python_volume']],key=lambda r:r['python_bscan'])
        assert len(rr)==512 and [r['python_bscan'] for r in rr]==list(range(512))
        valid=np.array([r['status']=='valid' for r in rr]);values={}
        for method in ['cf','diffoct']:
            raw=np.array([r[method+'_mean_mm_inv'] if r[method+'_mean_mm_inv'] is not None else np.nan for r in rr])
            values[method]=raw[valid]
            q1,q3=np.quantile(raw[valid],[.25,.75]);lo=q1-1.5*(q3-q1);hi=q3+1.5*(q3-q1)
            keep=valid & (raw>=lo)&(raw<=hi)
            thresholds.append(dict(acquisition=v['acquisition'],concentration_percent=v['concentration_percent'],method=method,
                n_total=512,n_valid=int(valid.sum()),q1=float(q1),q3=float(q3),lower=float(lo),upper=float(hi),
                n_retained=int(keep.sum()),n_removed=int(valid.sum()-keep.sum()),retained_mean_mm_inv=float(raw[keep].mean())))
            members.append(dict(volume=v['python_volume'],method=method,values=raw,retained=keep))
        cf,diff=values['cf'],values['diffoct']
        groups.append(dict(acquisition=v['acquisition'],concentration_percent=v['concentration_percent'],python_volume=v['python_volume'],
            n_frames=512,n_valid_frames=int(valid.sum()),n_empty_or_invalid_frames=int((~valid).sum()),
            cf_mean_mm_inv=float(cf.mean()),cf_sd_mm_inv=float(cf.std(ddof=1)),
            diffoct_mean_mm_inv=float(diff.mean()),diffoct_sd_mm_inv=float(diff.std(ddof=1)),
            bias_mm_inv=float(np.mean(diff-cf)),relative_bias_percent=float(100*np.mean(diff-cf)/cf.mean())))
    regs=[];offset=[]
    for a in ['H','L','H+L']:
        gg=[g for g in groups if a=='H+L' or g['acquisition']==a];x=[g['concentration_percent'] for g in gg]
        for method in ['cf','diffoct']:
            regs.append(dict(acquisition=a,method=method,analysis='secondary pooled' if a=='H+L' else 'primary',
                             **regression(x,[g[method+'_mean_mm_inv'] for g in gg])))
        bias=np.array([g['bias_mm_inv'] for g in gg]);trend=regression(x,bias)
        offset.append(dict(acquisition=a,min_signed_offset_mm_inv=float(bias.min()),max_signed_offset_mm_inv=float(bias.max()),
                           min_magnitude_mm_inv=float(np.abs(bias).min()),max_magnitude_mm_inv=float(np.abs(bias).max()),
                           slope_of_offset_vs_concentration=trend['slope_mm_inv_per_percentage_point']))
    return groups,regs,offset,members,thresholds


def plot_violins(series,manifest,members,regs,thresholds,cf_offset_mm_inv=0.,
                 figure_name='violin_cf_vs_diffoct_full_depth_HL'):
    colors={'cf':'#ff9966','diffoct':'#74b4ff'}
    fig,ax=plt.subplots(figsize=(15,7),layout='constrained')
    concentrations=np.array(manifest['concentrations'])
    ymax=0
    for acquisition,shift,style in [('H',-.09,'-'),('L',.09,'--')]:
        for method,offset in [('cf',-.04),('diffoct',.04)]:
            values=[]
            for c in concentrations:
                v=next(v['python_volume'] for v in manifest['volumes'] if v['acquisition']==acquisition and v['concentration_percent']==c)
                m=next(m for m in members if m['volume']==v and m['method']==method)
                values.append(m['values'][m['retained']]+(cf_offset_mm_inv if method=='cf' else 0.))
            positions=concentrations+shift+offset
            violins=ax.violinplot(values,positions=positions,widths=.075,showextrema=False,points=200)
            for body in violins['bodies']:
                body.set_facecolor(colors[method]);body.set_edgecolor(colors[method]);body.set_alpha(.9 if acquisition=='H' else .55)
            # Only the retained mean is marked; no individual frame points or fliers.
            ax.scatter(positions,[y.mean() for y in values],s=26,color='black',zorder=5)
            ymax=max(ymax,max(float(y.max()) for y in values))
            r=next(r for r in regs if r['acquisition']==acquisition and r['method']==method)
            x=np.linspace(.5,3,150);y=r['intercept_mm_inv']+r['slope_mm_inv_per_percentage_point']*x+(cf_offset_mm_inv if method=='cf' else 0.)
            ymax=max(ymax,float(y.max()));ax.plot(x,y,style,color=colors[method],lw=1.5)
        for c in concentrations:ax.text(c+shift,.012,acquisition,transform=ax.get_xaxis_transform(),ha='center',fontsize=10)
    handles=[]
    for a,ls in [('H','-'),('L','--')]:
        for method in ['cf','diffoct']:
            r=next(r for r in regs if r['acquisition']==a and r['method']==method)
            intercept=r['intercept_mm_inv']+(cf_offset_mm_inv if method=='cf' else 0.)
            shifted=' (shifted)' if method=='cf' and cf_offset_mm_inv else ''
            label=(f'{"CF" if method=="cf" else "DiffOCT"} {a}{shifted}: '
                   rf'$\mu_t = {intercept:.4f} {r["slope_mm_inv_per_percentage_point"]:+.4f}\,C$, '
                   rf'$R^2 = {r["r_squared"]:.4f}$')
            handles.append(Line2D([],[],color=colors[method],ls=ls,lw=2,label=label))
    handles.append(Line2D([],[],marker='o',color='black',ls='None',label='Mean of displayed values'))
    ax.legend(handles=handles,loc='upper left',fontsize=10,
              title='Unfiltered regressions: C in %, μₜ in mm⁻¹',title_fontsize=10)
    ax.set(xlabel='Intralipid concentration (%)',ylabel='Attenuation coefficient (mm⁻¹)',xticks=concentrations,
           xticklabels=[f'{c:.1f}' for c in concentrations],xlim=(.27,3.23),ylim=(0,ymax*1.07))
    ax.grid(alpha=.18);ax.set_axisbelow(True)
    subtitle='512 frame means per group before display filtering · H and L acquisitions'
    if cf_offset_mm_inv:subtitle+=f'\nCF display shift: {cf_offset_mm_inv:+g} mm⁻¹ (distributions, means and fitted lines)'
    fig.suptitle('Full-depth DiffOCT with fixed CF gain per A-line\n'+subtitle,fontsize=14)
    removed=sum(t['n_removed'] for t in thresholds)
    fig.supxlabel(f'Display only: Tukey 1.5×IQR within each group; {removed} of 10,240 method-specific frame values omitted. Saved estimates and regressions are unfiltered.',fontsize=9)
    savefig(fig,series,figure_name)


def diagnostic(series,manifest,rows):
    run=Path(manifest['diagnostic']['run'])
    if not (run/'status.json').exists() or read(run/'status.json')['status']!='complete':
        return dict(status='pending_or_failed',run=str(run),non_blocking=True)
    q=read(run/'results/full_depth/summary.json')[0]
    r=next(r for r in rows if r['acquisition']=='H' and r['concentration_percent']==3 and r['python_bscan']==219)
    with np.load(run/'loss_history.npz') as f:
        steps=f['iteration'];mean=f['mean_mu'][:,0];late=mean[steps>=48000]
    value=dict(status='complete',concentration_percent=3,acquisition='H',matlab_bscan=220,
        mean_15000_mm_inv=r['diffoct_mean_mm_inv'],mean_60000_mm_inv=q['diffoct_mean_mm_inv'],
        difference_15000_minus_60000_mm_inv=r['diffoct_mean_mm_inv']-q['diffoct_mean_mm_inv'],
        late_range_mm_inv=float(np.ptp(late)),late_mean_mm_inv=float(late.mean()),
        late_range_fraction=float(np.ptp(late)/late.mean()),run=str(run))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(steps,mean,c='#3487da',label='60k-budget run')
    short_run=Path(r['plot_data']).parent
    with np.load(short_run/'loss_history.npz') as f:
        b=int(np.where(f['python_bscan']==219)[0][0]);short=f['mean_mu'][-1,b]
        axes[0].scatter([15000],[short],marker='D',color='#e98850',label='15k-budget final state')
    axes[0].set(xlabel='Iteration',ylabel='Mean μ in CF window (mm⁻¹; all A-lines)');axes[0].legend(fontsize=9)
    with np.load(run/'loss_history.npz') as f:
        for key,color in [('total_loss','black'),('data_loss','#3487da'),('tv_loss','#e98850')]:axes[1].semilogy(steps,f[key][:,0],c=color,label=key)
    axes[1].set(xlabel='Iteration',ylabel='Weighted loss');axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('3% H · MATLAB frame 220 · fixed per-A-line CF gain · full depth')
    savefig(fig,series,'convergence_spot_check_3pct_H')
    return value


def report(series):
    series=Path(series);manifest=read(series/'manifest.json')
    rows,index,verification=load_frames(series,manifest)
    groups,regs,offset,members,thresholds=summarize(rows,manifest)
    write_csv(series/'frame_means.csv',rows);write_csv(series/'volume_summary.csv',groups)
    write_csv(series/'regressions.csv',regs);write_csv(series/'offset_summary.csv',offset)
    write_csv(series/'display_filter_thresholds.csv',thresholds)
    write_json(series/'plot_data_index.json',index)
    np.savez_compressed(series/'plot_data_index.npz',plot_data=np.array([r['plot_data'] for r in rows]),
         batch_index=np.array([r['batch_index'] for r in rows]),python_volume=np.array([r['python_volume'] for r in rows]),
         python_bscan=np.array([r['python_bscan'] for r in rows]),acquisition=np.array([r['acquisition'] for r in rows]),
         concentration_percent=np.array([r['concentration_percent'] for r in rows]))
    np.savez_compressed(series/'frame_summary.npz',python_volume=np.array([r['python_volume'] for r in rows]),
         python_bscan=np.array([r['python_bscan'] for r in rows]),acquisition=np.array([r['acquisition'] for r in rows]),
         concentration_percent=np.array([r['concentration_percent'] for r in rows]),
         cf_mean_mm_inv=np.array([r['cf_mean_mm_inv'] for r in rows],dtype=float),diffoct_mean_mm_inv=np.array([r['diffoct_mean_mm_inv'] for r in rows],dtype=float),
         bias_mm_inv=np.array([r['bias_mm_inv'] for r in rows],dtype=float))
    np.savez_compressed(series/'display_membership.npz',python_volume=np.array([m['volume'] for m in members]),method=np.array([m['method'] for m in members]),
         frame_values=np.stack([m['values'] for m in members]),retained=np.stack([m['retained'] for m in members]),python_bscan=np.arange(512))
    plt.rcParams.update({'font.family':'serif','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    plot_violins(series,manifest,members,regs,thresholds)
    spot=diagnostic(series,manifest,rows);write_json(series/'convergence_spot_check.json',spot)
    summary=dict(volumes=groups,regressions=regs,offsets=offset,convergence_spot_check=spot,
                 gain_disclosure=manifest['gain_disclosure'],primary_regressions_unfiltered=True,expected_frames=5120,
                 empty_or_invalid_frames=sum(g['n_empty_or_invalid_frames'] for g in groups),display_removed_counts=thresholds)
    write_json(series/'summary.json',summary);write_json(series/'verification.json',dict(shards=verification,frame_count=5120,all_unique=True))
    lines=['# Production: full-depth fitting with fixed per-A-line CF gain','',
      'Five concentrations × two acquisitions × 512 frames = **5,120 frames**. Eight Slurm array elements each optimized 64 frames simultaneously per volume. Every fit completed exactly 15,000 iterations; the final state is primary. All original arrays, fitted maps and histories are preserved.','',
      '## Unfiltered primary concentration regressions','',
      '| Acquisition | Method | Slope (mm⁻¹ per percentage point) | Intercept (mm⁻¹) | R² |','| --- | --- | ---: | ---: | ---: |']
    for r in regs:lines.append(f"| {r['acquisition']} | {r['method']} | {r['slope_mm_inv_per_percentage_point']:.6f} | {r['intercept_mm_inv']:.6f} | {r['r_squared']:.6f} |")
    lines+=['','H and L regressions use five equally weighted volume means. The pooled H+L fit is secondary and uses ten volume means. No regression or bias estimate uses display filtering.','', '## DiffOCT−CF offsets','']
    for r in offset:
        lines.append(f"- {r['acquisition']}: signed offset **{r['min_signed_offset_mm_inv']:.6f} to {r['max_signed_offset_mm_inv']:.6f} mm⁻¹**; offset slope versus concentration **{r['slope_of_offset_vs_concentration']:+.6f} mm⁻¹ per percentage point**.")
    lines+=['','Offsets are reported in absolute units; relative bias is retained only in CSV columns. `volume_summary.csv` contains each acquisition/concentration mean, frame SD, bias, and invalid-frame count.','', '## Convergence spot check','']
    if spot['status']=='complete':lines.append(f"For 3% H MATLAB frame 220, paired μ is **{spot['mean_15000_mm_inv']:.6f} mm⁻¹ at 15k** and **{spot['mean_60000_mm_inv']:.6f} mm⁻¹ at 60k**; difference **{spot['difference_15000_minus_60000_mm_inv']:+.6f} mm⁻¹**. The final-20% range of the all-A-line window mean is **{spot['late_range_mm_inv']:.8f} mm⁻¹**. This spot check does not establish convergence for every production frame.")
    else:lines.append('The non-blocking 3% H 60k diagnostic is pending or failed; its status is recorded separately.')
    lines+=['','## Violin display','',
      'Orange: CF; blue: DiffOCT. H and L appear side by side at each numeric concentration. Tukey 1.5×IQR is applied separately to each acquisition × concentration × method group of 512 frame means for display only. The density ends at retained observed extrema. Only one retained-mean marker appears per violin; there are no individual points or fliers. Lines are the unfiltered primary regressions over 0.5–3%.','',
      '| Acquisition | Concentration | Method | Removed from display | Retained |','| --- | ---: | --- | ---: | ---: |']
    for t in thresholds:lines.append(f"| {t['acquisition']} | {t['concentration_percent']:g}% | {t['method']} | {t['n_removed']} | {t['n_retained']} |")
    lines+=['','Thresholds and exact membership are saved in `display_filter_thresholds.csv` and `display_membership.npz`. All original frame results remain saved.','',
      '## Replot from NPZ files','',
      '`plot_data_index.json` and `.npz` map concentration, acquisition and frame to a self-contained 64-frame `plot_data.npz`. Each contains full μ maps, per-A-line CF μ, measured intensities, CF and DiffOCT predicted intensities in measured and corrected domains, masks, gains, CF intercepts, calibration, coordinates and histories. `plot_data_schema.json` documents fields and units. `frame_summary.npz` is a compact table for new statistical plots.','',
      'Each shard also contains `loss_history.npz` with total/data/TV loss, raw data MSE, iterations and mean μ for every frame. `loss_curves.pdf` and its PNG pages plot every frame separately.','',
      '```python','import numpy as np',"index = np.load('plot_data_index.npz')","match = (index['concentration_percent'] == 0.5) & (index['acquisition'] == 'H') & (index['python_bscan'] == 219)","i = np.flatnonzero(match)[0]","d = np.load(str(index['plot_data'][i]))","b, x = int(index['batch_index'][i]), 250  # MATLAB frame 220, A-line 251","z = d['depth_mm']","mu = d['mu_final'][b, :, x]","cf_mu = d['cf_mu'][b, x]","measured_db = d['measured_db'][b, :, x]","cf_prediction_db = 10*np.log10(d['cf_prediction_measured_linear'][b, :, x])","diffoct_prediction_db = d['diffoct_prediction_measured_db'][b, :, x]","comparison_mask = d['comparison_mask'][b, :, x]",'```','',
      'To regenerate aggregate figures/tables without fitting, run `python code/production_report.py /path/to/this/series` in the diff-oct environment.','',
      '## Method interpretation','',
      'The per-A-line CF gain was fixed to make the μ comparison share the same gain. CF fits an intercept per A-line; profiling gain freely in the tested DiffOCT objective allowed μ to shrink along a gain–μ ambiguity. This production run uses fixed CF-derived gain, whereas the original manuscript method re-profiles it each iteration. The slope comparison is conditional on shared gain and does not independently validate absolute μ.','',
      'CF-window and G1/G2 variants were explored in round 2 and set aside: the finite fitting window leaves its lower edge weakly constrained and those runs showed concentration-dependent behavior; the scalar variants alter brightness calibration and their 15k budget was not converged. They were not rerun for production. Manuscript files were not changed.','']
    (series/'README.md').write_text('\n'.join(lines))
    section=['# Production completed '+datetime.now(timezone.utc).strftime('%Y-%m-%d'),'','Full-depth only, fixed per-A-line CF gain, 5,120 frames, 15,000 iterations. All NPZ plotting data and loss curves are saved.','',
             f"[Full production report](runs/{series.name}/README.md) · [Violin figure](runs/{series.name}/violin_cf_vs_diffoct_full_depth_HL.pdf)",'']
    for r in regs:
        if r['acquisition']!='H+L':section.append(f"- {r['acquisition']} {r['method']}: slope {r['slope_mm_inv_per_percentage_point']:.6f}, intercept {r['intercept_mm_inv']:.6f} mm⁻¹, R² {r['r_squared']:.6f}.")
    section+=['']+[f"- {r['acquisition']} DiffOCT−CF offset: {r['min_signed_offset_mm_inv']:.6f} to {r['max_signed_offset_mm_inv']:.6f} mm⁻¹; offset slope {r['slope_of_offset_vs_concentration']:+.6f} mm⁻¹ per percentage point." for r in offset if r['acquisition']!='H+L']
    if spot['status']=='complete':section+=['',f"3% H frame-220 convergence check: 15k {spot['mean_15000_mm_inv']:.6f}, 60k {spot['mean_60000_mm_inv']:.6f} mm⁻¹; difference {spot['difference_15000_minus_60000_mm_inv']:+.6f} mm⁻¹."]
    section+=['','This comparison uses fixed CF-derived gain; the original manuscript method re-profiles gain. Absolute-μ validation remains conditional on shared gain. CF-window/G1/G2 runs remain diagnostic only and were set aside as documented in the handoff.','']
    (series/'RESULTS_section.md').write_text('\n'.join(section))
    experiment=next(p for p in series.parents if p.name == "REBUTTAL_liquid_intralipid_cf_comparison")
    results=experiment/'RESULTS.md';old=results.read_text();marker=f'<!-- production:{series.name} -->'
    if marker not in old:results.write_text(marker+'\n'+'\n'.join(section)+'\n---\n\n'+old)
    write_json(series/'status.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
          frame_count=5120,shards=80,loss_curves=5120,diagnostic_status=spot['status']))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('series',type=Path);report(p.parse_args().series)
