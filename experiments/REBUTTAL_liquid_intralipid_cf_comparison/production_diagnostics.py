"""Handoff section 10: post-production diagnostics from saved NPZ arrays only."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from production_report import read, write_json, write_csv, regression, savefig, sha256

K=20/np.log(10)
RULES=dict(surface_transition_threshold_db=10.,surface_transition_early_pixels=[0,5],
           surface_transition_late_pixels=[10,15],surface_transition_signal='measured_db; mean within each band then median difference over all nonempty A-lines',
           reflection_threshold_db=10.,reflection_search_pixels=[0,20],
           reflection_signal='maximum valid corrected signal in first 20 window pixels minus analytic CF line at window top; paired A-lines only',
           window_bins=[[0,5],[5,10],[10,90],[90,100]],bin_convention='zero-based half-open intervals',
           low_absolute_mu_mm_inv=.1,low_relative_mu_fraction=.25,lower_bound_tolerance_factor=1.001,
           stats_support='valid comparison pixels in paired A-lines; missing observations excluded and counted separately',
           example_frame_python=255,example_concentrations=[.5,1.,2.,3.],
           example_aline='paired A-line in frame 255 closest to the volume median of per-A-line DiffOCT window mean minus CF mu over all 512 frames',
           example_depth_max_mm=3.,color_limits_mm_inv=[-1.,1.],
           worst_frame='flagged frame with highest fraction of valid comparison pixels below 0.25*CF mu; ties: volume then frame; if no flags, highest fraction overall',
           worst_aline='paired A-line with largest count of valid low-relative-mu window pixels in the worst frame')
CMAP=LinearSegmentedColormap.from_list('mu_difference',['#2166ac','#67a9cf','#9e9e9e','#ef8a62','#b2182b'])
NORM=Normalize(-1,1)


def ls_slope(y,valid,dz):
    """Independent A-line slopes in dB/mm, with actual valid depth coordinates."""
    n=valid.sum(axis=0);z=np.arange(y.shape[0],dtype=float)[:,None]*dz
    yy=np.where(valid,y,0).astype(float)
    zsum=np.where(valid,z,0).sum(axis=0);ysum=yy.sum(axis=0)
    den=np.where(valid,z*z,0).sum(axis=0)-zsum*zsum/np.maximum(n,1)
    num=(yy*z).sum(axis=0)-zsum*ysum/np.maximum(n,1)
    return np.divide(num,den,out=np.full(n.shape,np.nan),where=(n>=2)&(den>0))


def longest_run(mask):
    current=np.zeros(mask.shape[1],dtype=int);longest=current.copy()
    for line in mask:
        current=np.where(line,current+1,0);longest=np.maximum(longest,current)
    return longest


def window_array(array,starts,length=100):
    positions=starts[None,:]+np.arange(length)[:,None]
    values=array[np.clip(positions,0,499),np.arange(500)[None,:]]
    return values,(positions>=0)&(positions<500)


def get_frame(data,b,entry):
    keys={'mu':'mu_final','pred':'diffoct_prediction_corrected_db','cfl':'cf_analytic_line_corrected_db',
          't':'target_db','meas':'measured_db','cm':'comparison_mask','lm':'full_depth_loss_mask',
          'fit':'original_fit_mask','excl':'excluded_mask','cf':'cf_mu','paired':'paired_alines'}
    f={k:data[v][b].copy() for k,v in keys.items()}
    f.update(zs=f['fit'].argmax(axis=0),dz=float(data['dz_mm']),frame=int(data['python_bscan'][b]),
             concentration=entry['concentration_percent'],acquisition=entry['acquisition'],volume=entry['python_volume'])
    return f


def colored(ax,x,y,value):
    p=np.column_stack([x,y]);seg=np.stack([p[:-1],p[1:]],axis=1);v=(value[:-1]+value[1:])/2
    good=np.isfinite(seg).all(axis=(1,2))&np.isfinite(v)
    artist=LineCollection(seg[good],cmap=CMAP,norm=NORM,linewidths=1.8)
    artist.set_array(v[good]);ax.add_collection(artist)


def aline_panel(top,bottom,f,x,extent=3.,title=None,window_only=False):
    start=int(f['zs'][x])
    if window_only:
        z=np.flatnonzero(f['fit'][:,x])
        extent=float((z[-1]-start)*f['dz'])
    else:
        z=np.arange(start,min(500,start+int(extent/f['dz'])+1))
    depth=(z-start)*f['dz']
    inwindow=f['fit'][z,x]
    valid=f['lm'][z,x]&np.isfinite(f['t'][z,x])
    measured=np.where(valid,f['t'][z,x],np.nan)
    predicted=np.where(valid,f['pred'][z,x],np.nan)
    cf_line=np.where(valid,f['cfl'][z,x],np.nan)
    top.plot(depth,measured,color='#00843d',lw=1.2,ls=(0,(4,2)),zorder=4,
             label='Measured target (corrected)')
    top.plot(depth,np.where(inwindow,cf_line,np.nan),color='black',lw=1.3,label='CF in window')
    if not window_only:
        top.plot(depth,np.where(~inwindow,cf_line,np.nan),color='black',lw=1.2,ls=':',label='CF extrapolated')
    colored(top,depth,predicted,f['mu'][z,x]-f['cf'][x])
    m=f['cm'][:,x];cf_rmse=float(np.sqrt(np.mean((f['t'][m,x]-f['cfl'][m,x])**2)))
    diff_rmse=float(np.sqrt(np.mean((f['t'][m,x]-f['pred'][m,x])**2)))
    mean=float(f['mu'][m,x].mean(dtype=np.float64))
    top.text(.98,.96,f"CF μ {f['cf'][x]:.3f} mm⁻¹\nDiffOCT μ {mean:.3f} mm⁻¹\nRMSE: CF {cf_rmse:.3f}; DiffOCT {diff_rmse:.3f} dB",transform=top.transAxes,ha='right',va='top',fontsize=8)
    top.set_title(title or f"{f['concentration']:g}% {f['acquisition']} · frame {f['frame']+1}, A-line {x+1}",fontsize=10)
    mu=np.where(valid,f['mu'][z,x],np.nan);cf=float(f['cf'][x])
    bottom.fill_between(depth,cf,mu,where=mu>=cf,color='#b2182b',alpha=.35,interpolate=True,lw=0)
    bottom.fill_between(depth,cf,mu,where=mu<cf,color='#2166ac',alpha=.35,interpolate=True,lw=0)
    bottom.plot(depth,mu,color='#555555',lw=.8);bottom.axhline(cf,color='black',lw=1.2)
    bottom.set(xlabel='Depth within CF window (mm)' if window_only else 'Depth below window start (mm)',ylim=(0,None))
    for ax in [top,bottom]:
        if not window_only:ax.axvspan(0,99*f['dz'],color='gray',alpha=.10,lw=0)
        ax.grid(alpha=.18);ax.set_xlim(0,extent)
    finite=np.concatenate([line[np.isfinite(line)] for line in [measured,predicted,cf_line]])
    top.set_ylim(float(finite.min())-1,float(finite.max())+6)
    return dict(cf_mu_mm_inv=cf,diffoct_window_mean_mm_inv=mean,cf_rmse_db=cf_rmse,diffoct_rmse_db=diff_rmse)


def run(series):
    series=Path(series);manifest=read(series/'manifest.json')
    assert read(series/'verification.json')['frame_count']==5120
    assert read(series/'diagnostic_rules.json')==RULES
    out=series/'post_diagnostics';out.mkdir(exist_ok=True)
    write_json(out/'rules.json',RULES)
    frame_metrics=[];low_rows=[];flag_rows=[];reflection_rows=[];residual_rows=[];volume_rows=[];sensitivity_frames=[]
    selected_frames={};medians={};residual_profiles={};flag_files=[]
    worst_flagged=None;worst_any=None
    required=['mu_final','diffoct_prediction_corrected_db','cf_analytic_line_corrected_db','target_db','measured_db',
              'comparison_mask','full_depth_loss_mask','original_fit_mask','excluded_mask','cf_mu','paired_alines',
              'dz_mm','python_bscan','diffoct_mu_cf_window_mean']
    for volume in manifest['volumes']:
        v=volume['python_volume'];entries=[e for e in manifest['entries'] if e['python_volume']==v]
        # Exact residual quantiles, all paired A-lines, with bounded per-volume memory.
        residual_cf=np.full((512,500,500),np.nan,dtype=np.float32)
        residual_diff=np.full_like(residual_cf,np.nan)
        transitions=np.zeros(512);frame_flags=np.zeros(512,dtype=bool);reflections=np.zeros((512,500),dtype=bool)
        line_differences=[];volume_frames=[]
        for entry in entries:
            run_path=Path(entry['run']);complete=read(run_path/'plot_data_complete.json')
            assert sha256(run_path/'plot_data.npz')==complete['sha256']
            with np.load(run_path/'plot_data.npz') as archive:data={k:archive[k] for k in required}
            for b,y in enumerate(data['python_bscan']):
                y=int(y);f=get_frame(data,b,entry);dz=f['dz'];paired=f['paired'];mask=f['cm']&paired[None,:]
                if y==255:selected_frames[v]=f
                cf=f['cf'].astype(float);means=data['diffoct_mu_cf_window_mean'][b]
                line_differences.extend((means[paired]-cf[paired]).tolist())
                counts=mask.sum(axis=0);counts_safe=np.maximum(counts,1)
                masked_mu=np.where(mask,f['mu'],np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    median=np.nanmedian(masked_mu,axis=0)
                    geometric=np.exp(np.nanmean(np.log(masked_mu),axis=0))
                app=-ls_slope(f['pred'],mask,dz)/K
                logterm=-ls_slope(10*np.log10(f['mu']),mask,dz)/K
                # Exact discrete attenuation contribution is weighted by LS, not generally the arithmetic mean.
                pred_without_log=f['pred']-10*np.log10(f['mu'])
                transport_equivalent=-ls_slope(pred_without_log,mask,dz)/K
                relative=np.arange(500)[:,None]-f['zs'][None,:]
                measured=-ls_slope(f['t'],mask&(relative>=7),dz)/K
                inside_bias=np.where(mask,f['t']-f['pred'],0).sum(axis=0)/counts_safe
                metric=dict(acquisition=volume['acquisition'],concentration_percent=volume['concentration_percent'],python_volume=v,python_bscan=y,
                    paired_alines=int(paired.sum()),cf_mu=float(cf[paired].mean()),mu_mean=float(means[paired].mean()),
                    mu_median=float(median[paired].mean()),mu_geometric=float(geometric[paired].mean()),
                    apparent_mu_prediction=float(app[paired].mean()),local_log_mu_decay=float(logterm[paired].mean()),
                    attenuation_term_mu=float(transport_equivalent[paired].mean()),measured_mu_skip7=float(measured[paired].mean()),
                    diffoct_window_residual_bias_db=float(inside_bias[paired].mean()),
                    mean_minus_median_mm_inv=float((means[paired]-median[paired]).mean()),
                    approximate_closure_error_mm_inv=float((app[paired]-means[paired]-logterm[paired]).mean()),
                    exact_closure_max_error_mm_inv=float(np.max(np.abs(app[paired]-transport_equivalent[paired]-logterm[paired]))))
                assert metric['exact_closure_max_error_mm_inv']<1e-4
                metric['prediction_minus_measured_decay_mm_inv']=metric['apparent_mu_prediction']-metric['measured_mu_skip7']
                metric['cf_minus_measured_decay_mm_inv']=metric['cf_mu']-metric['measured_mu_skip7']
                frame_metrics.append(metric);volume_frames.append(metric)
                raw,inbounds=window_array(f['meas'],f['zs'],100)
                gap=raw[10:15].mean(axis=0)-raw[:5].mean(axis=0)
                eligible=f['fit'].any(axis=0)&np.isfinite(gap)
                transitions[y]=np.median(gap[eligible]);frame_flags[y]=transitions[y]>RULES['surface_transition_threshold_db']
                target,_=window_array(f['t'],f['zs'],100);valid_window,_=window_array(f['cm'],f['zs'],100)
                mu_window,_=window_array(f['mu'],f['zs'],100);valid_window &= paired[None,:]&inbounds
                peak=np.max(np.where(valid_window[:20],target[:20],-np.inf),axis=0)
                top=f['cfl'][f['zs'],np.arange(500)];margin=peak-top
                reflections[y]=paired&np.isfinite(margin)&(margin>RULES['reflection_threshold_db'])
                low_abs=valid_window&(mu_window<.1);low_rel=valid_window&(mu_window<.25*cf[None,:])
                at_bound=valid_window&(mu_window<=1e-6*RULES['lower_bound_tolerance_factor'])
                for lo,hi in RULES['window_bins']:
                    n=int(valid_window[lo:hi].sum())
                    low_rows.append(dict(acquisition=volume['acquisition'],concentration_percent=volume['concentration_percent'],python_volume=v,python_bscan=y,
                        start_pixel=lo,stop_pixel_exclusive=hi,n_valid_pixels=n,
                        lower_bound_count=int(at_bound[lo:hi].sum()),below_0p1_count=int(low_abs[lo:hi].sum()),below_quarter_cf_count=int(low_rel[lo:hi].sum()),
                        lower_bound_fraction=float(at_bound[lo:hi].sum()/n) if n else None,
                        below_0p1_fraction=float(low_abs[lo:hi].sum()/n) if n else None,
                        below_quarter_cf_fraction=float(low_rel[lo:hi].sum()/n) if n else None))
                longest_zero=longest_run(at_bound);longest_low=longest_run(low_abs)
                frac=float(low_rel.sum()/max(1,valid_window.sum()))
                row=dict(acquisition=volume['acquisition'],concentration_percent=volume['concentration_percent'],python_volume=v,python_bscan=y,matlab_bscan=y+1,
                    surface_transition_db=float(transitions[y]),window_above_surface_flag=bool(frame_flags[y]),
                    reflection_alines=int(reflections[y].sum()),paired_alines=int(paired.sum()),low_relative_fraction=frac,
                    low_relative_pixels=int(low_rel.sum()),low_relative_in_first5=int(low_rel[:5].sum()),
                    valid_window_pixels=int(valid_window.sum()),excluded_window_pixels=int(f['excl'].sum()),
                    longest_lower_bound_run=int(longest_zero.max()),longest_below_0p1_run=int(longest_low.max()))
                flag_rows.append(row)
                candidate=dict(score=frac,frame=f,worst_x=int(np.argmax(low_rel.sum(axis=0))),flag=row)
                if worst_any is None or frac>worst_any['score']:worst_any=candidate
                if (frame_flags[y] or reflections[y].any()) and (worst_flagged is None or frac>worst_flagged['score']):worst_flagged=candidate
                for x in np.flatnonzero(reflections[y]):
                    reflection_rows.append(dict(acquisition=volume['acquisition'],concentration_percent=volume['concentration_percent'],python_volume=v,python_bscan=y,x_python=int(x),x_matlab=int(x+1),peak_margin_db=float(margin[x]),cf_mu_mm_inv=float(cf[x]),diffoct_mu_mm_inv=float(means[x])))
                for policy in ['all','exclude_flagged_frames','exclude_reflection_alines','exclude_both']:
                    include=paired.copy()
                    if policy in ['exclude_flagged_frames','exclude_both'] and frame_flags[y]:include[:]=False
                    if policy in ['exclude_reflection_alines','exclude_both']:include &= ~reflections[y]
                    n=int(include.sum())
                    sensitivity_frames.append(dict(policy=policy,acquisition=volume['acquisition'],concentration_percent=volume['concentration_percent'],python_volume=v,python_bscan=y,paired_alines=n,
                        cf_mean_mm_inv=float(cf[include].mean()) if n else None,diffoct_mean_mm_inv=float(means[include].mean()) if n else None))
                src=f['zs'][:,None]+np.arange(500)[None,:];inside=src<500;src=np.minimum(src,499);xs=np.arange(500)[:,None]
                keep=inside & f['lm'][src,xs] & paired[:,None]
                residual_cf[y]=np.where(keep,f['t'][src,xs]-f['cfl'][src,xs],np.nan)
                residual_diff[y]=np.where(keep,f['t'][src,xs]-f['pred'][src,xs],np.nan)
            del data
        medians[v]=float(np.median(line_differences));del line_differences
        profile={}
        for name,res in [('cf',residual_cf),('diffoct',residual_diff)]:
            flat=res.reshape(-1,500)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                q=np.nanquantile(flat,[.25,.5,.75],axis=0)
            n=np.isfinite(flat).sum(axis=0);profile[name]=dict(q1=q[0],median=q[1],q3=q[2],count=n)
            for region,sl in [('window',slice(0,100)),('below_window',slice(100,None))]:
                values=flat[:,sl];finite=np.isfinite(values);number=int(finite.sum())
                sums=float(np.nansum(values,dtype=np.float64));squares=float(np.nansum(values.astype(np.float64)**2))
                residual_rows.append(dict(acquisition=volume['acquisition'],concentration_percent=volume['concentration_percent'],method=name,region=region,n_pixels=number,
                        residual_bias_db=sums/number,residual_rmse_db=float(np.sqrt(squares/number))))
        residual_profiles[v]=profile;del residual_cf,residual_diff
        id_keys=['acquisition','concentration_percent','python_volume','python_bscan','paired_alines']
        metrics={k:float(np.mean([m[k] for m in volume_frames])) for k in volume_frames[0] if k not in id_keys}
        volume_rows.append(dict(acquisition=volume['acquisition'],concentration_percent=volume['concentration_percent'],python_volume=v,n_frames=512,**metrics))
        flag_path=out/f"flags_{volume['tag']}.npz"
        np.savez_compressed(flag_path,python_bscan=np.arange(512),window_above_surface=frame_flags,surface_transition_db=transitions,surface_reflection_alines=reflections)
        flag_files.append(dict(**volume,flag_npz=str(flag_path),flagged_frames=np.flatnonzero(frame_flags).tolist(),reflection_alines=int(reflections.sum())))
        np.savez_compressed(out/f"residual_profile_{volume['tag']}.npz",relative_depth_mm=np.arange(500)*dz,
              **{name+'_'+k:val for name,d in profile.items() for k,val in d.items()})
        write_json(out/'progress.json',dict(completed_volumes=len(volume_rows),total_volumes=10))
        print('DIAGNOSTICS_VOLUME_COMPLETE '+volume['tag'],flush=True)
    write_csv(out/'attenuation_breakdown_frames.csv',frame_metrics);write_csv(out/'attenuation_breakdown_volumes.csv',volume_rows)
    write_csv(out/'low_mu_per_frame_by_position.csv',low_rows);write_csv(out/'artifact_flags_per_frame.csv',flag_rows)
    write_csv(out/'residual_bias_rmse.csv',residual_rows);write_csv(out/'flag_sensitivity_frames.csv',sensitivity_frames)
    if reflection_rows:write_csv(out/'reflection_flagged_alines.csv',reflection_rows)
    else:(out/'reflection_flagged_alines.csv').write_text('acquisition,concentration_percent,python_volume,python_bscan,x_python,x_matlab,peak_margin_db,cf_mu_mm_inv,diffoct_mu_mm_inv\n')
    write_json(out/'flagged_frames_by_volume.json',flag_files)
    # Fractions pooled by pixel count, separate from equal-frame attenuation statistics.
    volume_low=[]
    for v in manifest['volumes']:
        for lo,hi in RULES['window_bins']:
            rr=[r for r in low_rows if r['python_volume']==v['python_volume'] and r['start_pixel']==lo]
            n=sum(r['n_valid_pixels'] for r in rr)
            counts={k:sum(r[k] for r in rr) for k in ['lower_bound_count','below_0p1_count','below_quarter_cf_count']}
            volume_low.append(dict(acquisition=v['acquisition'],concentration_percent=v['concentration_percent'],start_pixel=lo,stop_pixel_exclusive=hi,n_valid_pixels=n,**counts,
                **{k.replace('_count','_fraction'):val/n if n else None for k,val in counts.items()}))
    write_csv(out/'low_mu_per_volume_by_position.csv',volume_low)
    sensitivity_volumes=[];sensitivity_regs=[]
    for policy in ['all','exclude_flagged_frames','exclude_reflection_alines','exclude_both']:
        for v in manifest['volumes']:
            rr=[r for r in sensitivity_frames if r['policy']==policy and r['python_volume']==v['python_volume'] and r['paired_alines']>0]
            if not rr:raise ValueError('Artifact rule removes all frames in a volume')
            cf=float(np.mean([r['cf_mean_mm_inv'] for r in rr]));diff=float(np.mean([r['diffoct_mean_mm_inv'] for r in rr]))
            sensitivity_volumes.append(dict(policy=policy,acquisition=v['acquisition'],concentration_percent=v['concentration_percent'],n_frames=len(rr),cf_mean_mm_inv=cf,diffoct_mean_mm_inv=diff,bias_mm_inv=diff-cf))
        for acquisition in ['H','L','H+L']:
            rr=[r for r in sensitivity_volumes if r['policy']==policy and (acquisition=='H+L' or r['acquisition']==acquisition)]
            for method in ['cf','diffoct']:
                sensitivity_regs.append(dict(policy=policy,acquisition=acquisition,method=method,**regression([r['concentration_percent'] for r in rr],[r[method+'_mean_mm_inv'] for r in rr])))
    write_csv(out/'flag_sensitivity_volumes.csv',sensitivity_volumes);write_csv(out/'flag_sensitivity_regressions.csv',sensitivity_regs)
    primary=read(series/'summary.json')['volumes']
    for r in sensitivity_volumes:
        if r['policy']=='all':
            p=next(p for p in primary if p['acquisition']==r['acquisition'] and p['concentration_percent']==r['concentration_percent'])
            np.testing.assert_allclose([r['cf_mean_mm_inv'],r['diffoct_mean_mm_inv']],[p['cf_mean_mm_inv'],p['diffoct_mean_mm_inv']],atol=1e-10,rtol=0)
    plt.rcParams.update({'font.family':'serif','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    examples=[]
    for acquisition in ['H','L']:
        fig,axes=plt.subplots(2,4,figsize=(17,7.5),sharex=True,gridspec_kw={'height_ratios':[2,1]},layout='constrained')
        for j,c in enumerate(RULES['example_concentrations']):
            v=next(v['python_volume'] for v in manifest['volumes'] if v['acquisition']==acquisition and v['concentration_percent']==c)
            f=selected_frames[v];xs=np.flatnonzero(f['paired']);means=np.array([f['mu'][f['cm'][:,x],x].mean(dtype=np.float64) for x in xs])
            x=int(xs[np.argmin(np.abs(means-f['cf'][xs]-medians[v]))])
            metrics=aline_panel(axes[0,j],axes[1,j],f,x,window_only=True)
            examples.append(dict(acquisition=acquisition,concentration_percent=c,python_bscan=255,x_python=x,volume_median_difference_mm_inv=medians[v],**metrics))
        axes[0,0].set_ylabel('Corrected signal (dB)');axes[1,0].set_ylabel('μ (mm⁻¹)');axes[0,0].legend(fontsize=7,loc='lower left')
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),ax=axes[0,:],extend='neither',shrink=.8);cb.set_label('DiffOCT μ − CF μ (mm⁻¹)')
        fig.suptitle(f'{acquisition}: preselected frame 256 · A-line closest to volume median difference\nFull-depth fit, fixed CF gain per A-line; displayed depth restricted to the CF comparison window')
        savefig(fig,out,f'aline_fits_{acquisition}')
        fig,axes=plt.subplots(1,5,figsize=(18,4),sharey=True,layout='constrained')
        for ax,c in zip(axes,manifest['concentrations']):
            v=next(v['python_volume'] for v in manifest['volumes'] if v['acquisition']==acquisition and v['concentration_percent']==c)
            x=np.arange(500)*selected_frames[v]['dz']
            for name,color in [('cf','black'),('diffoct','#3487da')]:
                q=residual_profiles[v][name];ax.plot(x,q['median'],color=color,label=name);ax.fill_between(x,q['q1'],q['q3'],color=color,alpha=.15,lw=0)
            ax.axhline(0,color='gray',lw=.7);ax.axvspan(0,99*selected_frames[v]['dz'],color='gray',alpha=.1)
            ax.set(title=f'{c:g}% {acquisition}',xlabel='Depth below window start (mm)',xlim=(0,3));ax.grid(alpha=.2)
        axes[0].set_ylabel('Measured − model (dB)\nmedian and IQR');axes[0].legend(fontsize=9)
        fig.suptitle('All paired A-lines across 512 frames per volume; below-window CF line is extrapolated')
        savefig(fig,out,f'residual_profiles_{acquisition}')
    write_csv(out/'example_alines.csv',examples)
    # Typical line is fixed by the same median-difference rule; worst-case selection uses recorded flags.
    typical=next(e for e in examples if e['acquisition']=='H' and e['concentration_percent']==1.)
    typical_v=next(v['python_volume'] for v in manifest['volumes'] if v['acquisition']=='H' and v['concentration_percent']==1.)
    worst=worst_flagged if worst_flagged is not None else worst_any;f=worst['frame']
    fig=plt.figure(figsize=(16,7.5),layout='constrained');grid=fig.add_gridspec(2,3)
    aline_panel(fig.add_subplot(grid[0,0]),fig.add_subplot(grid[1,0]),selected_frames[typical_v],typical['x_python'],extent=1.15,title='Typical: 1% H, frame 256')
    ax=fig.add_subplot(grid[:,1]);top=max(0,int(f['zs'].min())-15);bottom=min(500,top+180)
    ax.imshow(f['meas'][top:bottom],cmap='gray',vmin=75,vmax=120,extent=(.5,500.5,bottom+.5,top+.5),aspect='auto')
    ax.plot(np.arange(1,501),f['zs']+1,c='cyan',lw=.8);ax.plot(np.arange(1,501),f['zs']+100,c='cyan',lw=.8)
    low=f['cm']&f['paired'][None,:]&(f['mu']<.25*f['cf'][None,:]);yy,xx=np.where(low[top:bottom])
    ax.scatter(xx+1,yy+top+1,s=2,color='#67a9cf',label='μ < 0.25 CF μ')
    yy,xx=np.where(f['excl'][top:bottom]);ax.scatter(xx+1,yy+top+1,s=2,color='#ef8a62',label='Excluded measurement')
    flagged=[r['x_python'] for r in reflection_rows if r['python_volume']==f['volume'] and r['python_bscan']==f['frame']]
    if flagged:ax.scatter(np.array(flagged)+1,f['zs'][flagged]+1,marker='v',s=25,c='red',label='Reflection flag')
    ax.set(title=f"Worst flagged frame: {f['concentration']:g}% {f['acquisition']}, frame {f['frame']+1}\nSurface transition flag: {worst['flag']['window_above_surface_flag']}",xlabel='A-line (MATLAB)',ylabel='Depth pixel (MATLAB)');ax.legend(fontsize=8,loc='lower right')
    aline_panel(fig.add_subplot(grid[0,2]),fig.add_subplot(grid[1,2]),f,worst['worst_x'],extent=1.15,title=f"Worst A-line {worst['worst_x']+1} in flagged frame")
    fig.suptitle('Pixel-level μ and acquisition artifact flags; both methods use the same measured signal')
    savefig(fig,out,'pixel_mu_artifact_diagnostic')
    write_json(out/'pixel_example_selection.json',dict(typical=typical,worst_frame=worst['flag'],worst_aline_python=worst['worst_x'],any_frame_flagged=worst_flagged is not None))
    text=['# Post-production diagnostics','',
      'All calculations read the verified production NPZ files; no fitting or data exclusion was added to the primary result. Gain is fixed from CF per A-line; the original manuscript method re-profiles gain.','',
      '## Fitted A-lines','',
      'Frame MATLAB 256 is fixed in advance. At each volume, the selected paired A-line is closest to the median DiffOCT-window-mean minus CF μ over all paired A-lines and all 512 frames. Main H and supplementary L figures show 0.5, 1, 2 and 3%, restricted to the original 100-pixel CF comparison window. Both the signal and μ panels span the first through last window sample centers. Colors show μ−μ_CF on shared ±1 mm⁻¹ limits with flat color bars. The measured corrected signal is green dashed and the CF fit is black; no extrapolated signal is displayed. `example_alines.csv` records selected indices, μ and residual RMSE.','',
      'The intended comparison is whether both methods reproduce the measured signal in the fit window while DiffOCT follows it in more spatial detail. Residual RMSE is not evidence of more accurate μ: the parameter counts differ. Below-window divergence of the extrapolated CF line measures a change in decay and does not establish multiple scattering.','',
      '## Attenuation breakdown (mm⁻¹)','',
      '| Acquisition | Concentration | CF μ | DiffOCT mean | Median | Geometric mean | Apparent predicted decay | Local log μ decay | Measured decay, skip 7 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in volume_rows:text.append(f"| {r['acquisition']} | {r['concentration_percent']:g}% | {r['cf_mu']:.4f} | {r['mu_mean']:.4f} | {r['mu_median']:.4f} | {r['mu_geometric']:.4f} | {r['apparent_mu_prediction']:.4f} | {r['local_log_mu_decay']:.4f} | {r['measured_mu_skip7']:.4f} |")
    text+=['','Each metric is averaged over paired A-lines within a frame, then equally over 512 frames. Prediction decay equals the LS-weighted attenuation contribution plus the local log-μ contribution; replacing the former with the arithmetic mean μ is approximate. Both closure errors are saved. `prediction_minus_measured_decay_mm_inv` describes the change from measured window decay (first seven pixels excluded) to predicted window decay. `mean_minus_median_mm_inv` measures the effect of pixel variation. Use these production values when explaining the absolute offset.','',
      '## Artifact rules fixed before production','',
      'A frame is flagged when the median across nonempty A-lines of mean measured dB at window positions [10,15) minus [0,5) exceeds 10 dB. A paired A-line is flagged when its maximum valid corrected signal at [0,20) exceeds the analytic CF line at the window top by more than 10 dB. These flags describe acquisition artifacts that can affect both methods. They do not establish the physical identity of a bright reflector.','',
      'Pixel fractions use valid comparison pixels in paired A-lines. Bins are [0,5), [5,10), [10,90), [90,100). Missing measurements are excluded from these fractions and counted separately. Lower-bound and below-0.1-mm⁻¹ contiguous runs are recorded. Flag lists, counts and flagged A-line CF/DiffOCT μ are saved.','',
      'The worst displayed frame is the flagged frame with the highest low-relative-μ fraction; its displayed A-line has the most such pixels. If no frame is flagged, the maximum-fraction frame is shown and this fallback is recorded.','',
      '## Sensitivity to artifact exclusions','',
      'Primary production results retain all valid data. `flag_sensitivity_volumes.csv` and `flag_sensitivity_regressions.csv` show all data, exclusion of flagged frames, exclusion of reflection-flagged A-lines, and both exclusions. No exclusion is silently adopted.','',
      'Files include per-frame and per-volume low-μ fractions, residual profiles (NPZ), residual bias/RMSE inside and below the window, attenuation breakdowns, exact flag masks (NPZ), representative A-line figures, and the pixel-level diagnostic.','']
    (out/'README.md').write_text('\n'.join(text))
    write_json(out/'status.json',dict(status='complete',volumes=10,frames=5120,refitting=False,primary_result_unchanged=True))
    from datetime import datetime,timezone
    state=read(series/'status.json');state.update(post_diagnostics='complete',post_diagnostics_completed_utc=datetime.now(timezone.utc).isoformat());write_json(series/'status.json',state)
    experiment=next(p for p in series.parents if p.name == "REBUTTAL_liquid_intralipid_cf_comparison");results=experiment/'RESULTS.md'
    marker=f'<!-- post-diagnostics:{series.name} -->';old=results.read_text()
    if marker not in old:
        section=f"{marker}\n# Post-production diagnostics — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\nSaved-array diagnostics are complete: representative H/L A-lines, all-line residual profiles, attenuation breakdown, low-μ fractions, artifact flags and sensitivity to those flags. Primary estimates retain all valid data.\n\n[Diagnostic report and CSV files](runs/{series.name}/post_diagnostics/README.md) · [H A-line figure](runs/{series.name}/post_diagnostics/aline_fits_H.pdf) · [L supplement](runs/{series.name}/post_diagnostics/aline_fits_L.pdf)\n\nOffsets and attenuation contributions are in mm⁻¹. These results share CF-derived gain; they are not independent absolute-μ validation.\n\n---\n\n"
        results.write_text(section+old)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('series',type=Path);run(p.parse_args().series)
