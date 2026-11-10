"""Check L mapping, pixel exclusions and CF conventions before GPU fitting."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_alignment import load_frames, DATA
from gain import fixed_cf_gain
from pilot import HERE, write_json


def audit(series, acquisition='L'):
    out = Path(series) / ('alignment_'+acquisition)
    out.mkdir(exist_ok=False)
    old = json.loads((HERE / 'runs/alignment_05_20260928T153033Z/alignment_report.json').read_text())
    old_l = {f['python_bscan']: f for f in old['frames'] if f['acquisition']==acquisition}
    fig, axes = plt.subplots(3, 5, figsize=(17, 10), constrained_layout=True, sharex=True, sharey=True)
    fig2, axes2 = plt.subplots(3, 5, figsize=(17, 9), constrained_layout=True, sharex=True, sharey=True)
    reports = []
    with h5py.File(DATA/'CF_Method_Outputs.mat') as cf, h5py.File(DATA/'CF_Method_tom_predict.mat') as pred:
        for col, c in enumerate([.5, 1., 1.5, 2., 3.]):
            frames, calibration, _ = load_frames([0,255,511], [acquisition], c)
            dz = calibration['dz_water_um']/1000
            for row, f in enumerate(frames):
                r = f['report']; v,y = r['python_volume'],r['python_bscan']
                assert v == 2*col+int(acquisition=='L') and r['noncontiguous_alines']==0
                assert r['fit_width_histogram']=={'100':500}
                if c==.5:
                    assert r['sample_hashes_sha256']==old_l[y]['sample_hashes_sha256']
                    assert r['nonpositive_corrected_pixels_in_fit_mask']==old_l[y]['nonpositive_corrected_pixels_in_fit_mask']
                mu, intercept = cf['mu'][v,y,:,0], cf['G'][v,y,:,0]
                gain,fallback = fixed_cf_gain(intercept,mu,f['fit'],dz)
                raw = pred['tom'][v,y].T.astype(float)
                corrected = (raw-f['noise'][:,None])/(f['sensitivity']*f['confocal'])[:,None]
                valid = f['fit'] & ~fallback[None,:] & np.isfinite(corrected) & (corrected>0)
                z=np.arange(500)[:,None]
                expected=(10/np.log(10))*(intercept[None,:]-2*mu[None,:]*dz*z)
                with np.errstate(invalid='ignore',divide='ignore'):
                    actual=10*np.log10(corrected)
                rmse=float(np.sqrt(np.mean((expected[valid]-actual[valid])**2)))
                r.update(cf_prediction_rmse_db=rmse, valid_prediction_pixels=int(valid.sum()),
                         reviewed_05_hash_match=c==.5, gain_fallback_count=int(fallback.sum()))
                reports.append(r)
                for ax, bottom in [(axes[row,col],500),(axes2[row,col],180)]:
                    im=ax.imshow(f['db'],origin='upper',vmin=75,vmax=115,cmap='gray',extent=(.5,500.5,500.5,.5),aspect='auto')
                    ax.plot(np.arange(1,501),f['starts']+1,c='#00dcff',lw=.65)
                    ax.plot(np.arange(1,501),f['ends']+1,c='#00dcff',lw=.65)
                    yy,xx=np.where(f['invalid'])
                    ax.scatter(xx+1,yy+1,s=.3,c='#ff4055',alpha=.8)
                    ax.set_ylim(bottom+.5,.5)
                    ax.set_title(f'{c:g}% {acquisition} · frame {y+1}\nexcluded pixels: {int(f["invalid"].sum())}',fontsize=10)
                    if col==0: ax.set_ylabel('Depth pixel (MATLAB)')
                    if row==2: ax.set_xlabel('A-line (MATLAB)')
    for f,name in [(fig,acquisition+'_alignment_full'),(fig2,acquisition+'_alignment_cf_window')]:
        f.suptitle(acquisition+' mapping review · original CF boundaries in cyan · invalid corrected pixels in red\nNo additional shifts or flips',fontsize=13)
        for ext in ['pdf','png']: f.savefig(out/f'{name}.{ext}',dpi=160)
        plt.close(f)
    write_json(out/'audit.json',dict(frames=reports,acquisition=acquisition,first_middle_last_each_volume=True,
               original_05_review_hashes_match=True, all_original_bands_contiguous_100_pixels=True,
               max_cf_prediction_rmse_db=max(r['cf_prediction_rmse_db'] for r in reports),
               geometry='TIFF(z,x); MAT[v,y,x,z].T; zFP z=1..500; no extra shift'))
    print(json.dumps({'max_prediction_rmse_db':max(r['cf_prediction_rmse_db'] for r in reports),
                      'exclusions':[(r['concentration_percent'],r['matlab_bscan'],r['nonpositive_corrected_pixels_in_fit_mask']) for r in reports]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('series',type=Path)
    audit(p.parse_args().series)
