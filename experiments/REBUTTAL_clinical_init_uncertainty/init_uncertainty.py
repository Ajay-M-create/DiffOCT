#!/usr/bin/env python3
"""S8: seed and initialization sensitivity for the five Figure 5 frames.

Run 20 seeds from U(0.01,10) and the ten Figure 4(d) conditions, five seeds
per condition. The clinical informed condition starts from the Figure 5 map
resampled onto the native polar grid. CoVs describe masked frame means;
pixelwise SD maps are saved separately. No accuracy claim is made.

Each repetition contains all five frames. When R repetitions share a GPU
batch, both loss weights are multiplied by R to preserve the gradients of
one five-frame batch, including its pooled TV normalization and Adam epsilon.
The original clinical optimizer, forward model, and checkpoint rule are used.
Random maps are drawn on the CPU and then transferred to the GPU: the audit
reproduces Figure 5 with this convention, unlike the earlier GPU random draws.

Usage: init_uncertainty.py {seed,init,stage} RUN_DIR
Env: NUM_ITERS (25000), REPEATS_PER_BATCH (10).
"""
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments'))
FRAMES = [0, 42, 85, 127, 170]
CONDITIONS = {'0_3': (0, 3), '1_1': (1, 1), '3_7': (3, 7), '5_5': (5, 5),
              '0_10': (0, 10), '0_20': (0, 20), '7_15': (7, 15),
              'informed': None, '10_10': (10, 10), '15_15': (15, 15)}
VOLUME = Path('/home/gridsan/groups/erelab/Ajay/DiffOCT-IVLDatasetsAndSegsPrePost/Pre_Post/02008Pre_target_Post.nii.gz')
FIG5 = REPO / 'figures/data/figure5_clinical/scratch_02008Pre'
FIGDATA = REPO / 'figures/data/figureS8_init_uncertainty'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def run(mode, run_dir):
    print(f'S8 {mode}: importing clinical pipeline', flush=True)
    import nibabel as nib
    import torch
    from torch import nn
    from omegaconf import OmegaConf
    import generate_patient_oac as g
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / f'{mode}.npz').exists():
        raise FileExistsError(f'Preserving existing {mode}.npz; use a new run directory')
    config, opt = g.build_ivoct_config(), g.build_opt_config()
    device = torch.device('cuda')
    data = np.asarray(nib.load(str(VOLUME)).dataobj, dtype=np.float32)
    assert data.shape == (500, 500, 171), data.shape
    scale = 130 / float(data.max())
    raw = np.stack([g.cartesian_to_polar(data[:, :, f] * scale) for f in FRAMES])
    del data
    target = torch.tensor(raw, device=device)
    nf = float(config.forward_model_system.noise_floor_dB)
    target = 10 * torch.log10(torch.clamp(torch.pow(10., target / 10) - 10 ** (nf / 10), min=1e-12))
    mask = (target >= nf).float()
    informed = np.stack([g.cartesian_to_polar(np.load(FIG5 / f'frame_{f:03d}/mu_cart.npy')) for f in FRAMES])
    informed = torch.tensor(informed, device=device).clamp(float(opt.optimization.mu_min_mm_inv), float(opt.optimization.mu_max_mm_inv)) * mask
    repetitions = [('seed', s) for s in range(20)] if mode == 'seed' else [(c, s) for c in CONDITIONS for s in range(5)]
    repeat_batch = int(os.environ.get('REPEATS_PER_BATCH', '10'))
    assert repeat_batch > 0
    stored = {k: [] for k in ['mu_polar', 'mu_init_polar', 'rendered_polar', 'best_mse_db2', 'gain_db', 'best_iter']}
    seed_ids = []
    started = time.time()
    for start in range(0, len(repetitions), repeat_batch):
        reps = repetitions[start:start + repeat_batch]
        R, F = len(reps), len(FRAMES)
        print(f'{mode}: repetitions {start + 1}--{start + R} of {len(repetitions)}', flush=True)
        t, m = target.repeat(R, 1, 1), mask.repeat(R, 1, 1)
        mu0 = torch.empty_like(t)
        for j, (condition, seed_index) in enumerate(reps):
            for i, frame in enumerate(FRAMES):
                seed = 42 + frame + 1000 * seed_index
                seed_ids.append(seed)
                torch.manual_seed(seed)
                if condition == 'informed':
                    mu0[j * F + i] = informed[i]
                else:
                    lo, hi = (.01, 10) if condition == 'seed' else CONDITIONS[condition]
                    if lo == hi:
                        mu0[j * F + i].fill_(lo)
                    else:
                        mu0[j * F + i] = torch.empty(250, 500, device='cpu').uniform_(lo, hi).to(device)
        mu0 *= m
        # Preserve the objective gradient for each five-frame repetition.
        batched_opt = OmegaConf.create(OmegaConf.to_container(opt, resolve=True))
        batched_opt.optimization.loss_scale *= R
        batched_opt.optimization.mu_tv_weight *= R
        model = g.BatchedSSForward(config, R * F, device).to(device)
        best, _, mse, _, it = g.batched_optimize(model, nn.Parameter(mu0.clone()), nn.Parameter(m.clone()), t, m, batched_opt, device, freeze_alpha=True)
        with torch.no_grad():
            model.gain_dB.zero_()
            pred = model(best, torch.ones_like(best), noise_mask=m)
            gain = ((t - pred) * m).sum((1, 2)) / m.sum((1, 2))
            pred += gain[:, None, None]
        values = dict(mu_polar=best, mu_init_polar=mu0, rendered_polar=pred, best_mse_db2=mse, gain_db=gain, best_iter=it)
        for key, value in values.items():
            stored[key].append(value.detach().cpu().numpy().reshape((R, F) + tuple(value.shape[1:])))
        del model, best, pred, mu0, values, t, m
        torch.cuda.empty_cache()
    arrays = {key: np.concatenate(parts, axis=0).swapaxes(0, 1) for key, parts in stored.items()}
    arrays.update(target_polar=target.cpu().numpy(), mask_polar=mask.cpu().numpy().astype(bool),
                  frames=np.array(FRAMES), condition=np.array([r[0] for r in repetitions]),
                  seed_index=np.array([r[1] for r in repetitions]), init_seed=np.array(seed_ids).reshape(-1, len(FRAMES)).T)
    for key, a in arrays.items():
        if np.issubdtype(a.dtype, np.number): assert np.isfinite(a).all(), key
    np.savez_compressed(run_dir / f'{mode}.npz', **arrays)
    metadata = dict(mode=mode, frames=FRAMES, frame_labels=list('ABCDE'), n_seed=20, n_init_seeds=5,
        conditions=CONDITIONS, seed_rule='42 + frame + 1000 * seed_index; CPU torch uniform draws transferred to GPU, matching Figure 5 audit',
        informed='Figure 5 Cartesian attenuation map resampled to polar, clamped to [0.01,20], then masked',
        volume=str(VOLUME), volume_sha256=sha256(VOLUME), db_scale=130,
        forward_model_system=dict(config.forward_model_system), optimization=dict(opt.optimization),
        checkpoint='lowest masked intensity MSE across post-update iterations 0 through NUM_ITERS, inclusive; published pipeline convention',
        batch_normalization='R complete five-frame repetitions; multiply both loss weights by R to preserve each five-frame objective gradient',
        repeats_per_batch=repeat_batch, elapsed_seconds=time.time()-started, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, python=platform.python_version(), job_id=os.environ.get('SLURM_JOB_ID'),
        script_sha256=sha256(__file__), pipeline_sha256=sha256(REPO/'experiments/generate_patient_oac.py'),
        figure5_sha256={str(f): sha256(FIG5/f'frame_{f:03d}/mu_cart.npy') for f in FRAMES},
        created_utc=datetime.now(timezone.utc).isoformat())
    (run_dir / f'{mode}_config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Finished {mode}: {len(FRAMES)*len(repetitions)} fits; {metadata["elapsed_seconds"]:.1f}s', flush=True)


def stage(run_dir):
    import generate_patient_oac as g
    seed, init = np.load(run_dir/'seed.npz'), np.load(run_dir/'init.npz')
    assert np.array_equal(seed['frames'], FRAMES) and np.array_equal(init['frames'], FRAMES)
    assert seed['mu_polar'].shape == (5, 20, 250, 500)
    assert init['mu_polar'].shape == (5, 50, 250, 500)
    assert np.array_equal(seed['mask_polar'], init['mask_polar'])
    assert np.array_equal(seed['target_polar'], init['target_polar'])
    cov = lambda a: float(100*np.std(a, ddof=1)/np.mean(a))
    rows, details, run_rows = [], [], []
    means, seed_sds, init_sds = [], [], []
    for i, (label, frame) in enumerate(zip('ABCDE', FRAMES)):
        mask = seed['mask_polar'][i]
        s = seed['mu_polar'][i].astype(np.float64)
        v = init['mu_polar'][i].astype(np.float64).reshape(10, 5, 250, 500)
        sm = s[:, mask].mean(axis=1)
        im = v[:, :, mask].mean(axis=2).mean(axis=1)
        sd = s.std(axis=0, ddof=1)
        isd = v.mean(axis=1).std(axis=0, ddof=1)
        means.append(s.mean(axis=0)); seed_sds.append(sd); init_sds.append(isd)
        rows.append(dict(Frame=label, seed_cov_percent=cov(sm), init_cov_percent=cov(im)))
        pub = np.load(FIG5/f'frame_{frame:03d}/mu_cart.npy'); cart = g.polar_to_cartesian(s[0]); pm = pub > 0
        details.append(dict(frame=frame, label=label, seed_mean_mu=float(sm.mean()), seed_mean_sd=float(sm.std(ddof=1)),
            init_condition_means=im.tolist(), seed_pixel_sd_median=float(np.median(sd[mask])),
            seed_pixel_sd_iqr=np.percentile(sd[mask],[25,75]).tolist(),
            init_pixel_sd_median=float(np.median(isd[mask])), init_pixel_sd_iqr=np.percentile(isd[mask],[25,75]).tolist(),
            baseline_cart_mae_vs_figure5=float(np.abs(cart-pub)[pm].mean()),
            published_cart_mean=float(pub[pm].mean()), baseline_cart_mean=float(cart[pm].mean())))
        for sweep, d in [('seed',seed),('init',init)]:
            for j, mu in enumerate(d['mu_polar'][i]):
                run_rows.append(dict(frame=frame, label=label, sweep=sweep, condition=d['condition'][j],
                    seed_index=int(d['seed_index'][j]), init_seed=int(d['init_seed'][i,j]),
                    mu_tissue_mean=float(mu[mask].astype(np.float64).mean()), best_mse_db2=float(d['best_mse_db2'][i,j]),
                    best_iter=int(d['best_iter'][i,j])))
    for name, records in [('summary.csv',rows),('per_run.csv',run_rows)]:
        with (run_dir/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    validation = dict(
        criterion='CPU seed 42+frame must reproduce each Figure 5 Cartesian map with tissue MAE below 0.001 mm^-1',
        frames=[dict(frame=d['frame'], mean_abs_diff_mm_inv=d['baseline_cart_mae_vs_figure5']) for d in details],
        passed=all(d['baseline_cart_mae_vs_figure5'] < .001 for d in details),
        earlier_mismatch='The old 0.979 mm^-1 disagreement used GPU random draws. CPU draws with the same integer seeds reproduce Figure 5.',
        checkpoint='Each run retains its lowest masked intensity MSE; no conditions or runs are discarded.')
    (run_dir/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    if not validation['passed']:
        raise RuntimeError('Baseline reproduction failed; inspect validation.json before staging')
    summary = dict(frames=details, cov_definition='100 * sample SD / mean (ddof=1); masked native-polar frame means; Init CoV across 10 condition means after averaging 5 seeds each',
                   pixel_sd_definition='sample SD across 20 seed maps, or across 10 condition-mean maps (5 seeds each); calculated before Cartesian interpolation',
                   fits=350, conditions=list(CONDITIONS), seed_count=20, init_seeds_per_condition=5)
    (run_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(run_dir/'maps.npz', frames=np.array(FRAMES), mean_mu=np.array(means), seed_sd=np.array(seed_sds), init_sd=np.array(init_sds), mask=seed['mask_polar'])
    FIGDATA.mkdir(parents=True,exist_ok=True)
    for name in ['seed.npz','init.npz','seed_config.json','init_config.json','summary.csv','per_run.csv','summary.json','maps.npz','validation.json']:
        shutil.copy2(run_dir/name, FIGDATA/name)
    (FIGDATA/'source_manifest.json').write_text(json.dumps(dict(run_dir=str(run_dir),sha256={n:sha256(run_dir/n) for n in ['seed.npz','init.npz','summary.csv','maps.npz']}),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    mode, directory = sys.argv[1], Path(sys.argv[2]).resolve()
    if mode == 'stage': stage(directory)
    elif mode in ['seed','init']: run(mode,directory)
    else: raise ValueError(mode)
