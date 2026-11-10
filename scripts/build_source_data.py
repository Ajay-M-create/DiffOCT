"""Build the Supplementary Data workbook: source data for main Figures 1-5.

One sheet per figure, plus a README sheet. Every value is read from the cached
arrays the figure notebooks use (figures/data/), except the Figure 4(c) seed
sweep and the Figure 5 cohort metrics, which are read from the saved runs below.

    python scripts/build_source_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'figures/data'
SAVED = ROOT.parent / 'DiffOCT-Saved_Results'
SEED_SWEEP_CSV = SAVED / '9_identifiability_adjusted/summary/phase1_seed_sweep_results.csv'
CLINICAL_DIR = SAVED / '22_all_pre_render_psnr'
OUT = ROOT / 'figures/output/Supplementary_Data_1_Source_Data.xlsx'

OPTIMIZERS = ['Random Search', 'CMA-ES', 'Adam', 'L-BFGS']
INIT_ORDER = ['0_3', '1_1', '3_7', '5_5', '0_10', '0_20', '7_15', 'informed', '10_10', '15_15']
CONSISTENT_INITS = ['3_7', '5_5', '0_10', '0_20', '7_15', 'informed', '10_10']
ACQ_LABEL = {i: f"{c}% (F{r})" for i, (c, r) in enumerate(
    [(1.0, 1), (1.0, 2), (1.5, 1), (1.5, 2), (2.0, 1), (2.0, 2), (3.0, 1), (3.0, 2), (5.0, 1), (5.0, 2)])}


def init_label(k):
    return 'Informed' if k == 'informed' else '[{}, {}]'.format(*k.split('_'))


def concordance(gt, rec):
    lr = stats.linregress(gt, rec)
    return dict(r_squared=lr.rvalue ** 2, ols_slope=lr.slope, ols_intercept_mm_inv=lr.intercept,
                mape_percent=float(np.mean(np.abs(rec - gt) / (gt + 1e-12)) * 100),
                mae_mm_inv=float(np.mean(np.abs(rec - gt))), n_pixels=gt.size)


# ── Figure 1 ─────────────────────────────────────────────────────────────────
def figure1():
    data = json.loads((DATA / 'figure1_single_aline_scaling/results_single_aline.json').read_text())
    sizes = sorted(int(k) for k in data if not k.startswith('_'))
    per_seed, summary = [], []
    for n in sizes:
        for opt in OPTIMIZERS:
            r = data[str(n)][opt]
            for s, mape, ms, ev, wall in zip(r['seeds_present'], r['seed_mape'], r['seed_ms_per_fwd'],
                                             r['seed_fwd_evals'], r['seed_wall_sec']):
                per_seed.append({'Problem size K (depth parameters)': n, 'Optimizer': opt, 'Seed': s,
                                 'MAPE (%)': mape, 'Wall time per objective evaluation (ms)': ms,
                                 'Objective evaluations': ev, 'Total wall time (s)': wall})
            summary.append({'Problem size K (depth parameters)': n, 'Optimizer': opt, 'n seeds': len(r['seed_mape']),
                            'MAPE mean (%)': r['avg_mape'], 'MAPE SD (%)': r['std_mape'],
                            'Time per evaluation mean (ms)': r['avg_ms_per_fwd'],
                            'Time per evaluation SD (ms)': r['std_ms_per_fwd']})
    return [('Figure 1. Summary plotted in the figure (mean and SD across eight paired initializations; '
             'SD is the population SD, ddof = 0)', pd.DataFrame(summary)),
            ('Figure 1. Per-seed values', pd.DataFrame(per_seed))]


# ── Figure 2 ─────────────────────────────────────────────────────────────────
def figure2():
    d = DATA / 'figure2_verification'
    fwd = np.load(d / 'fig1_forward_validation.npz')
    err = fwd['abs_err'].astype(np.float64)
    panel_i = pd.DataFrame([
        {'Metric': 'R^2 (forward model vs analytical)', 'Value': float(fwd['r2'])},
        {'Metric': 'MAE (dB)', 'Value': err.mean()},
        {'Metric': 'Max absolute error (dB)', 'Value': err.max()},
        {'Metric': 'Pixels in error map', 'Value': err.size},
    ])

    gt, rec = np.load(d / 'showcase_mu_gt_mm.npy'), np.load(d / 'showcase_mu_recovered_mm.npy')
    band = slice(230, 270)                                   # tissue band used for panel (n)
    n = concordance(gt[band].ravel().astype(np.float64), rec[band].ravel().astype(np.float64))
    panel_n = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in n.items()])

    film = np.load(d / 'convergence_filmstrip_data.npz')
    pgt = film['pixel_gt_all'].astype(np.float64)
    prec = film['pixel_rec_all'].astype(np.float64)
    iters = film['snapshot_iters'].astype(int)
    snaps = []
    for panel, it in zip('pqrs', [0, 1000, 10000, 30000]):
        k = int(np.argmin(np.abs(iters - it)))
        snaps.append({'Panel': panel, 'Iteration': int(iters[k]), **concordance(pgt, prec[k])})
    curve = [{'Iteration': int(it), 'MAPE (%)': concordance(pgt, prec[k])['mape_percent'],
              'R^2': concordance(pgt, prec[k])['r_squared']} for k, it in enumerate(iters)]
    return [('Figure 2(i). Forward-model error map summary', panel_i),
            ('Figure 2(n). Concordance of recovered vs ground-truth mu, representative B-scan '
             '(tissue band, depth rows 230-269, all 500 A-lines)', panel_n),
            ('Figure 2(p)-(s). Concordance across all 500 computational phantoms (100 tissue pixels each) '
             'at four snapshots; R^2 and slope from ordinary least squares', pd.DataFrame(snaps)),
            ('Figure 2(t). Convergence curve (all 117 logged snapshots)', pd.DataFrame(curve))]


# ── Figure 3 ─────────────────────────────────────────────────────────────────
def figure3():
    d = DATA / 'figure3_liquid_phantom'
    concs = [0.5, 1.0, 1.5, 2.0, 3.0]
    with np.load(d / 'violin_frame_means.npz') as z:
        vol, method, vals, keep, bscan = z['python_volume'], z['method'], z['frame_values'], z['retained'], z['python_bscan']
    frames, summary = [], []
    for row in range(len(vol)):
        conc, acq = concs[vol[row] // 2], 'HL'[vol[row] % 2]
        name = {'diffoct': 'DiffOCT', 'cf': 'CF'}[method[row]]
        for b, v, k in zip(bscan, vals[row], keep[row]):
            frames.append({'Concentration (%)': conc, 'Focal position': acq, 'Method': name,
                           'B-scan (1-based)': int(b) + 1, 'B-scan mean mu in CF windows (mm^-1)': v,
                           'Shown in violin (Tukey 1.5 IQR filter)': bool(k)})
        summary.append({'Concentration (%)': conc, 'Focal position': acq, 'Method': name,
                        'N B-scans': vals[row].size, 'N shown in violin': int(keep[row].sum()),
                        'Volume mean, all B-scans (mm^-1), used for regression': vals[row].mean(),
                        'SD, all B-scans (mm^-1)': vals[row].std(ddof=1),
                        'Mean of shown B-scans (mm^-1), black marker': vals[row][keep[row]].mean()})
    summary = pd.DataFrame(summary).sort_values(['Method', 'Focal position', 'Concentration (%)'],
                                                ascending=[False, True, True])
    reg = pd.read_csv(d / 'violin_regressions.csv')
    reg['method'] = reg['method'].map({'diffoct': 'DiffOCT', 'cf': 'CF'})
    reg['plotted in Figure 3(f)'] = [(a, m) in {('H', 'DiffOCT'), ('L', 'DiffOCT'), ('H+L', 'CF')}
                                     for a, m in zip(reg['acquisition'], reg['method'])]
    reg = reg.rename(columns={'acquisition': 'Focal position', 'method': 'Method', 'analysis': 'Analysis',
                              'slope_mm_inv_per_percentage_point': 'Slope (mm^-1 per % IL)',
                              'intercept_mm_inv': 'Intercept (mm^-1)', 'r_squared': 'R^2',
                              'n_volume_means': 'n volume means'})
    # The regressions must reproduce from the volume means listed above.
    for _, r in reg.iterrows():
        sel = summary[(summary['Method'] == r['Method']) & summary['Focal position'].isin(list(r['Focal position'].replace('+', '')))]
        lr = stats.linregress(sel['Concentration (%)'], sel['Volume mean, all B-scans (mm^-1), used for regression'])
        assert np.isclose(lr.slope, r['Slope (mm^-1 per % IL)']) and np.isclose(lr.rvalue ** 2, r['R^2'])
    shown = [0, 3]                                           # rows plotted in panel (g): 1% H and 3% L
    alines = pd.read_csv(d / 'aline_metrics_selected.csv').iloc[shown][
        ['acquisition', 'concentration_percent', 'matlab_bscan', 'matlab_aline',
         'cf_mu_mm_inv', 'diffoct_window_mean_mm_inv', 'cf_rmse_db', 'diffoct_rmse_db']]
    alines.columns = ['Focal position', 'Concentration (%)', 'B-scan (1-based)', 'A-line (1-based)',
                      'CF mu (mm^-1)', 'DiffOCT mean mu in CF window (mm^-1)',
                      'CF intensity RMSE (dB)', 'DiffOCT intensity RMSE (dB)']
    with np.load(d / 'aline_signals_selected.npz') as z:
        D = {k: z[k] for k in z.files}
    profiles = []
    for i, (_, a) in zip(shown, alines.iterrows()):
        win = np.flatnonzero(D['original_fit_mask'][i].astype(bool))   # CF fitting window, as plotted
        valid = D['valid_signal_mask'][i][win].astype(bool) & np.isfinite(D['measured_corrected_db'][i][win])
        shown_or_nan = lambda x: np.where(valid, x[i][win], np.nan)
        mu = shown_or_nan(D['diffoct_mu_mm_inv'])
        cf = float(D['cf_mu_mm_inv'][i])
        profiles.append(pd.DataFrame({
            'A-line': f"{a['Concentration (%)']:g}% {a['Focal position']}, B-scan {a['B-scan (1-based)']}, "
                      f"A-line {a['A-line (1-based)']}",
            'Depth below CF window start (mm)': D['depth_below_window_start_mm'][i][win],
            'Measured, corrected (dB)': shown_or_nan(D['measured_corrected_db']),
            'CF fit (dB)': shown_or_nan(D['cf_analytic_line_corrected_db']),
            'DiffOCT prediction (dB)': shown_or_nan(D['diffoct_prediction_corrected_db']),
            'DiffOCT mu (mm^-1)': mu, 'CF mu (mm^-1)': cf, 'mu_DiffOCT - mu_CF (mm^-1)': mu - cf}))
    profiles = pd.concat(profiles, ignore_index=True)
    return [('Figure 3(f). Regression parameters (fits to the five unfiltered volume means per acquisition)', reg),
            ('Figure 3(f). Per-violin summary', summary),
            ('Figure 3(g). Example A-lines: CF attenuation and intensity RMSE within the CF window', alines),
            ('Figure 3(g). A-line depth profiles within the CF fitting window (blank = excluded signal)', profiles),
            ('Figure 3(f). Per-B-scan values behind each violin (N = 512 per violin)', pd.DataFrame(frames))]


# ── Figure 4 ─────────────────────────────────────────────────────────────────
def figure4():
    d = DATA / 'figure4_identifiability_robustness'
    seed = pd.read_csv(SEED_SWEEP_CSV)
    init = pd.read_csv(d / 'phase2_init_range_results.csv')
    cond = init.groupby(['phantom_index', 'init_condition'])['mu_tissue_mean'].mean().reset_index()
    sel = cond[cond['init_condition'].isin(CONSISTENT_INITS)].groupby('phantom_index')['mu_tissue_mean']
    g = seed.groupby('phantom_index')['mu_tissue_mean']
    table_c = pd.DataFrame({
        'Acquisition': [f'A{i}' for i in range(10)],
        'Concentration and focal position': [ACQ_LABEL[i] for i in range(10)],
        'Seed CoV, 20 seeds (%)': (100 * g.std(ddof=1) / g.mean()).values,
        'Init CoV, 7 selected conditions (%)': (100 * sel.std(ddof=1) / sel.mean()).values,
    })
    ref = sel.mean()
    cond['Acquisition'] = cond['phantom_index'].map(lambda i: f'A{i}')
    cond['Concentration and focal position'] = cond['phantom_index'].map(ACQ_LABEL)
    cond['Initialization range (mm^-1)'] = cond['init_condition'].map(init_label)
    cond['Selected (consistent) condition'] = cond['init_condition'].isin(CONSISTENT_INITS)
    cond['Deviation from mean of selected conditions (%)'] = (
        100 * (cond['mu_tissue_mean'] - cond['phantom_index'].map(ref)) / cond['phantom_index'].map(ref))
    cond['_o'] = cond['init_condition'].map(INIT_ORDER.index)
    panel_d = cond.sort_values(['phantom_index', '_o']).rename(
        columns={'mu_tissue_mean': 'Recovered tissue mu, mean of 5 seeds (mm^-1)'})[
        ['Acquisition', 'Concentration and focal position', 'Initialization range (mm^-1)',
         'Selected (consistent) condition', 'Recovered tissue mu, mean of 5 seeds (mm^-1)',
         'Deviation from mean of selected conditions (%)']]
    init['_o'] = init['init_condition'].map(INIT_ORDER.index)
    panel_d_seeds = init.sort_values(['phantom_index', '_o', 'seed']).assign(
        Acquisition=lambda x: x['phantom_index'].map(lambda i: f'A{i}'),
        cfp=lambda x: x['phantom_index'].map(ACQ_LABEL),
        rng=lambda x: x['init_condition'].map(init_label))[
        ['Acquisition', 'cfp', 'rng', 'seed', 'mu_tissue_mean', 'mse_final']]
    panel_d_seeds.columns = ['Acquisition', 'Concentration and focal position', 'Initialization range (mm^-1)',
                             'Seed', 'Recovered tissue mu (mm^-1)', 'Final MSE (dB^2)']
    seeds = seed.assign(Acquisition=lambda x: x['phantom_index'].map(lambda i: f'A{i}'),
                        cfp=lambda x: x['phantom_index'].map(ACQ_LABEL))[
        ['Acquisition', 'cfp', 'seed', 'mu_tissue_mean', 'mse_final']]
    seeds.columns = ['Acquisition', 'Concentration and focal position', 'Seed',
                     'Recovered tissue mu (mm^-1)', 'Final MSE (dB^2)']

    curves = pd.read_csv(d / 'per_acquisition_noise_curves.csv')
    tb = pd.read_csv(d / 'tableB_physical_by_snr.csv')
    rows = [{'SNR (dB)': 'Original (baseline)', '|Delta mu| vs baseline, mean (%)': 0.0,
             '|Delta mu| vs baseline, SD (%)': None, 'R^2 (mean over acquisitions)': 1.0,
             'Max sigma_mu (mm^-1)': None}]
    for s in [16, 12, 8, 4, 0]:
        sub, t = curves[curves['snr_dB'] == s], tb[tb['snr_dB'] == str(s)].iloc[0]
        rows.append({'SNR (dB)': s, '|Delta mu| vs baseline, mean (%)': float(t['delta_pct_mean']),
                     '|Delta mu| vs baseline, SD (%)': float(t['delta_pct_std']),
                     'R^2 (mean over acquisitions)': sub['r2'].mean(),
                     'Max sigma_mu (mm^-1)': sub['sigma_mm_inv'].max()})
    curves = curves.assign(Acquisition=lambda x: x['phantom_index'].map(lambda i: f'A{i}'))[
        ['Acquisition', 'label', 'concentration_pct', 'focal_plane', 'snr_dB', 'baseline_snr_dB',
         'delta_pct', 'r2', 'sigma_mm_inv']].sort_values(['Acquisition', 'snr_dB'])
    curves.columns = ['Acquisition', 'Label', 'Concentration (%)', 'Focal position', 'SNR (dB)',
                      'Baseline SNR of acquisition (dB)', '|Delta mu| vs baseline (%)',
                      'R^2 vs baseline', 'sigma_mu (mm^-1)']
    return [('Figure 4(c). Identifiability summary (CoV = sample SD / mean)', table_c),
            ('Figure 4(f). Noise robustness summary', pd.DataFrame(rows)),
            ('Figure 4(d). Recovered tissue mu per initialization condition (mean of 5 seeds); '
             'deviation is relative to the mean over the 7 selected conditions', panel_d),
            ('Figure 4(g). Per-acquisition curves versus SNR', curves),
            ('Figure 4(d). Per-seed values (5 seeds per condition)', panel_d_seeds),
            ('Figure 4(a,c). Seed sweep, per-seed values (20 seeds per acquisition)', seeds)]


# ── Figure 5 ─────────────────────────────────────────────────────────────────
def figure5():
    patients = sorted(p.name for p in CLINICAL_DIR.iterdir() if (p / 'psnr_summary.npz').exists())
    assert len(patients) == 128
    frames, per_patient = [], []
    for i, pid in enumerate(patients, 1):
        code = f'P{i:03d}'                                    # anonymized; trial IDs withheld
        with np.load(CLINICAL_DIR / pid / 'psnr_summary.npz') as s:
            ok = ~np.isnan(s['psnr_s1'])
            f = pd.DataFrame({'Patient': code, 'Frame index': s['frame_indices'][ok],
                              'PSNR Stage 1 (dB)': s['psnr_s1'][ok], 'PSNR Stage 2 (dB)': s['psnr_s2'][ok],
                              'Dice Stage 1': s['dice_s1'][ok], 'Dice Stage 2': s['dice_s2'][ok]})
        frames.append(f)
        per_patient.append({'Patient': code, 'Frames': len(f),
                            **{f'{c} mean': f[c].mean() for c in f.columns[2:]}})
    frames = pd.concat(frames, ignore_index=True)
    assert len(frames) == 29269
    cohort = pd.DataFrame([{'Metric': c, 'Mean': frames[c].astype(np.float64).mean(),
                            'SD': frames[c].astype(np.float64).std(ddof=0), 'Frames': len(frames),
                            'Patients': len(patients)} for c in frames.columns[2:]])
    return [('Figure 5. Cohort summary over all frames (as reported in the caption; SD is population SD). '
             'Dice uses a 15 dB noise-floor threshold.', cohort),
            ('Figure 5. Per-patient means (patients anonymized)', pd.DataFrame(per_patient)),
            ('Figure 5. Per-frame values (patients anonymized)', frames)]


README = [
    ('Title', 'Supplementary Data 1. Source data for main Figures 1-5'),
    ('Manuscript', 'Differentiable Inverse Rendering Recovers Intrinsic Properties from Intravascular Optical Coherence Tomography '
                   '(Manicka et al.)'),
    ('Layout', 'One sheet per main figure. Each sheet stacks several tables; each table starts with a bold '
               'title naming the panel(s) it supports.'),
    ('Figure 1', 'MAPE (%) and wall time per objective evaluation (ms) for each optimizer and problem size K; '
                 'mean and SD plus all per-seed values (8 seeds).'),
    ('Figure 2', 'Panel (i) error-map summary; panel (n) concordance metrics; panels (p)-(s) R^2, slope and '
                 'MAPE at the four snapshots; panel (t) convergence curve. The 50,000 individual scatter points '
                 'are not listed.'),
    ('Figure 3', 'Panel (f): per-B-scan mean mu for each concentration, focal position (H, L) and method '
                 '(DiffOCT, CF), with the violin display filter, per-violin summaries and regression parameters. '
                 'Panel (g): depth profiles of measured intensity, CF fit, DiffOCT prediction and attenuation for '
                 'both example A-lines, with their RMSE values.'),
    ('Figure 4', 'Panel (c) and (f) tables; panel (d) per-condition recovered mu and deviation; panel (g) '
                 'per-acquisition curves versus SNR; per-seed values for the seed and initialization sweeps.'),
    ('Figure 5', 'Cohort, per-patient and per-frame PSNR and Dice for Stage 1 (mu, alpha = 1) and Stage 2 '
                 '(mu + alpha). Patients are coded P001-P128. Clinical images and pixel maps are not included; '
                 'they are available under the data-access terms in the Data Availability statement.'),
    ('Units', 'mu and attenuation in mm^-1; intensities and PSNR in dB; MAPE, CoV and deviations in %.'),
]


def write_sheet(writer, name, blocks):
    row = 0
    for title, df in blocks:
        pd.DataFrame([[title]]).to_excel(writer, sheet_name=name, startrow=row, index=False, header=False)
        df.to_excel(writer, sheet_name=name, startrow=row + 1, index=False)
        ws = writer.sheets[name]
        ws.cell(row + 1, 1).font = Font(bold=True)
        for c in range(1, df.shape[1] + 1):
            ws.cell(row + 2, c).alignment = Alignment(wrap_text=True, vertical='top')
        row += len(df) + 4
    ws = writer.sheets[name]
    for col in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col)].width = 22
    ws.freeze_panes = None


def main():
    OUT.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT, engine='openpyxl') as w:
        readme = pd.DataFrame(README, columns=['Item', 'Description'])
        readme.to_excel(w, sheet_name='README', index=False)
        w.sheets['README'].column_dimensions['A'].width = 14
        w.sheets['README'].column_dimensions['B'].width = 120
        for r in w.sheets['README'].iter_rows(min_row=2):
            r[1].alignment = Alignment(wrap_text=True, vertical='top')
        for name, fn in [('Figure 1', figure1), ('Figure 2', figure2), ('Figure 3', figure3),
                         ('Figure 4', figure4), ('Figure 5', figure5)]:
            write_sheet(w, name, fn())
    print(OUT)


if __name__ == '__main__':
    main()
