"""Plot four user-selected production A-lines from saved profile arrays."""
import argparse
import csv
from pathlib import Path

import numpy as np
from production_diagnostics import aline_panel, CMAP, NORM
from production_report import read, savefig, sha256, write_csv, write_json
import matplotlib.pyplot as plt


# Acquisition, concentration (%), MATLAB frame, MATLAB A-line, in display order.
SELECTIONS = [('H', 1., 256, 23), ('H', 3., 256, 491),
              ('L', 2., 256, 398), ('L', 3., 256, 179)]


def plot(series, output):
    source = series / 'post_diagnostics' / 'example_aline_signals.npz'
    with np.load(source) as archive:
        data = {key: archive[key] for key in archive.files}
    with (series / 'post_diagnostics' / 'example_alines.csv').open() as stream:
        original_metrics = list(csv.DictReader(stream))
    dz = read(series / 'config.json')['dz_mm']
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]},
                             layout='constrained')
    indices, rows = [], []
    for j, (acquisition, concentration, frame, aline) in enumerate(SELECTIONS):
        match = ((data['acquisition'] == acquisition)
                 & (data['concentration_percent'] == concentration)
                 & (data['python_bscan'] == frame - 1)
                 & (data['x_python'] == aline - 1))
        found = np.flatnonzero(match)
        if len(found) != 1:
            raise ValueError(f'Expected one saved profile for {SELECTIONS[j]}')
        i = int(found[0]); indices.append(i)
        fields = {'mu': 'diffoct_mu_mm_inv', 'pred': 'diffoct_prediction_corrected_db',
                  'cfl': 'cf_analytic_line_corrected_db', 't': 'measured_corrected_db',
                  'cm': 'comparison_mask', 'lm': 'valid_signal_mask',
                  'fit': 'original_fit_mask'}
        f = {key: data[name][i, :, None] for key, name in fields.items()}
        f.update(cf=data['cf_mu_mm_inv'][i:i+1], zs=f['fit'].argmax(axis=0),
                 dz=dz, frame=frame - 1, acquisition=acquisition,
                 concentration=concentration)
        np.testing.assert_allclose(data['depth_below_window_start_mm'][i],
                                   (np.arange(500)-f['zs'][0])*dz, rtol=0, atol=1e-12)
        metrics = aline_panel(axes[0, j], axes[1, j], f, 0, window_only=True,
                              title=f'{concentration:g}% {acquisition} · frame {frame}, A-line {aline}')
        prior = next(row for row in original_metrics
                     if row['acquisition'] == acquisition
                     and float(row['concentration_percent']) == concentration
                     and int(row['python_bscan']) == frame - 1
                     and int(row['x_python']) == aline - 1)
        for key, value in metrics.items():
            np.testing.assert_allclose(value, float(prior[key]), rtol=0, atol=1e-10)
        rows.append(dict(panel=j + 1, acquisition=acquisition,
                         concentration_percent=concentration, matlab_bscan=frame,
                         matlab_aline=aline, python_bscan=frame - 1,
                         x_python=aline - 1, selection='user_handpicked', **metrics))
    axes[0, 0].set_ylabel('Corrected signal (dB)\n' + r'$10\log_{10}[(I-N)/(S H)]$')
    axes[1, 0].set_ylabel('μ (mm⁻¹)')
    axes[0, 0].legend(fontsize=7, loc='lower left')
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
                      ax=axes[0, :], extend='neither', shrink=.8)
    cb.set_label('DiffOCT μ − CF μ (mm⁻¹)')
    fig.suptitle('User-selected A-lines · frame 256 (MATLAB)\n'
                 'CF comparison window · green dashed: measured corrected signal; '
                 'colored: DiffOCT; black: CF')
    savefig(fig, output, 'aline_fits_selected')
    np.savez_compressed(output / 'selected_aline_signals.npz',
                        **{key: value[indices] for key, value in data.items()})
    write_csv(output / 'selected_alines.csv', rows)
    write_json(output / 'selected_alines.json', dict(
        selection_basis='Explicitly handpicked by user after reviewing the diagnostic figures',
        panels=rows, source_npz=str(source), source_npz_sha256=sha256(source),
        displayed_depth='Original CF window only; valid corrected measurements',
        fit='Saved full-depth 15000-iteration production result',
        colorbar_limits_mm_inv=[-1., 1.], colorbar_extend='neither', refitting=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('series', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plot(args.series, args.output or args.series / 'post_diagnostics')
