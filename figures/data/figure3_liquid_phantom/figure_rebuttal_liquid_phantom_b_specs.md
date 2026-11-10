# Liquid phantom panel (b): specifications and provenance

| Parameter | Panel value | Source / definition |
|---|---|---|
| Central wavelength | 1300 nm | User-supplied SLZ1301 specification |
| Full bandwidth, Δλ | 95 nm | User-supplied manufacturer bandwidth; half-width is 47.5 nm |
| Experimental NA | ≈0.05 | User correction: beam diameter is 3.6 mm |
| Mean z_FP, H | 201 px | Mean exported focal position, rounded to the nearest pixel |
| Mean z_FP, L | 130 px | Mean exported focal position, rounded to the nearest pixel |
| Average noise floor | 81 dB | 10 log10 of mean exported linear noise power, rounded to the nearest dB |
| Axial pixel size in water | 7.35 µm | `CF_Method_Outputs.mat:zPixelSize_water_um` |

## Values used in the completed optimization

The production run `production_HL_full_depth_cf_gain_20260928T175833Z` used exported `zR = 58.68257158581` px
(0.431489 mm), per-B-scan `zFP`, per-volume depth-dependent
`nosiefloor` and `sensitivity`, and axial spacing `7.35294117647059` µm in water.
The saved H/L preparation arrays were checked against these exported noise and
confocal curves. The frozen loader is `code/plot_alignment.py:load_frames`.
It evaluates `H(z) = 1 / (1 + ((z - zFP) / zR)**2)` at `z = 1,...,500`.
Preprocessing applies `(I-N)/(S*H)` before optimization.

The exported NA, `0.02692193235832656`, is incorrect according to the user's
September 29 correction. The panel therefore reports experimental NA ≈0.05.
The completed run read zR directly; it did not calculate zR from NA or read
wavelength/bandwidth as optimization inputs. Correcting the panel's experimental
NA does not establish that the exported zR was recalibrated for that NA.

## Averaging and coordinates

The panel summarizes the ten production volumes: 0.5%, 1%, 1.5%, 2%, and 3%,
each with H and L acquisitions. Focal means cover all 512 B-scans per volume.
Mean focal positions are H 201.387396 px and
L 130.401803 px. These are exported image coordinates, rather
than distances below the liquid surface.

Noise averaging equally weights all 500 depths and all ten volumes:
`10*log10(mean(N_linear)) = 81.262169781` dB. Averaging after the logarithm
instead gives `mean(10*log10(N_linear)) = 81.244996924` dB.
The optimization used the full curves, rather than either scalar average.

`AxialScanParms.mat:zPxSize_um` is 4.901960784313726 µm. The production run's
saved calibration identifies `CF_Method_Outputs.mat:zPixelSize_water_um`
(7.35294117647059 µm) as its axial spacing. The export's lateral spacing is
9.765625000 µm; the panel's pixel-size row describes the axial spacing only.

## Bandwidth convention and appearance

95 nm is the full bandwidth supplied by the user; 47.5 nm is its half-width.
The old Figure 3b displays 45 nm while the manuscript describes a 90 nm
bandwidth, without defining a half-width convention in the panel. This new
panel reports the full width as Δλ = 95 nm. The exact 95 nm instrument-specific
value is user supplied, not independently verified from an SLZ1301 datasheet.
Thorlabs defines sampled bandwidth as the wavelength difference between the
first and last acquired points in its
[Data Acquisition with Swept Source OCT note](https://media.thorlabs.com/globalassets/family-pages/sharedassets/a/ap/apphighlight_dataacquisitionwithsweptsourceoct.pdf).
No conversion from a sweep range to a spectral FWHM is assumed.

The gray box, black border, and heading match the style of `figures/output/figure3.jpeg`.
`figures/figure3_phantom_validation.ipynb` says panel (b) was assembled manually;
it contains no panel-(b) generator. This panel has a vector PDF and a 600 dpi JPEG.

Regenerate from the repository root:

```bash
/home/gridsan/amanicka/.conda/envs/diff-oct/bin/python figures/make_figure_rebuttal_liquid_phantom_b.py
```
