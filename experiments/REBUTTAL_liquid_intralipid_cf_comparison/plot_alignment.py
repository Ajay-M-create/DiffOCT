"""Plot supplied 0.5% TIFFs and masks for review before DiffOCT optimization.

Run with the diff-oct environment. Reads inputs; writes a new directory under
this experiment's runs/. Does not shift images, regenerate masks, or fit CF.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import tifffile
from preprocessing import correct_and_mask


HERE = next(p for p in Path(__file__).resolve().parents if p.name == "REBUTTAL_liquid_intralipid_cf_comparison")
DATA = HERE.parents[1] / "inputs" / "LiquidIntralipidPhantoms_20260915"
FIT_COLOR = "#00dcff"
SAMPLE_COLOR = "#76ff55"
FOCUS_COLOR = "#ffcc33"
BAD_COLOR = "#ff4055"
PROFILE_COLORS = ["#0072b2", "#cc79a7", "#d55e00"]
PROFILE_X = [50, 250, 450]  # Python indices; displayed as MATLAB 51, 251, 451.
DISPLAY_LIMITS = (75, 115)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_frames(indices, labels=("H", "L"), concentration=0.5):
    if not labels or any(label not in ("H", "L") for label in labels):
        raise ValueError("Labels must select H and/or L")
    filenames = []
    token = f"{concentration:.1f}".replace(".", "..")
    for label in ("H", "L"):
        matches = sorted(DATA.glob(f"*IL_{token}_LSM03_{label}*DespeckledLogInt.tif"))
        if len(matches) != 1:
            raise ValueError(f"Expected one {concentration}% {label} TIFF, found {matches}")
        filenames.append(matches[0])
    frames = []
    with h5py.File(DATA / "CF_Method_Outputs.mat", "r") as params, \
            h5py.File(DATA / "CF_Method_fit_mask.mat", "r") as fit_file, \
            h5py.File(DATA / "IntMask.mat", "r") as sample_file:
        concentrations = params["concentrations"][()].reshape(-1)
        matches = np.flatnonzero(np.isclose(concentrations, concentration))
        if len(matches) != 1:
            raise ValueError(f"Concentration {concentration} absent or duplicated in CF export")
        first_volume = 2 * int(matches[0])
        zr = float(params["zR"][0, 0])
        calibration = {
            "zR_px": zr,
            "dz_water_um": float(params["zPixelSize_water_um"][0, 0]),
            "dx_um": float(params["latPixelSIze_um"][0, 0]),
        }
        for y in indices:
            # MATLAB (z,x,y,v) is exposed by h5py as (v,y,x,z).
            fit_pair = fit_file["zFitMask"][first_volume:first_volume+2, y, :, :].transpose(0, 2, 1).astype(bool)
            sample_pair = sample_file["phantomMask"][first_volume:first_volume+2, y, :, :].transpose(0, 2, 1).astype(bool)
            for local_v, label in enumerate(("H", "L")):
                if label not in labels:
                    continue
                v = first_volume + local_v
                with tifffile.TiffFile(filenames[local_v]) as tif:
                    assert len(tif.pages) == 512
                    assert "rangeMax=[150," in tif.pages[0].description
                    raw = tif.pages[y].asarray()
                assert raw.shape == (500, 500) and raw.dtype == np.uint16
                db = raw.astype(np.float64) * (150.0 / 65535.0)
                fit, sample = fit_pair[local_v], sample_pair[local_v]
                noise = params["nosiefloor"][v, 0, 0, :].astype(np.float64)
                sensitivity = params["sensitivity"][v, 0, 0, :].astype(np.float64)
                zfp = float(params["zFP"][v, y, 0, 0])
                confocal = 1.0 / (1.0 + ((np.arange(1, 501) - zfp) / zr) ** 2)
                assert np.all(np.isfinite(noise)) and np.all(noise > 0)
                assert np.all(np.isfinite(sensitivity)) and np.all(sensitivity > 0)
                corrected = correct_and_mask(db, noise, sensitivity, confocal, fit)
                invalid = corrected["excluded_mask"]
                counts = fit.sum(axis=0)
                starts = fit.argmax(axis=0)
                ends = 499 - fit[::-1].argmax(axis=0)
                nonempty = counts > 0
                raw_cf = params["mu"][v, y, :, 0]
                widths, frequencies = np.unique(counts, return_counts=True)
                report = {
                    "concentration_percent": float(concentration),
                    "acquisition": label,
                    "candidate_matlab_volume": v + 1,
                    "python_volume": v,
                    "matlab_bscan": y + 1,
                    "python_bscan": y,
                    "filename": filenames[local_v].name,
                    "zFP_px": zfp,
                    "fit_width_histogram": {str(int(w)): int(n) for w, n in zip(widths, frequencies)},
                    "fit_first_sample_matlab_min_max": [int(starts[nonempty].min() + 1), int(starts[nonempty].max() + 1)],
                    "fit_last_sample_matlab_min_max": [int(ends[nonempty].min() + 1), int(ends[nonempty].max() + 1)],
                    "noncontiguous_alines": int((nonempty & (ends - starts + 1 != counts)).sum()),
                    "nonpositive_corrected_pixels_in_fit_mask": int(invalid.sum()),
                    "alines_with_nonpositive_corrected_pixels": int(invalid.any(axis=0).sum()),
                    "affected_alines_with_finite_cf_mu": int((invalid.any(axis=0) & np.isfinite(raw_cf)).sum()),
                    "comparison_pixels_retained": int(corrected["comparison_mask"].sum()),
                    "alines_with_comparison_pixels": int(corrected["comparison_mask"].any(axis=0).sum()),
                    "sample_hashes_sha256": {"tiff_uint16_zx": digest(raw), "fit_mask_bool_zx": digest(fit), "sample_mask_bool_zx": digest(sample)},
                }
                frames.append(dict(db=db, fit=fit, sample=sample, invalid=invalid,
                                   comparison_mask=corrected["comparison_mask"],
                                   noise=noise, sensitivity=sensitivity, confocal=confocal,
                                   starts=starts, ends=ends, report=report))
                print(json.dumps({k: report[k] for k in ("acquisition", "matlab_bscan", "nonpositive_corrected_pixels_in_fit_mask", "noncontiguous_alines")}), flush=True)
    source_paths = [p for label, p in zip(("H", "L"), filenames) if label in labels] + [DATA / n for n in ("CF_Method_Outputs.mat", "CF_Method_fit_mask.mat", "IntMask.mat")]
    sources = [{"path": str(p.resolve()), "size_bytes": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns} for p in source_paths]
    return frames, calibration, sources


def legend_handles():
    return [Line2D([], [], color=FIT_COLOR, lw=2, label="CF fit mask boundary (zFitMask)"),
            Line2D([], [], color=SAMPLE_COLOR, lw=1.5, ls="--", label="Sample mask boundary (IntMask; context)"),
            Line2D([], [], color=FOCUS_COLOR, lw=1.5, ls="--", label="Exported focal depth zFP"),
            Patch(facecolor=BAD_COLOR, alpha=0.8, label="I ≤ N inside CF fit mask")]


def image_panel(ax, frame, depth_max=500, profiles=False):
    im = ax.imshow(frame["db"], cmap="gray", vmin=DISPLAY_LIMITS[0], vmax=DISPLAY_LIMITS[1],
                   origin="upper", extent=(0.5, 500.5, 500.5, 0.5), aspect="auto", interpolation="nearest")
    xz = np.arange(1, 501)
    ax.contour(xz, xz, frame["sample"], levels=[0.5], colors=[SAMPLE_COLOR], linewidths=0.9, linestyles="--")
    ax.contour(xz, xz, frame["fit"], levels=[0.5], colors=[FIT_COLOR], linewidths=1.1)
    if frame["invalid"].any():
        rgba = np.zeros((500, 500, 4))
        rgba[frame["invalid"]] = matplotlib.colors.to_rgba(BAD_COLOR, 0.85)
        ax.imshow(rgba, origin="upper", extent=(0.5, 500.5, 500.5, 0.5), aspect="auto", interpolation="nearest")
    ax.axhline(frame["report"]["zFP_px"], color=FOCUS_COLOR, lw=1.2, ls="--")
    if profiles:
        for xi, color in zip(PROFILE_X, PROFILE_COLORS):
            ax.axvline(xi + 1, color=color, lw=0.8, alpha=0.8)
    ax.set(xlim=(0.5, 500.5), ylim=(depth_max + 0.5, 0.5),
           xlabel="A-line x (MATLAB index →)", ylabel="Depth z (MATLAB index ↓)")
    ax.set_xticks([1, 100, 200, 300, 400, 500])
    ax.set_yticks([1, 100, 200, 300, 400, 500] if depth_max == 500 else [1, 10, 20, 30, 40, 50, 60])
    if depth_max == 160:
        ax.set_yticks([1, 20, 40, 60, 80, 100, 120, 140, 160])
    return im


def save_figure(fig, out, name, pdf):
    fig.savefig(out / f"{name}.png", dpi=180, facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def overview(frames, out, pdf, zoom=False):
    nrows = len(frames) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3.05 * nrows + 1.8), squeeze=False)
    for ax, frame in zip(axes.flat, frames):
        r = frame["report"]
        im = image_panel(ax, frame, 60 if zoom else 500)
        ax.set_title(f"0.5% {r['acquisition']} → candidate MAT volume {r['candidate_matlab_volume']}  |  "
                     f"B-scan {r['matlab_bscan']} (Python {r['python_bscan']})\n"
                     f"zFP = {r['zFP_px']:.2f} px; I ≤ N in fit mask: {r['nonpositive_corrected_pixels_in_fit_mask']:,} pixels",
                     fontsize=10, loc="left")
    fig.suptitle("0.5% liquid Intralipid — " + ("surface alignment, first 60 depth pixels" if zoom else "full-depth alignment"),
                 fontsize=17, y=0.985)
    fig.text(0.5, 0.953, "TIFF displayed as (z, x); MATLAB masks transposed to (z, x). H/L assignment and zFP origin await review.",
             ha="center", fontsize=10)
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.49, 0.020), ncol=2, fontsize=10, frameon=False)
    fig.text(0.5, 0.012, "All indices shown are 1-based unless labelled Python. Cyan/green contours follow pixel edges; yellow uses the stored zFP value.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.065, right=0.91, top=0.912, bottom=0.10, hspace=0.44, wspace=0.23)
    cax = fig.add_axes([0.93, 0.22, 0.015, 0.53])
    fig.colorbar(im, cax=cax, label="Measured TIFF intensity (dB)", extend="both")
    save_figure(fig, out, "surface_zoom" if zoom else "alignment_overview", pdf)


def detail(frame, out, pdf):
    r = frame["report"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    im = image_panel(axes[0, 0], frame, profiles=True)
    axes[0, 0].set_title("A. Full measured B-scan", loc="left")
    image_panel(axes[0, 1], frame, 160, profiles=True)
    axes[0, 1].set_title("B. Surface and entire CF fitting band", loc="left")
    z = np.arange(1, 501)
    ax = axes[1, 0]
    for xi, color in zip(PROFILE_X, PROFILE_COLORS):
        ax.plot(frame["db"][:, xi], z, color=color, lw=1.1, label=f"A-line {xi + 1} (Python {xi})")
        first, last = frame["starts"][xi] + 1, frame["ends"][xi] + 1
        ax.scatter(frame["db"][[first - 1, last - 1], xi], [first, last], color=FIT_COLOR,
                   edgecolors="black", linewidths=0.4, s=22, zorder=4)
    ax.plot(10 * np.log10(frame["noise"]), z, color="black", ls=":", lw=1.7, label="Exported measured-space noise N")
    ax.axhline(r["zFP_px"], color=FOCUS_COLOR, lw=1.2, ls="--")
    ax.set(xlim=(65, 120), ylim=(500.5, 0.5), xlabel="Measured intensity / noise (dB)",
           ylabel="Depth z (MATLAB index ↓)", title="C. Measured A-lines and exported noise\nCyan dots mark fit endpoints")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.2)
    ax = axes[1, 1]
    ax.plot(frame["sensitivity"], z, color="#0072b2", label="Sensitivity S")
    ax.plot(frame["confocal"], z, color="#cc79a7", label="Confocal H (assuming z = 1…500)")
    ax.plot(frame["sensitivity"] * frame["confocal"], z, color="black", label="S × H")
    ax.axhline(r["zFP_px"], color=FOCUS_COLOR, lw=1.2, ls="--")
    ax.set(xlim=(0, 1.03), ylim=(500.5, 0.5), xlabel="Intensity multiplier", ylabel="Depth z (MATLAB index ↓)",
           title="D. Exported calibration at this B-scan")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.suptitle(f"0.5% {r['acquisition']} — B-scan {r['matlab_bscan']} (Python {r['python_bscan']}); candidate MAT volume {r['candidate_matlab_volume']}",
                 fontsize=16, y=0.985)
    first, last = r["fit_first_sample_matlab_min_max"], r["fit_last_sample_matlab_min_max"]
    fig.text(0.5, 0.952, f"Fit starts: z = {first[0]}–{first[1]}; ends: z = {last[0]}–{last[1]}. "
             f"Nonpositive corrected signal: {r['nonpositive_corrected_pixels_in_fit_mask']:,} pixels in "
             f"{r['alines_with_nonpositive_corrected_pixels']} A-lines.", ha="center", fontsize=10)
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.48, 0.035), ncol=2, fontsize=9, frameon=False)
    fig.text(0.5, 0.018, r["filename"], ha="center", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.13, top=0.90, hspace=0.29, wspace=0.24)
    cax = fig.add_axes([0.948, 0.56, 0.014, 0.32])
    fig.colorbar(im, cax=cax, label="Measured intensity (dB)", extend="both")
    save_figure(fig, out, f"{r['acquisition']}_frame_{r['matlab_bscan']:03d}_detail", pdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 7, 255, 511], help="Zero-based TIFF B-scan indices")
    args = parser.parse_args()
    if len(set(args.frames)) != len(args.frames) or not all(0 <= y < 512 for y in args.frames):
        parser.error("Frames must be distinct indices in 0…511")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (HERE / "runs" / f"alignment_05_{stamp}").resolve()
    if HERE not in out.parents:
        raise ValueError("Output must resolve within this experiment directory")
    out.mkdir(parents=True, exist_ok=False)
    frames, calibration, sources = load_frames(args.frames)
    report = {
        "created_utc": stamp,
        "purpose": "Visual alignment review only; no CF fitting or DiffOCT optimization",
        "noise_convention": "User confirmed 2026-09-28: J=(I-N)/(S*H); exported N is measured-space linear intensity",
        "pixel_exclusion_policy": "User requested exclusion of individual nonpositive corrected pixels, retaining other pixels in each A-line",
        "candidate_mapping": {"0.5% H": "MATLAB volume 1 / Python 0", "0.5% L": "MATLAB volume 2 / Python 1"},
        "coordinates": "TIFF page (z,x); HDF5 mask[v,y,x,z].T; display z,x=1..500; no added shifts or flips; zFP origin remains to confirm",
        "calibration": calibration,
        "sources": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frames": [f["report"] for f in frames],
    }
    with PdfPages(out / "alignment_review.pdf") as pdf:
        overview(frames, out, pdf)
        overview(frames, out, pdf, zoom=True)
        for frame in frames:
            detail(frame, out, pdf)
    (out / "alignment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(out / "comparison_masks.npz",
                        original_cf_mask=np.stack([f["fit"] for f in frames]),
                        comparison_mask=np.stack([f["comparison_mask"] for f in frames]),
                        added_exclusions=np.stack([f["invalid"] for f in frames]),
                        python_volume=np.array([f["report"]["python_volume"] for f in frames]),
                        python_frame=np.array([f["report"]["python_bscan"] for f in frames]))
    text = """# 0.5% liquid Intralipid alignment review

The user confirmed on September 28 that Sebastian's CF subtracts the exported
noise in measured linear intensity before sensitivity/confocal correction:
`J = (I - N) / (S * H)`. This resolves the pending noise-domain question in the
September 20 plan. The user requested excluding individual nonpositive corrected
pixels from DiffOCT fitting and comparison summaries, retaining the remaining
valid pixels in each A-line. CF's internal treatment of such pixels is unverified.

## Plots

- [Full-depth overview](alignment_overview.png)
- [Surface zoom](surface_zoom.png)
- [Complete PDF, including individual frame profiles/calibration](alignment_review.pdf)
- [Indices, mask statistics, calibration and source metadata](alignment_report.json)
- [Original CF mask, comparison mask and added exclusions](comparison_masks.npz)

Both 0.5% H/L TIFFs are shown at MATLAB B-scans 1, 8, 256 and 512
(Python 0, 7, 255 and 511 for the default invocation).

Colors: cyan = supplied CF fit-mask boundary; green dashed = supplied sample-mask
boundary for context; yellow dashed = exported focal depth; red = I <= N inside
the fit mask. The original 100-pixel CF mask is preserved for provenance.
The saved comparison mask excludes every red pixel individually.
Images show measured intensity decoded with the 150 dB range, using a shared
75–115 dB display window. Display clipping does not change numerical values.

## Conventions to review

1. Candidate association: H TIFF with MATLAB volume 1; L TIFF with volume 2.
2. TIFF rows are depth z, columns are lateral A-lines x. HDF5 masks are transposed
   from (x,z) into (z,x). Depth increases downward. No extra shift or flip is applied.
3. Axes use MATLAB indices starting at 1. Frame titles also state Python indices.
   Mask boundaries are at pixel edges; the reported first/last mask samples are
   at pixel centers. Focal depth uses the exported value directly on z = 1…500;
   its exact origin remains to confirm and cannot reliably be settled by a
   full-volume plot alone.

The individual pages include A-lines 51, 251 and 451 (Python 50, 250, 450),
the exported noise curve, and sensitivity/confocal curves. The first L frame
has nonpositive corrected pixels inside the supplied fitting band; the JSON
reports their number and overlap with finite exported CF estimates.

These figures are for alignment review before the 0.5% optimization. No model
has been fitted. Source data and the manuscript are unchanged.
"""
    (out / "README.md").write_text(text)
    print(f"Wrote alignment review: {out}", flush=True)


if __name__ == "__main__":
    main()
