"""Read-only inspection of Sebastian's liquid phantom exports; never runs CF.

Run from the repository root using the existing diff-oct Python environment.
Only audit JSON under this experiment directory is written. Large tomograms
and masks are sampled by B-scan, never loaded as complete 4-D arrays.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import tifffile


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "inputs" / "LiquidIntralipidPhantoms_20260915"
SAMPLE_Y = (7, 255)


def stats(a):
    a = np.asarray(a)
    f = a[np.isfinite(a)]
    return {
        "size": int(a.size), "finite": int(f.size),
        "mean": float(f.mean()) if f.size else None,
        "quantiles_0_1_50_99_100": np.percentile(f, [0, 1, 50, 99, 100]).tolist() if f.size else [],
    }


def main():
    report = {
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(DATA), "resolved_data_path": str(DATA.resolve()),
        "scope": "Export inspection only. No CF fitting, DiffOCT optimization, or Slurm submission.",
        "sample_bscans_zero_based": list(SAMPLE_Y),
        "mat_files": {}, "tiffs": [], "per_volume": [], "mask_samples": {},
    }
    for p in sorted(DATA.glob("*.mat")):
        entry = {"bytes": p.stat().st_size, "datasets": {}}
        with h5py.File(p, "r") as h:
            def visit(name, obj):
                if not isinstance(obj, h5py.Dataset) or name.startswith("#"):
                    return
                d = {"shape_hdf5": list(obj.shape), "dtype": str(obj.dtype),
                     "chunks": list(obj.chunks) if obj.chunks else None}
                if obj.size < 100:
                    d["values"] = obj[...].tolist()
                entry["datasets"][name] = d
            h.visititems(visit)
        report["mat_files"][p.name] = entry
    print("MAT schemas inspected", flush=True)
    with h5py.File(DATA / "CF_Method_Outputs.mat", "r") as h:
        arrays = {k: h[k][...] for k in ("mu", "muFilt", "G", "GFilt", "nosiefloor", "sensitivity", "zFP")}
        report["array_statistics"] = {k: stats(a) for k, a in arrays.items()}
        for v in range(14):
            row = {"matlab_volume_one_based": v + 1}
            row.update({k: stats(a[v]) for k, a in arrays.items()})
            raw, filt = arrays["mu"][v], arrays["muFilt"][v]
            common = np.isfinite(raw) & np.isfinite(filt)
            row["muFilt_removed_finite_mu"] = int((np.isfinite(raw) & ~np.isfinite(filt)).sum())
            row["muFilt_changed_retained_values"] = int((raw[common] != filt[common]).sum())
            report["per_volume"].append(row)
    print("Small parameter arrays inspected", flush=True)
    with h5py.File(DATA / "CF_Method_fit_mask.mat", "r") as h:
        for y in SAMPLE_Y:
            a = h["zFitMask"][:, y, :, :].astype(bool)  # v,x,z
            rows = []
            for v in range(14):
                counts = a[v].sum(axis=-1)
                valid = counts > 0
                start = a[v].argmax(axis=-1)
                end = 499 - a[v, :, ::-1].argmax(axis=-1)
                lengths, frequencies = np.unique(counts, return_counts=True)
                row = {"matlab_volume_one_based": v + 1,
                       "valid_alines": int(valid.sum()),
                       "depth_count_histogram": dict(zip(map(str, lengths), map(int, frequencies))),
                       "start_z_zero_based": stats(start[valid]),
                       "noncontiguous_alines": int(((end - start + 1 != counts) & valid).sum())}
                rows.append(row)
            report["mask_samples"][str(y)] = rows
    print("Mask B-scans inspected", flush=True)
    for p in sorted(DATA.glob("*_DespeckledLogInt.tif")):
        match = re.search(r"IL_(\d+)\.\.(\d+)_LSM03_([HL])", p.name)
        with tifffile.TiffFile(p) as t:
            entry = {"filename": p.name, "bytes": p.stat().st_size,
                     "concentration_percent": float(match[1] + "." + match[2]),
                     "acquisition_label": match[3], "pages": len(t.pages),
                     "shape": list(t.series[0].shape), "dtype": str(t.series[0].dtype),
                     "axes": t.series[0].axes, "description": t.pages[0].description,
                     "sample_bscans": {}}
            for y in SAMPLE_Y:
                a = t.pages[y].asarray().astype(np.float64) * (150.0 / 65535.0)
                profile = np.median(a, axis=1)
                entry["sample_bscans"][str(y)] = {
                    "dB": stats(a),
                    "largest_positive_depth_gradient_zero_based": int(np.argmax(np.diff(profile)) + 1),
                    "median_profile_peak_zero_based": int(np.argmax(profile)),
                }
            report["tiffs"].append(entry)
    print("TIFF headers and sample B-scans inspected", flush=True)
    report["tomogram_samples"] = {}
    for fname, key in [("CF_Method_mu_tom.mat", "tomMu"), ("CF_Method_tom_predict.mat", "tom")]:
        with h5py.File(DATA / fname, "r") as h:
            a = h[key][:, SAMPLE_Y[0], :, :]
            report["tomogram_samples"][fname] = [stats(a[v]) for v in range(14)]
    # This ordering is a documented candidate, not embedded filename metadata
    # from MATLAB. Sample surfaces support it; full alignment remains a gate.
    ordered = sorted(report["tiffs"], key=lambda t: (t["concentration_percent"], t["acquisition_label"]))
    report["candidate_volume_mapping"] = [
        {"matlab_volume_one_based": v + 1, "filename": t["filename"],
         "concentration_percent": t["concentration_percent"],
         "acquisition_label": t["acquisition_label"],
         "run_diffoct": t["concentration_percent"] <= 3.0,
         "status": "Candidate: numeric concentration order, H then L; verify alignment before runs."}
        for v, t in enumerate(ordered)
    ]
    output = HERE / "inspection.json"
    with output.open("w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
