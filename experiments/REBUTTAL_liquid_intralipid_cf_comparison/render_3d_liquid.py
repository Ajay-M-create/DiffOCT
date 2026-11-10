"""3D renders of a liquid Intralipid phantom: Figure 3(j)/(k) equivalents.

Style follows the user's reference panels: white background, one PyVista
offscreen view per panel (1600x1600, camera azimuth -50 / elevation 25 as in
src/visualization.render_3d_volume), bold title above each panel, and each
colour bar drawn as an outlined strip lying in the bottom plane of the volume
box along its front edge, with tick labels and title in the same plane.
  (j) measured dB volume, gray colormap, opacity ramp [0, .2, .4, .6, .8]
  (k) recovered mu volume, viridis, opacity ramp [0, 0, .5, .8, 1]
      (the mu bar is faded by the same opacity ramp, as in the reference)

Inputs (read-only): measured_db and mu_final from the production run's
plot_data shards (full-depth fit, fixed per-A-line CF gain, 15,000 iterations).
Both volumes keep the full 500-pixel coordinate system and share a geometric
depth band: start at each A-line's original CF-window start and retain the next
DEPTH_PIXELS samples (default 225, including the start). Only finite measured
and recovered values can be drawn. No comparison-mask or intensity-threshold
filter is applied to this band. Other voxels are set to the lower display
limit, where the opacity ramp is 0 (fully transparent).
dB display range: [median over depth of 10*log10(N) for this volume, max of
drawn voxels]. mu display range: [0, MU_MAX].
Pixel sizes: dx = exported lateral spacing; dy assumed equal to dx (the
slow-axis spacing is not exported); dz = exported water axial spacing.

Usage:
    ACQ=H CONC=3 python experiments/REBUTTAL_liquid_intralipid_cf_comparison/render_3d_liquid.py
Env: ACQ (H|L), CONC (percent), MU_MAX (default 4), DOWNSAMPLE (default 1),
     DEPTH_PIXELS (default 225), SERIES (production run folder; default the published run),
     RENDER_IMAGES (default 1; 0 writes only render_data_*.npz and render_params_*.json).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.visualization import volume_to_imagedata, setup_camera, downsample_volume  # noqa: E402

SERIES = Path(os.environ.get("SERIES", HERE / "runs" / "production_HL_full_depth_cf_gain_20260928T175833Z"))
RENDER_IMAGES = os.environ.get("RENDER_IMAGES", "1") == "1"   # 0: raw volume + params only (the notebook renders)
OUT = REPO / "figures" / "data" / "figure3_liquid_phantom"

ACQ = os.environ.get("ACQ", "H")
CONC = float(os.environ.get("CONC", "3"))
MU_MAX = float(os.environ.get("MU_MAX", "4"))
DOWNSAMPLE = int(os.environ.get("DOWNSAMPLE", "1"))
DEPTH_PIXELS = int(os.environ.get("DEPTH_PIXELS", "225"))
if not 1 <= DEPTH_PIXELS <= 500:
    raise ValueError("DEPTH_PIXELS must be between 1 and 500")
AZIMUTH, ELEVATION = -50, 25                   # as render_3d_volume defaults
WINDOW = (1600, 1600)                          # one view of the paper's 3200x1600
OPACITY_DB = [0.0, 0.2, 0.4, 0.6, 0.8]         # as render_3d_volume
OPACITY_MU = [0.0, 0.0, 0.5, 0.8, 1.0]         # as render_3d_volume
TAG = f"{ACQ}_{CONC:g}pct".replace(".", "p") + f"_cfdepth{DEPTH_PIXELS}"
PANELS = {
    "db": dict(cmap="gray", title=f"OCT Phantom ({CONC:g}% Liquid IL)", bar_title="dB",
               opacity=OPACITY_DB, fade_bar=False),
    "mu": dict(cmap="viridis", title=r"Recovered $\mu$ (SS)", bar_title="mu (mm^-1)",
               opacity=OPACITY_MU, fade_bar=True),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_volume():
    ctag = f"{CONC:g}".replace(".", "p")
    shards = sorted(SERIES.glob(f"shards/{ACQ}_{ctag}_*"))
    if len(shards) != 8:
        raise FileNotFoundError(f"Expected 8 shards for {ACQ} {CONC}%, found {len(shards)}")
    db = np.zeros((512, 500, 500), np.float32)
    mu = np.zeros((512, 500, 500), np.float32)
    keep = np.zeros((512, 500, 500), bool)
    starts = np.full((512, 500), -1, np.int16)
    seen = np.zeros(512, bool)
    requested_voxels = 0
    noise_db, sources = [], []
    for s in shards:
        path = s / "plot_data.npz"
        with np.load(path) as d:
            b = d["python_bscan"]
            m = d["mu_final"]
            measured = d["measured_db"]
            fit = d["original_fit_mask"]
            start = d["original_cf_window_start_python"]
            np.testing.assert_array_equal(start, fit.argmax(axis=1))
            nonempty = fit.any(axis=1)
            z = np.arange(m.shape[1])[None, :, None]
            band = (nonempty[:, None, :] & (z >= start[:, None, :])
                    & (z < start[:, None, :] + DEPTH_PIXELS))
            requested_voxels += int(band.sum())
            starts[b] = np.where(nonempty, start, -1)
            keep[b] = band & np.isfinite(m) & np.isfinite(measured)
            db[b] = measured
            mu[b] = np.where(keep[b], m, 0.0)
            noise_db.append(10 * np.log10(d["noise"]))
            dx, dz = float(d["dx_mm"]), float(d["dz_mm"])
        seen[b] = True
        sources.append(dict(path=str(path), sha256=sha256(path)))
        print(f"  loaded {s.name}", flush=True)
    if not seen.all():
        raise RuntimeError("Missing B-scans")
    noise_floor = float(np.median(np.concatenate(noise_db)))
    db = np.where(keep, db, noise_floor)          # lower display limit: opacity 0
    mask_stats = dict(requested_depth_voxels=requested_voxels,
                      retained_voxels=int(keep.sum()),
                      excluded_nonfinite_voxels=requested_voxels - int(keep.sum()),
                      samples_per_aline_min=int(keep.sum(axis=1).min()),
                      samples_per_aline_max=int(keep.sum(axis=1).max()))
    print(f"  depth-band validation: {mask_stats}", flush=True)
    return db, mu, keep, starts, mask_stats, dx, dz, noise_floor, sources


def add_colorbar_3d(pl, bounds, cmap, clim, title, opacity=None, n_labels=5, fmt="{:.1f}"):
    """Outlined colour strip in the box's bottom plane, outside its front (y_min)
    edge, running along x; tick labels and title lie in the same plane."""
    import pyvista as pv
    x0, x1, y0, y1, z0, z1 = bounds
    L = x1 - x0
    w, gap = 0.07 * L, 0.07 * L
    ya, yb = y0 - gap - w, y0 - gap
    xs, xe = x1 - 0.08 * L, x0 + 0.08 * L      # low end at large x: lower left on screen
    n = 256
    t = np.linspace(0, 1, n)
    pts = np.array([[xs + f * (xe - xs), y, z1] for f in t for y in (ya, yb)])
    strip = pv.StructuredGrid()
    strip.points = pts
    strip.dimensions = (2, n, 1)
    strip.point_data["s"] = np.repeat(clim[0] + t * (clim[1] - clim[0]), 2)
    kw = dict(opacity=opacity) if opacity is not None else {}
    pl.add_mesh(strip, scalars="s", cmap=cmap, clim=clim, show_scalar_bar=False,
                lighting=False, **kw)
    corners = np.array([[xs, ya, z1], [xe, ya, z1], [xe, yb, z1], [xs, yb, z1], [xs, ya, z1]])
    pl.add_mesh(pv.lines_from_points(corners), color="black", line_width=4)
    # text in the bottom plane, readable from the camera side:
    # local +x -> world -x (along the bar), local +y -> world +y (toward box), normal -> -z
    R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float)

    def text(s, height, cx, cy):
        m = pv.Text3D(s, depth=0.0)
        b = m.bounds
        m.points = (m.points - [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, 0]) * (height / max(b[3] - b[2], 1e-9))
        m.points = m.points @ R.T + [cx, cy, z1]
        pl.add_mesh(m, color="black", lighting=False)

    th = 0.05 * L
    for i in range(n_labels):
        f = i / (n_labels - 1)
        text(fmt.format(clim[0] + f * (clim[1] - clim[0])), th, xs + f * (xe - xs), ya - 1.0 * th)
    text(title, 1.1 * th, (xs + xe) / 2, ya - 2.9 * th)


def render(vol, name, clim, dx, dz, png):
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = PANELS[name]
    ds = DOWNSAMPLE
    grid = volume_to_imagedata(downsample_volume(vol, ds), dx * ds, dx * ds, dz * ds, "v")
    pl = pv.Plotter(off_screen=True, window_size=WINDOW, border=False)
    pl.set_background("white")
    pl.add_volume(grid, scalars="v", cmap=p["cmap"], clim=clim, opacity=p["opacity"],
                  show_scalar_bar=False)
    pl.add_mesh(grid.outline(), color="gray", line_width=1)
    add_colorbar_3d(pl, grid.bounds, p["cmap"], clim, p["bar_title"],
                    opacity=p["opacity"] if p["fade_bar"] else None)
    setup_camera(pl, grid, azimuth=AZIMUTH, elevation=ELEVATION)
    pl.screenshot(str(png), transparent_background=False)
    pl.close()


def save_panel(png, name):
    """Crop the white margin, add the bold title, save PDF + 600 dpi JPEG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    img = np.array(Image.open(png).convert("RGB"))
    ink = (img < 245).any(axis=2)
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    pad = 20
    img = img[max(rows[0] - pad, 0):rows[-1] + pad, max(cols[0] - pad, 0):cols[-1] + pad]
    fig, ax = plt.subplots(figsize=(6, 6 * img.shape[0] / img.shape[1]))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(PANELS[name]["title"], fontsize=11, fontweight="bold", loc="left", pad=4)
    stem = OUT / f"liquid_3d_{name}_{TAG}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(f"{stem}.jpeg", dpi=600, bbox_inches="tight", pad_inches=0.05,
                facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)
    return [f"{stem.name}.pdf", f"{stem.name}.jpeg"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {ACQ} {CONC}% (downsample {DOWNSAMPLE})", flush=True)
    db, mu, keep, starts, mask_stats, dx, dz, noise_floor, sources = load_volume()
    print(f"  drawn voxels: {keep.mean():.4f} of the volume", flush=True)
    clims = dict(db=[noise_floor, float(db[keep].max())], mu=[0.0, MU_MAX])
    raw = OUT / f"render_data_{TAG}.npz"
    np.savez_compressed(raw, db_vol=db, mu_vol=mu, keep_mask=keep,
                        cf_window_start_python=starts, depth_pixels=DEPTH_PIXELS)
    outputs = [raw.name]
    for name, vol in (("db", db), ("mu", mu)) if RENDER_IMAGES else ():
        png = OUT / f"render_{name}_{TAG}.png"
        render(vol, name, clims[name], dx, dz, png)
        outputs += [png.name] + save_panel(png, name)
        print(f"  saved {name}", flush=True)
    summary = {}
    with open(SERIES / "volume_summary.csv") as f:
        header = f.readline().strip().split(",")
        for line in f:
            row = dict(zip(header, line.strip().split(",")))
            if row["acquisition"] == ACQ and float(row["concentration_percent"]) == CONC:
                summary = row
    params = dict(
        acquisition=ACQ, concentration_percent=CONC,
        figure3_equivalent={"j": f"liquid_3d_db_{TAG}", "k": f"liquid_3d_mu_{TAG}"},
        pixel_size_mm=dict(dx=dx, dy=dx, dz=dz),
        dy_note="Slow-axis spacing not exported; assumed equal to the exported lateral spacing",
        volume_shape_bscan_depth_aline=list(db.shape),
        mask_rule=(f"Per A-line: original CF-window start <= depth < start + {DEPTH_PIXELS}; "
                   "intersect finite measured_db and mu_final; identical mask for both volumes. "
                   "No comparison_mask or intensity-threshold filtering. Other voxels are set "
                   "to the lower display limit (opacity 0); full depth coordinates kept."),
        depth_pixels=DEPTH_PIXELS, depth_band_width_mm=DEPTH_PIXELS * dz,
        mask_validation=mask_stats,
        drawn_fraction_of_volume=float(keep.mean()),
        db_clim=clims["db"], db_clim_low_rule="median over depth of 10*log10(noise) for this volume",
        mu_clim=clims["mu"], opacity_db=OPACITY_DB, opacity_mu=OPACITY_MU,
        cmap_db=PANELS["db"]["cmap"], cmap_mu=PANELS["mu"]["cmap"],
        style=("white background; bold titles above panels; colour bars as outlined strips in the "
               "box's bottom plane along its front edge (mu bar faded by its opacity ramp)"),
        window_size=list(WINDOW),
        camera=dict(azimuth=AZIMUTH, elevation=ELEVATION, up=[0, 0, -1], view_angle=30,
                    distance="2.2 x volume diagonal (src.visualization.setup_camera)"),
        downsample=DOWNSAMPLE, diffoct_cf_window_mean_mm_inv=float(summary["diffoct_mean_mm_inv"]),
        cf_mean_mm_inv=float(summary["cf_mean_mm_inv"]),
        outputs=outputs, source_shards=sources,
        script=str(Path(__file__).resolve()), script_sha256=sha256(__file__),
        raw_npz_sha256=sha256(raw))
    (OUT / f"render_params_{TAG}.json").write_text(json.dumps(params, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
