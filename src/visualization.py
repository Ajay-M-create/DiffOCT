"""3D volume rendering utilities for OCT data.

Provides PyVista-based volume rendering for:
  1. Phantom validation: side-by-side phantom dB (gray) + recovered mu (inferno)
  2. Clinical arteries: side-by-side OCT (Reds) + OAC map (plasma)

PyVista is imported lazily so that scripts can import this module without
requiring PyVista at the top level.
"""
from __future__ import annotations

import os

import numpy as np

# Physical pixel sizes (mm) — matches config.yaml
DX_MM = 0.0093   # lateral (A-line spacing)
DY_MM = 0.0093   # elevation (B-scan spacing)
DZ_MM = 0.0054   # axial (depth pixel spacing)

# Encoding constants
DB_SCALE = 130.0 / 65535.0
MU_SCALE = 20.0 / 65535.0
NOISE_FLOOR_DB = 79.0

TOMOGRAM_INFO = {
    0: ("1.0%", 1), 1: ("1.0%", 2),
    2: ("1.5%", 1), 3: ("1.5%", 2),
    4: ("2.0%", 1), 5: ("2.0%", 2),
    6: ("3.0%", 1), 7: ("3.0%", 2),
    8: ("5.0%", 1), 9: ("5.0%", 2),
}


def phantom_label(idx: int) -> str:
    """Human-readable label for a tomogram index."""
    conc, focal = TOMOGRAM_INFO[idx]
    return f"IL_{conc.rstrip('%')}pct_rep{focal}"


def volume_to_imagedata(volume: np.ndarray, dx: float, dy: float, dz: float,
                        scalar_name: str = "values"):
    """Convert a numpy volume (n_bscans, n_depth, n_alines) to PyVista ImageData."""
    import pyvista as pv

    n_bscans, n_depth, n_alines = volume.shape
    data = volume.transpose(2, 0, 1).flatten(order="F")

    grid = pv.ImageData(
        dimensions=(n_alines, n_bscans, n_depth),
        spacing=(dx, dy, dz),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data[scalar_name] = data
    return grid


def downsample_volume(volume: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a volume by an integer factor along all axes using block mean."""
    if factor <= 1:
        return volume
    ny, nz, nx = volume.shape
    ny2 = (ny // factor) * factor
    nz2 = (nz // factor) * factor
    nx2 = (nx // factor) * factor
    v = volume[:ny2, :nz2, :nx2]
    v = v.reshape(ny2 // factor, factor, nz2 // factor, factor,
                  nx2 // factor, factor)
    return v.mean(axis=(1, 3, 5)).astype(np.float32)


def add_volume_render(plotter, grid, cmap: str, clim: list[float],
                      scalar_name: str, opacity: list[float],
                      show_scalar_bar: bool = True,
                      scalar_bar_title: str = "") -> None:
    """Add semi-transparent volume rendering and bounding box to the plotter."""
    sbar_args = None
    if show_scalar_bar:
        sbar_args = dict(
            title=scalar_bar_title, color="white",
            label_font_size=28, title_font_size=32,
            n_labels=5, fmt="%.1f",
        )
    plotter.add_volume(
        grid, scalars=scalar_name, cmap=cmap, clim=clim,
        opacity=opacity, show_scalar_bar=show_scalar_bar,
        scalar_bar_args=sbar_args,
    )
    plotter.add_mesh(grid.outline(), color="gray", line_width=1)


def setup_camera(plotter, grid, azimuth: float = -50,
                 elevation: float = 25) -> tuple[float, float, float]:
    """Position camera for 3D perspective with depth (Z) pointing down."""
    bounds = grid.bounds
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2

    diag = np.sqrt(
        (bounds[1] - bounds[0]) ** 2
        + (bounds[3] - bounds[2]) ** 2
        + (bounds[5] - bounds[4]) ** 2
    )
    r = diag * 2.2
    az = np.radians(azimuth)
    el = np.radians(elevation)
    cam_x = cx + r * np.cos(el) * np.cos(az)
    cam_y = cy + r * np.cos(el) * np.sin(az)
    cam_z = cz - r * np.sin(el)

    plotter.camera.position = (cam_x, cam_y, cam_z)
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.up = (0, 0, -1)
    plotter.camera.view_angle = 30
    return (cx, cy, cz)


def load_intmask_tissue_mean(phantom_index: int, input_base: str,
                             total_frames: int = 512) -> float:
    """Compute IntMask-filtered tissue mean from per-frame intmask_mu.npy files."""
    label = phantom_label(phantom_index)
    phantom_dir = os.path.join(input_base, label)
    all_pixels = []
    for b in range(total_frames):
        mu_path = os.path.join(phantom_dir, f"bscan_{b:03d}", "intmask_mu.npy")
        if os.path.exists(mu_path):
            all_pixels.append(np.load(mu_path))
    if all_pixels:
        pool = np.concatenate(all_pixels)
        return float(pool.mean())
    return float("nan")


def load_volumes(phantom_index: int, model: str, input_base: str,
                 phantom_tiff_path: str,
                 total_frames: int = 512) -> tuple[np.ndarray, np.ndarray, float]:
    """Load phantom dB and recovered mu volumes.

    Args:
        phantom_index: Tomogram index (0-9).
        model: 'ss' or 'ms'.
        input_base: Base output directory containing per-phantom results.
        phantom_tiff_path: Path to the despeckled TIFF file.
        total_frames: Number of B-scan frames.

    Returns:
        phantom_db: (n_bscans, n_depth, n_alines) float32, dB values
        mu_mm: (n_bscans, n_depth, n_alines) float32, mm^-1
        tissue_mean: IntMask-filtered tissue mean mu (mm^-1)
    """
    import tifffile

    print(f"Loading phantom: {os.path.basename(phantom_tiff_path)}")
    phantom_db = tifffile.imread(phantom_tiff_path).astype(np.float32) * DB_SCALE
    n_bscans = phantom_db.shape[0]

    label = phantom_label(phantom_index)
    phantom_dir = os.path.join(input_base, label)

    print(f"Assembling mu volume from {phantom_dir}/bscan_NNN/{model}/mu_final_mm.npy ...")
    mu_mm = np.zeros_like(phantom_db)
    loaded = 0
    for b in range(n_bscans):
        npy_path = os.path.join(phantom_dir, f"bscan_{b:03d}", model, "mu_final_mm.npy")
        if os.path.exists(npy_path):
            mu_mm[b] = np.load(npy_path)
            loaded += 1
    print(f"  Loaded {loaded}/{n_bscans} frames")
    if loaded == 0:
        raise FileNotFoundError(f"No mu_final_mm.npy files found in {phantom_dir}")

    tissue_mean = load_intmask_tissue_mean(phantom_index, input_base, total_frames)

    print(f"  Phantom: {phantom_db.shape}, dB [{phantom_db.min():.1f}, {phantom_db.max():.1f}]")
    print(f"  Mu: {mu_mm.shape}, [{mu_mm.min():.3f}, {mu_mm.max():.3f}] mm^-1")
    print(f"  IntMask tissue mu: {tissue_mean:.2f} mm^-1")
    return phantom_db, mu_mm, tissue_mean


def render_3d_volume(phantom_db: np.ndarray, mu_mm: np.ndarray,
                     output_path: str, model: str,
                     phantom_index: int, tissue_mean: float = float("nan"),
                     downsample: int = 1,
                     azimuth: float = -50, elevation: float = 25,
                     render_gif: bool = False, gif_frames: int = 720) -> None:
    """Publication-ready side-by-side 3D rendering: OCT phantom + recovered mu.

    Args:
        phantom_db: (n_bscans, n_depth, n_alines) dB volume.
        mu_mm: (n_bscans, n_depth, n_alines) attenuation volume.
        output_path: Path for the output PNG.
        model: 'ss' or 'ms'.
        phantom_index: Tomogram index (0-9).
        tissue_mean: IntMask-filtered tissue mean mu.
        downsample: Downsampling factor for rendering.
        azimuth: Camera azimuth angle.
        elevation: Camera elevation angle.
        render_gif: Whether to also render a rotating GIF.
        gif_frames: Number of frames in the GIF.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    conc, focal = TOMOGRAM_INFO[phantom_index]
    mu_text = f"{tissue_mean:.2f}" if not np.isnan(tissue_mean) else "N/A"

    db_vol = downsample_volume(phantom_db, downsample)
    mu_vol = downsample_volume(mu_mm, downsample)
    ds = downsample
    dx, dy, dz = DX_MM * ds, DY_MM * ds, DZ_MM * ds

    print(f"Building ImageData grids (downsample={ds}, shape={db_vol.shape})...")
    grid_db = volume_to_imagedata(db_vol, dx, dy, dz, "dB")
    grid_mu = volume_to_imagedata(mu_vol, dx, dy, dz, "mu")

    bounds = grid_db.bounds
    x_mm = bounds[1] - bounds[0]
    y_mm = bounds[3] - bounds[2]
    z_mm = bounds[5] - bounds[4]
    n_bscans, n_depth, n_alines = phantom_db.shape

    db_vmin = NOISE_FLOOR_DB
    db_vmax = float(np.max(phantom_db))
    opacity_db = [0.0, 0.2, 0.4, 0.6, 0.8]
    opacity_mu = [0.0, 0.0, 0.5, 0.8, 1.0]

    pl = pv.Plotter(
        shape=(1, 2), off_screen=True,
        window_size=(3200, 1600), border=False,
    )

    # Left: OCT Phantom
    pl.subplot(0, 0)
    pl.set_background("black")
    add_volume_render(
        pl, grid_db, cmap="gray", clim=[db_vmin, db_vmax], scalar_name="dB",
        opacity=opacity_db,
        show_scalar_bar=True, scalar_bar_title="dB",
    )
    pl.add_text(
        f"OCT Phantom ({conc} Intralipid, F{focal})",
        position="upper_left", font_size=24, color="white",
    )
    dims_text = (f"{x_mm:.1f} x {y_mm:.1f} x {z_mm:.1f} mm  |  "
                 f"{n_bscans} x {n_depth} x {n_alines} voxels")
    pl.add_text(dims_text, position="lower_right", font_size=14, color="gray")

    # Right: Attenuation Coefficient
    pl.subplot(0, 1)
    pl.set_background("black")
    add_volume_render(
        pl, grid_mu, cmap="inferno", clim=[0, 12], scalar_name="mu",
        opacity=opacity_mu,
        show_scalar_bar=True, scalar_bar_title="\u03bc [mm\u207b\u00b9]",
    )
    pl.add_text(
        f"Recovered \u03bc ({model.upper()}) \u2014 tissue mean: {mu_text} mm\u207b\u00b9",
        position="upper_left", font_size=24, color="white",
    )
    pl.add_text(dims_text, position="lower_right", font_size=14, color="gray")

    pl.link_views()
    setup_camera(pl, grid_db, azimuth=azimuth, elevation=elevation)

    print(f"Rendering to {output_path} ...")
    pl.screenshot(output_path, transparent_background=False)
    pl.close()
    print(f"  Saved: {output_path}")

    # Optional rotating GIF
    if render_gif:
        gif_path = output_path.replace(".png", ".gif")
        _render_rotation_gif(
            phantom_db, mu_mm, gif_path, model, phantom_index,
            tissue_mean, max(downsample, 2), azimuth, elevation, gif_frames,
        )


def _render_rotation_gif(phantom_db: np.ndarray, mu_mm: np.ndarray,
                         output_path: str, model: str, phantom_index: int,
                         tissue_mean: float, downsample: int,
                         azimuth: float, elevation: float,
                         n_frames: int) -> None:
    """360-degree rotating GIF with volume rendering."""
    import pyvista as pv
    import imageio
    import time as _time
    pv.OFF_SCREEN = True

    conc, focal = TOMOGRAM_INFO[phantom_index]
    mu_text = f"{tissue_mean:.2f}" if not np.isnan(tissue_mean) else "N/A"

    db_vol = downsample_volume(phantom_db, downsample)
    mu_vol = downsample_volume(mu_mm, downsample)
    dx, dy, dz = DX_MM * downsample, DY_MM * downsample, DZ_MM * downsample

    grid_db = volume_to_imagedata(db_vol, dx, dy, dz, "dB")
    grid_mu = volume_to_imagedata(mu_vol, dx, dy, dz, "mu")

    db_vmin = NOISE_FLOOR_DB
    db_vmax = float(np.max(phantom_db))
    opacity_db = [0.0, 0.2, 0.4, 0.6, 0.8]
    opacity_mu = [0.0, 0.0, 0.5, 0.8, 1.0]

    pl = pv.Plotter(
        shape=(1, 2), off_screen=True,
        window_size=(1920, 960), border=False,
    )

    pl.subplot(0, 0)
    pl.set_background("black")
    add_volume_render(pl, grid_db, cmap="gray", clim=[db_vmin, db_vmax],
                      scalar_name="dB", opacity=opacity_db, show_scalar_bar=False)
    pl.add_text(f"OCT Phantom ({conc} IL, F{focal})",
                position="upper_left", font_size=12, color="white")

    pl.subplot(0, 1)
    pl.set_background("black")
    add_volume_render(pl, grid_mu, cmap="inferno", clim=[0, 12],
                      scalar_name="mu", opacity=opacity_mu, show_scalar_bar=False)
    pl.add_text(f"\u03bc ({model.upper()}) \u2014 {mu_text} mm\u207b\u00b9",
                position="upper_left", font_size=12, color="white")

    pl.link_views()
    setup_camera(pl, grid_db, azimuth=azimuth, elevation=elevation)

    pl.show(auto_close=False)
    pos = list(pl.camera.position)
    focal_pt = list(pl.camera.focal_point)
    up = list(pl.camera.up)
    dx_c = pos[0] - focal_pt[0]
    dy_c = pos[1] - focal_pt[1]
    radius = np.sqrt(dx_c ** 2 + dy_c ** 2)
    z_cam = pos[2]
    start_angle = np.degrees(np.arctan2(dy_c, dx_c))

    print(f"Rendering {n_frames} rotation frames ...")
    writer = imageio.get_writer(output_path, mode="I", duration=8, loop=0)
    t0 = _time.time()
    for i in range(n_frames):
        angle = start_angle + i * (360.0 / n_frames)
        rad = np.radians(angle)
        pl.camera.position = (
            focal_pt[0] + radius * np.cos(rad),
            focal_pt[1] + radius * np.sin(rad),
            z_cam,
        )
        pl.camera.focal_point = focal_pt
        pl.camera.up = up
        pl.render()
        writer.append_data(pl.screenshot(return_img=True))
    pl.close()
    writer.close()

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    elapsed = _time.time() - t0
    print(f"  Saved: {output_path} ({size_mb:.1f} MB, {elapsed:.0f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# Artery (clinical IV-OCT) volume rendering
# ═══════════════════════════════════════════════════════════════════════════

def _artery_volume_to_imagedata(volume: np.ndarray, dx: float, dy: float,
                                dz: float, scalar_name: str = "values"):
    """Convert an artery NIfTI volume (X, Y, Z) to PyVista ImageData.

    Unlike phantom volumes (n_bscans, n_depth, n_alines), artery NIfTIs
    have shape (rows, cols, frames) with no axis transposition needed.
    """
    import pyvista as pv

    nx, ny, nz = volume.shape
    data = volume.flatten(order="F")
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(dx, dy, dz),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data[scalar_name] = data
    return grid


def upsample_volume_z(volume: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a volume by integer factor along the Z (frame) axis only."""
    if factor <= 1:
        return volume
    return np.repeat(volume, factor, axis=2).astype(np.float32)


def render_artery_3d(oct_vol: np.ndarray, oac_vol: np.ndarray,
                     output_path: str, patient_id: str,
                     tissue_mean: float = float("nan"),
                     downsample: int = 1, upsample: int = 1,
                     mu_clim_max: float = 12.0,
                     azimuth: float = 10, elevation: float = 20,
                     render_gif: bool = False, gif_frames: int = 360) -> None:
    """Publication-ready side-by-side: OCT artery (Reds) + OAC (plasma).

    Args:
        oct_vol: (rows, cols, frames) raw OCT volume (linear intensity 0-1).
        oac_vol: (rows, cols, frames) OAC volume (mm^-1).
        output_path: Path for the output PNG.
        patient_id: e.g. '02008Pre' or '02008Post'.
        tissue_mean: Mean OAC over tissue voxels.
        downsample: Downsampling factor.
        upsample: Upsampling factor along Z axis.
        mu_clim_max: Upper clim for OAC colorbar.
        azimuth: Camera azimuth angle.
        elevation: Camera elevation angle.
        render_gif: Whether to also render a rotating GIF.
        gif_frames: Number of frames in the GIF.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb()
    except Exception:
        pass

    is_pre = patient_id.lower().endswith("pre")
    timepoint = "Pre-IVL" if is_pre else "Post-IVL"
    num = patient_id.replace("Pre", "").replace("Post", "")
    mu_text = f"{tissue_mean:.2f}" if not np.isnan(tissue_mean) else "N/A"

    from scipy.ndimage import gaussian_filter

    oct_ds = downsample_volume(oct_vol, downsample)
    oac_ds = downsample_volume(oac_vol, downsample)
    oct_ds = upsample_volume_z(oct_ds, upsample)
    oac_ds = upsample_volume_z(oac_ds, upsample)
    scale = downsample / upsample

    oct_ds = gaussian_filter(oct_ds, sigma=2.0).astype(np.float32)
    oac_ds = gaussian_filter(oac_ds, sigma=2.0).astype(np.float32)

    dx = dy = dz = 1.0 * scale

    print(f"Building ImageData grids (downsample={downsample}, upsample={upsample}, shape={oct_ds.shape})...")
    grid_oct = _artery_volume_to_imagedata(oct_ds, dx, dy, dz, "OCT")
    grid_oac = _artery_volume_to_imagedata(oac_ds, dx, dy, dz, "OAC")

    oct_vmin = 0.20
    oct_vmax = float(np.max(oct_ds))
    opacity_oct = [0.0, 0.1, 0.3, 0.5, 0.7]
    opacity_oac = [0.0, 0.0, 0.5, 0.8, 1.0]

    pl = pv.Plotter(
        shape=(1, 2), off_screen=True,
        window_size=(3200, 1600), border=False,
    )

    # Left: Raw OCT
    pl.subplot(0, 0)
    pl.set_background("black")
    sbar_oct = dict(title="Intensity", color="white",
                    label_font_size=28, title_font_size=32,
                    n_labels=5, fmt="%.2f")
    pl.add_volume(grid_oct, scalars="OCT", cmap="Reds",
                  clim=[oct_vmin, oct_vmax], opacity=opacity_oct,
                  show_scalar_bar=True, scalar_bar_args=sbar_oct)
    pl.add_text(f"Patient {num} {timepoint} \u2014 OCT",
                position="upper_left", font_size=24, color="white")
    pl.add_text(f"{oct_vol.shape[0]} x {oct_vol.shape[1]} x {oct_vol.shape[2]} voxels",
                position="lower_right", font_size=14, color="gray")

    # Right: OAC
    pl.subplot(0, 1)
    pl.set_background("black")
    sbar_oac = dict(title="\u03bc [mm\u207b\u00b9]", color="white",
                    label_font_size=28, title_font_size=32,
                    n_labels=5, fmt="%.1f")
    pl.add_volume(grid_oac, scalars="OAC", cmap="plasma",
                  clim=[0, mu_clim_max], opacity=opacity_oac,
                  show_scalar_bar=True, scalar_bar_args=sbar_oac)
    pl.add_text(f"Patient {num} {timepoint} \u2014 OAC (tissue mean: {mu_text} mm\u207b\u00b9)",
                position="upper_left", font_size=24, color="white")
    pl.add_text(f"{oac_vol.shape[0]} x {oac_vol.shape[1]} x {oac_vol.shape[2]} voxels",
                position="lower_right", font_size=14, color="gray")

    pl.link_views()
    setup_camera(pl, grid_oct, azimuth=azimuth, elevation=elevation)

    print(f"Rendering to {output_path} ...")
    pl.screenshot(output_path, transparent_background=False)
    pl.close()
    print(f"  Saved: {output_path}")

    if render_gif:
        gif_path = output_path.replace(".png", ".gif")
        _render_artery_rotation_gif(
            oct_vol, oac_vol, gif_path, patient_id, tissue_mean,
            max(downsample, 2), upsample, mu_clim_max,
            azimuth, elevation, gif_frames,
        )


def _render_artery_rotation_gif(oct_vol, oac_vol, output_path, patient_id,
                                tissue_mean, downsample, upsample,
                                mu_clim_max, azimuth, elevation, n_frames):
    """360-degree rotating GIF for artery volumes."""
    import pyvista as pv
    import imageio
    import time as _time
    pv.OFF_SCREEN = True

    is_pre = patient_id.lower().endswith("pre")
    timepoint = "Pre-IVL" if is_pre else "Post-IVL"
    num = patient_id.replace("Pre", "").replace("Post", "")
    mu_text = f"{tissue_mean:.2f}" if not np.isnan(tissue_mean) else "N/A"

    from scipy.ndimage import gaussian_filter

    oct_ds = downsample_volume(oct_vol, downsample)
    oac_ds = downsample_volume(oac_vol, downsample)
    oct_ds = upsample_volume_z(oct_ds, upsample)
    oac_ds = upsample_volume_z(oac_ds, upsample)
    scale = downsample / upsample

    oct_ds = gaussian_filter(oct_ds, sigma=2.0).astype(np.float32)
    oac_ds = gaussian_filter(oac_ds, sigma=2.0).astype(np.float32)

    dx = dy = dz = 1.0 * scale
    grid_oct = _artery_volume_to_imagedata(oct_ds, dx, dy, dz, "OCT")
    grid_oac = _artery_volume_to_imagedata(oac_ds, dx, dy, dz, "OAC")

    oct_vmin = 0.20
    oct_vmax = float(np.max(oct_ds))
    opacity_oct = [0.0, 0.1, 0.3, 0.5, 0.7]
    opacity_oac = [0.0, 0.0, 0.5, 0.8, 1.0]

    pl = pv.Plotter(shape=(1, 2), off_screen=True,
                    window_size=(1920, 960), border=False)

    pl.subplot(0, 0)
    pl.set_background("black")
    pl.add_volume(grid_oct, scalars="OCT", cmap="Reds",
                  clim=[oct_vmin, oct_vmax], opacity=opacity_oct,
                  show_scalar_bar=False)
    pl.add_text(f"{num} {timepoint} OCT",
                position="upper_left", font_size=12, color="white")

    pl.subplot(0, 1)
    pl.set_background("black")
    pl.add_volume(grid_oac, scalars="OAC", cmap="plasma",
                  clim=[0, mu_clim_max], opacity=opacity_oac,
                  show_scalar_bar=False)
    pl.add_text(f"\u03bc \u2014 {mu_text} mm\u207b\u00b9",
                position="upper_left", font_size=12, color="white")

    pl.link_views()
    setup_camera(pl, grid_oct, azimuth=azimuth, elevation=elevation)

    pl.show(auto_close=False)
    pos = list(pl.camera.position)
    focal_pt = list(pl.camera.focal_point)
    up = list(pl.camera.up)
    dx_c = pos[0] - focal_pt[0]
    dy_c = pos[1] - focal_pt[1]
    radius = np.sqrt(dx_c ** 2 + dy_c ** 2)
    z_cam = pos[2]
    start_angle = np.degrees(np.arctan2(dy_c, dx_c))

    print(f"Rendering {n_frames} rotation frames ...")
    writer = imageio.get_writer(output_path, mode="I", duration=8, loop=0)
    t0 = _time.time()
    for i in range(n_frames):
        angle = start_angle + i * (360.0 / n_frames)
        rad = np.radians(angle)
        pl.camera.position = (
            focal_pt[0] + radius * np.cos(rad),
            focal_pt[1] + radius * np.sin(rad),
            z_cam,
        )
        pl.camera.focal_point = focal_pt
        pl.camera.up = up
        pl.render()
        writer.append_data(pl.screenshot(return_img=True))
    pl.close()
    writer.close()

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    elapsed = _time.time() - t0
    print(f"  Saved: {output_path} ({size_mb:.1f} MB, {elapsed:.0f}s)")
