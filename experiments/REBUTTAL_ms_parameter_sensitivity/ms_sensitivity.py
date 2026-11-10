#!/usr/bin/env python3
"""Sensitivity of clinical mu to the assumed EHF parameters g and n_sample (reviewer R3.8, Figure S6).

The five Figure 5 showcase frames (patient 02008Pre, frames 0, 42, 85, 127, 170) are
re-inverted with the multiple-scattering (EHF) forward model while one assumed
parameter is swept and the other is held at its default:

    g sweep : g in G_VALUES,        n_sample = 1.35
    n sweep : n_sample in N_VALUES, g = 0.7

plus one single-scattering (SS) run per frame as the published reference. Everything
else is the published clinical pipeline (experiments/generate_patient_oac.py): same
volume, dB rescaling, polar resampling, linear noise-floor subtraction, noise-floor
mask, confocal parameters, optimizer settings (25,000 iterations, lr 1e-3 cosine,
TV 1e3, mu in [0.01, 20]) and per-iteration MSE-optimal gain. Only stage 1 (mu with
alpha = 1) is run, because the published attenuation maps are the stage-1 mu. Every
configuration of a frame starts from the same published initialization
(torch.manual_seed(42 + frame), mu ~ U(0.01, 10)), so differences between runs come
from the assumed parameter alone. All (frame, configuration) pairs run as one batch.

The MS model is a batched copy of src/forward.py:RasterScanOCTForwardMS with g and
n_sample stored per batch element; src/ is not modified.

Usage:
    python experiments/REBUTTAL_ms_parameter_sensitivity/ms_sensitivity.py run   RUN_DIR
    python experiments/REBUTTAL_ms_parameter_sensitivity/ms_sensitivity.py stage RUN_DIR
    python experiments/REBUTTAL_ms_parameter_sensitivity/ms_sensitivity.py all   RUN_DIR
Env: FRAMES (default "0,42,85,127,170"), NUM_ITERS (default 25000), DEVICE (default cuda).
"""
import hashlib
import json
import math
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
sys.path.insert(0, str(REPO / "experiments"))

PATIENT = "02008Pre"
VOLUME = Path("/home/gridsan/groups/erelab/Ajay/DiffOCT-IVLDatasetsAndSegsPrePost/Pre_Post/"
              "02008Pre_target_Post.nii.gz")        # 171-frame volume behind Figure 5
FRAMES = [int(f) for f in os.environ.get("FRAMES", "0,42,85,127,170").split(",")]
DB_SCALE = 130.0
G_REF, N_REF = 0.7, 1.35                              # defaults used in S5(a)
G_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]                  # symmetric about G_REF
N_VALUES = [1.25, 1.30, 1.35, 1.40, 1.45]             # symmetric about N_REF
FIG5 = REPO / "figures" / "data" / "figure5_clinical" / f"scratch_{PATIENT}"
FIGDATA = Path(os.environ.get("FIGDATA", REPO / "figures" / "data" / "figureS6_ms_parameter_sensitivity"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def configurations():
    """(model, g, n, sweep) per configuration; the reference MS run appears once."""
    cfgs = [("ss", np.nan, np.nan, "ss")]
    cfgs += [("ms", g, N_REF, "g") for g in G_VALUES]
    cfgs += [("ms", G_REF, n, "n") for n in N_VALUES if n != N_REF]
    return cfgs


def build_ms_forward(g_module, config, g_per_item, n_per_item, ms_per_item, device):
    """Batched EHF forward (RasterScanOCTForwardMS math) with per-item g, n and an SS switch."""
    import torch

    class BatchedMSForward(g_module.BatchedSSForward):
        def __init__(self):
            super().__init__(config, batch_size=len(g_per_item), device=device)
            fms = config.forward_model_system
            z_px = torch.arange(self.depth_samples, dtype=torch.float32, device=device)
            w0_mm = float(fms.central_wavelength_nm) * 1e-6 / (math.pi * float(fms.numerical_aperture))
            wH_sq = w0_mm ** 2 * (((z_px - float(fms.focal_plane_position_pixels))
                                   / float(fms.rayleigh_range_pixels)) ** 2 + 1.0)
            z_mm = z_px * self.axial_pixel_size_mm
            g_t = torch.tensor(g_per_item, dtype=torch.float32, device=device).nan_to_num(0.0)
            n_t = torch.tensor(n_per_item, dtype=torch.float32, device=device).nan_to_num(1.0)
            self.register_buffer("_wH_sq", wH_sq.view(1, -1, 1))
            self.register_buffer("_theta_rms_sq", (2.0 * (1.0 - g_t)).view(-1, 1, 1))
            self.register_buffer("_z_over_n_sq", (z_mm.view(1, -1, 1) / n_t.view(-1, 1, 1)) ** 2)
            self.register_buffer("_ms", torch.tensor(ms_per_item, device=device).view(-1, 1, 1))

        def forward(self, mu, alpha, noise_mask=None):
            eps = 1e-12
            mu_active = mu * noise_mask if noise_mask is not None else mu
            alpha_active = alpha * noise_mask if noise_mask is not None else alpha
            cumulative = torch.cumsum(mu_active * self.axial_pixel_size_mm, dim=1)
            tau = torch.cat([torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]], dim=1)
            wS_sq = self._wH_sq + (1.0 / 3.0) * tau * self._theta_rms_sq * self._z_over_n_sq
            e = torch.exp(-tau)
            T_ss = e * e
            T_cross = 4.0 * e * (1.0 - e) / (1.0 + wS_sq / (self._wH_sq + eps))
            T_ms = (1.0 - e).pow(2) * self._wH_sq / (wS_sq + eps)
            transport_dB = torch.where(self._ms, 10.0 * torch.log10(T_ss + T_cross + T_ms + eps),
                                       -self._K * tau)   # SS items: exactly BatchedSSForward
            return (self.gain_dB
                    + 10.0 * torch.log10(alpha_active + eps)
                    + 10.0 * torch.log10(mu_active + eps)
                    + self.confocal_dB
                    + transport_dB)

    return BatchedMSForward().to(device)


def run(run_dir):
    import nibabel as nib
    import torch
    from torch import nn
    import generate_patient_oac as g          # the published clinical pipeline

    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(os.environ.get("DEVICE", "cuda"))
    config, opt_config = g.build_ivoct_config(), g.build_opt_config()
    noise_floor_dB = float(config.forward_model_system.noise_floor_dB)

    vol = nib.load(str(VOLUME))
    assert vol.shape[2] == 171, vol.shape
    data = np.asarray(vol.dataobj, dtype=np.float32)
    scale = DB_SCALE / float(data.max())       # as in generate_patient_oac.main()
    targets_pol = np.stack([g.cartesian_to_polar(data[:, :, f] * scale) for f in FRAMES])
    del data

    cfgs = configurations()
    C, F, D, A = len(cfgs), len(FRAMES), g.DEPTH_PIXELS, g.NUM_ALINES
    item_frame = np.repeat(np.arange(F), C)    # batch item b = frame (b // C), config (b % C)
    item_cfg = np.tile(np.arange(C), F)
    B = C * F
    # The published loss is a mean over a five-frame batch. Scaling both weights by C keeps each
    # item's gradient equal to the published five-frame run (gradient clipping and Adam epsilon
    # otherwise see a loss rescaled by 1/C).
    opt_config.optimization.loss_scale = float(opt_config.optimization.loss_scale) * C
    opt_config.optimization.mu_tv_weight = float(opt_config.optimization.mu_tv_weight) * C

    target = torch.tensor(targets_pol[item_frame], device=device)
    lin = torch.pow(10.0, target / 10.0)       # linear noise-floor subtraction (process_batch)
    target = 10.0 * torch.log10(torch.clamp(lin - 10.0 ** (noise_floor_dB / 10.0), min=1e-12))
    mask = (target >= noise_floor_dB).float()

    mu0 = torch.empty(B, D, A, device=device)
    for fi, frame in enumerate(FRAMES):        # published per-frame init, shared by all configs
        torch.manual_seed(42 + frame)
        init = torch.empty(D, A).uniform_(0.01, 10.0).to(device)   # CPU draw reproduces the published Figure 5 maps (Codex audit, 2026-09-30)
        mu0[item_frame == fi] = init
    mu0 *= mask

    forward = build_ms_forward(
        g, config,
        g_per_item=[cfgs[c][1] for c in item_cfg], n_per_item=[cfgs[c][2] for c in item_cfg],
        ms_per_item=[cfgs[c][0] == "ms" for c in item_cfg], device=device)
    mu_param = nn.Parameter(mu0.clone())
    alpha_param = nn.Parameter(torch.ones(B, D, A, device=device) * mask)
    t0 = time.time()
    mu_best, _, best_mse, gain, best_iter = g.batched_optimize(
        forward, mu_param, alpha_param, target, mask, opt_config, device,
        freeze_alpha=True, freeze_mu=False)
    seconds = time.time() - t0

    shape = (F, C, D, A)
    np.savez_compressed(
        run_dir / "sweep.npz",
        mu_polar=mu_best.cpu().numpy().astype(np.float32).reshape(shape),
        mask_polar=mask.cpu().numpy().astype(bool).reshape(shape)[:, 0],
        target_polar=target.cpu().numpy().astype(np.float32).reshape(shape)[:, 0],
        best_mse_db2=best_mse.cpu().numpy().reshape(F, C), gain_db=gain.cpu().numpy().reshape(F, C),
        best_iter=best_iter.cpu().numpy().reshape(F, C), frames=np.array(FRAMES),
        cfg_model=np.array([c[0] for c in cfgs]), cfg_g=np.array([c[1] for c in cfgs]),
        cfg_n=np.array([c[2] for c in cfgs]), cfg_sweep=np.array([c[3] for c in cfgs]))
    config_record = dict(
        patient=PATIENT, frames=FRAMES, volume=str(VOLUME), volume_sha256=sha256(VOLUME),
        g_values=G_VALUES, n_values=N_VALUES, g_ref=G_REF, n_ref=N_REF,
        configurations=[dict(model=m, g=None if np.isnan(gv) else gv, n_sample=None if np.isnan(nv) else nv,
                             sweep=s) for m, gv, nv, s in cfgs],
        mu_init="U(0.01, 10) mm^-1 on noise-floor-masked pixels, torch.manual_seed(42 + frame) drawn on CPU, "
                "shared by every configuration of a frame",
        stage="1 (mu, alpha fixed at 1)", db_scale=DB_SCALE, noise_floor_dB=noise_floor_dB,
        forward_model_system=dict(config.forward_model_system),
        optimization=dict(opt_config.optimization), runtime_seconds=seconds,
        device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        python=platform.python_version(), torch=torch.__version__,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        script=str(Path(__file__).resolve()), script_sha256=sha256(__file__),
        pipeline_sha256=sha256(REPO / "experiments" / "generate_patient_oac.py"))
    (run_dir / "config.json").write_text(json.dumps(config_record, indent=2, default=str) + "\n")
    print(f"run complete: {F} frames x {C} configurations in {seconds:.0f} s -> {run_dir}", flush=True)


def summarize(d):
    """Per (frame, configuration) tissue statistics relative to the MS reference (g=0.7, n=1.35)."""
    mu, mask = d["mu_polar"], d["mask_polar"]
    model, gs, ns = d["cfg_model"], d["cfg_g"], d["cfg_n"]
    ref = int(np.flatnonzero((model == "ms") & np.isclose(gs, G_REF) & np.isclose(ns, N_REF))[0])
    ss = int(np.flatnonzero(model == "ss")[0])
    F, C = mu.shape[:2]
    mean_mu = np.zeros((F, C)); rel_mean = np.zeros((F, C)); med_abs_rel = np.zeros((F, C))
    frac_gt10 = np.zeros((F, C))
    for f in range(F):
        t = mask[f]
        r = mu[f, ref][t]
        for c in range(C):
            x = mu[f, c][t]
            mean_mu[f, c] = x.mean()
            rel_mean[f, c] = 100.0 * (x.mean() - r.mean()) / r.mean()
            med_abs_rel[f, c] = 100.0 * np.median(np.abs(x - r) / r)
            frac_gt10[f, c] = 100.0 * np.mean(np.abs(x - r) / r > 0.10)   # share of tissue pixels changing >10%
    return dict(ref_index=ref, ss_index=ss, tissue_mean_mu=mean_mu,
                rel_change_mean_pct=rel_mean, median_abs_pixel_change_pct=med_abs_rel,
                frac_pixels_change_gt10_pct=frac_gt10)


def stage(run_dir):
    """Check the SS run of frame 85 against the published map, then stage the S5(b) inputs."""
    import generate_patient_oac as g
    with np.load(run_dir / "sweep.npz") as z:
        d = {k: z[k] for k in z.files}
    s = summarize(d)
    validation = {}
    frames = list(d["frames"])
    if 85 in frames:
        ss_cart = g.polar_to_cartesian(d["mu_polar"][frames.index(85), s["ss_index"]])
        published = np.load(FIG5 / "frame_085" / "mu_cart.npy")
        tissue = published > 0
        diff = np.abs(ss_cart - published)[tissue]
        validation = dict(
            check="SS configuration of frame 85 reproduces the published Figure 5 mu map",
            tissue_pixels=int(tissue.sum()), mean_abs_diff_mm_inv=float(diff.mean()),
            mean_published_mu_mm_inv=float(published[tissue].mean()),
            mean_ss_mu_mm_inv=float(ss_cart[tissue].mean()),
            note="The published map came from a different batch; the loss is a batch mean, "
                 "so agreement is close but not bit for bit.")
    FIGDATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIGDATA / "sensitivity_summary.npz",
        frames=d["frames"], cfg_model=d["cfg_model"], cfg_g=d["cfg_g"], cfg_n=d["cfg_n"],
        cfg_sweep=d["cfg_sweep"], best_mse_db2=d["best_mse_db2"], g_ref=G_REF, n_ref=N_REF,
        **s)
    ref_i = frames.index(85) if 85 in frames else 0
    np.savez_compressed(                        # one frame's maps at the sweep extremes, for context
        FIGDATA / "example_frame_maps.npz", frame=d["frames"][ref_i],
        mu_polar=d["mu_polar"][ref_i], mask_polar=d["mask_polar"][ref_i],
        cfg_g=d["cfg_g"], cfg_n=d["cfg_n"], cfg_model=d["cfg_model"],
        target_polar=d["target_polar"][ref_i],
        target_cart=g.polar_to_cartesian(d["target_polar"][ref_i]),            # noise-floor-corrected OCT (dB)
        mu_ref_cart=g.polar_to_cartesian(d["mu_polar"][ref_i, s["ref_index"]]),  # MS reference OAC
        mask_cart=g.polar_to_cartesian(d["mask_polar"][ref_i].astype(np.float32)) > 0.5)
    shutil.copy2(run_dir / "config.json", FIGDATA / "config.json")
    (FIGDATA / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (FIGDATA / "source_manifest.json").write_text(json.dumps(dict(
        run_dir=str(run_dir), sweep_npz_sha256=sha256(run_dir / "sweep.npz"),
        staged_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")), indent=2) + "\n")
    for c in range(len(d["cfg_model"])):
        print(f"{d['cfg_model'][c]} g={d['cfg_g'][c]:.2f} n={d['cfg_n'][c]:.2f}: "
              f"mean mu {s['tissue_mean_mu'][:, c].mean():.3f} mm^-1, "
              f"d(mean) {s['rel_change_mean_pct'][:, c].mean():+.2f}%, "
              f"median |d mu| {s['median_abs_pixel_change_pct'][:, c].mean():.2f}%")
    print(json.dumps(validation, indent=2))
    print(f"staged -> {FIGDATA}", flush=True)


if __name__ == "__main__":
    mode, run_dir = sys.argv[1], Path(sys.argv[2]).resolve()
    if mode in ("run", "all"):
        run(run_dir)
    if mode in ("stage", "all"):
        stage(run_dir)
