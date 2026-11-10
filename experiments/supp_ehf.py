#!/usr/bin/env python
"""SS vs MS supplementary experiment on real phantom data (Figure S3).

Produces the four cached npz files the Figure S3 notebook reads:

    forward_model_profiles.npz   panels (a) + (b)  — SS / MS A-scan profiles
                                                     on a synthetic homogeneous
                                                     μ = 5.0 mm⁻¹ slab; sweep over
                                                     the Henyey-Greenstein g.
    synthetic_optimization.npz   reference only    — SS + MS recovered μ on a
                                                     synthetic uniform-μ phantom
                                                     (heavy GPU run; not consumed
                                                     by the figure layout but kept
                                                     for completeness).
    real_phantom_panel_c.npz     panel (c)         — SS + MS recovered μ maps for
                                                     the showcase real phantom
                                                     (IL_5.0pct_rep2, B-scan 256),
                                                     loaded from existing
                                                     phantom_validation outputs.
    phantom_boxplot_data.npz     panel (d)         — per-frame tissue-mean μ across
                                                     all 10 acquisitions × 512
                                                     B-scans for both SS and MS.

After all four files exist, _stage_to_figure_paper_S3() copies them into
``figures/data/figureS5_ss_vs_ms/supp_ehf_data/``.

This script assumes both SS and MS variants of phantom_validation_ss_or_ms.py
have already been run end-to-end (their outputs sit at
``outputs/phantom_validation_{ss,ms}/``) — only the synthetic forward profiles
+ panel (c) + panel (d) data come from this script directly.

Usage:
    python experiments/supp_ehf.py                  # produce all 4
    python experiments/supp_ehf.py --no-synthetic   # skip the heavy synthetic optimization
    STAGE_ONLY=1 python experiments/supp_ehf.py     # just stage existing
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PHANTOM_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PHANTOM_ROOT not in sys.path:
    sys.path.insert(0, PHANTOM_ROOT)

from src.forward import RasterScanOCTForward, RasterScanOCTForwardMS
from src.helpers import discover_despeckled_phantoms
from src.roi import load_intmask

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_BASE = os.path.join(PHANTOM_ROOT, "outputs", "supp_ehf")
SHOWCASE_PHANTOM_INDEX = 9          # IL_5.0pct_rep2
SHOWCASE_BSCAN          = 256
DEPTH                   = 500
A_LINES                 = 500
DZ_MM                   = 0.004
TISSUE_START, TISSUE_END = 100, 300
SHOWCASE_ALINE          = 250
MU_VAL                  = 5.0
G_SWEEP                 = [0.0, 0.3, 0.5, 0.7, 0.9]
CONC_ORDER              = ["1.0", "1.5", "2.0", "3.0", "5.0"]


# ---------------------------------------------------------------------------
# (1) forward_model_profiles.npz — synthetic A-scan profiles
# ---------------------------------------------------------------------------

def build_forward_model_profiles(out_path, device):
    """Run SS and MS forward models on a homogeneous μ-slab phantom and save
    the central-A-line profiles + a g sweep for the MS model. CPU/GPU light."""
    cfg = OmegaConf.load(os.path.join(PHANTOM_ROOT, "configs", "config.yaml"))
    cfg.forward_model_system.depth_samples = DEPTH
    cfg.forward_model_system.a_lines       = A_LINES

    # Homogeneous μ slab in the tissue band, otherwise zero
    mu_np = np.zeros((DEPTH, A_LINES), dtype=np.float32)
    mu_np[TISSUE_START:TISSUE_END, :] = MU_VAL
    alpha_np = np.ones_like(mu_np)
    mu_t    = torch.tensor(mu_np,    dtype=torch.float32, device=device)
    alpha_t = torch.tensor(alpha_np, dtype=torch.float32, device=device)

    cfg.forward_model_system.forward_model_type = "ss"
    fwd_ss = RasterScanOCTForward(cfg, device=device)
    fwd_ss.gain_dB.fill_(100.0)
    with torch.no_grad():
        ss_full = fwd_ss(mu_t, alpha_t).cpu().numpy()

    cfg.forward_model_system.forward_model_type = "ms"
    cfg.forward_model_system.g                  = 0.7
    fwd_ms = RasterScanOCTForwardMS(cfg, device=device)
    fwd_ms.gain_dB.fill_(100.0)
    with torch.no_grad():
        ms_full = fwd_ms(mu_t, alpha_t).cpu().numpy()

    # g-anisotropy sweep
    g_profiles = {}
    for g in G_SWEEP:
        cfg.forward_model_system.g = float(g)
        fwd = RasterScanOCTForwardMS(cfg, device=device)
        fwd.gain_dB.fill_(100.0)
        with torch.no_grad():
            out = fwd(mu_t, alpha_t).cpu().numpy()
        g_profiles[float(g)] = out[:, SHOWCASE_ALINE].astype(np.float32)

    payload = dict(
        z_px=np.arange(DEPTH, dtype=np.int64),
        ss_out=ss_full[:, SHOWCASE_ALINE].astype(np.float32),
        ms_out=ms_full[:, SHOWCASE_ALINE].astype(np.float32),
        tissue_start=np.int64(TISSUE_START),
        tissue_end=np.int64(TISSUE_END),
        mu_val=np.float64(MU_VAL),
        aline=np.int64(SHOWCASE_ALINE),
        dz=np.float64(DZ_MM),
        g_values=np.array(G_SWEEP, dtype=np.float64),
        **{f"g_{float(g)}": prof for g, prof in g_profiles.items()},
    )
    np.savez(out_path, **payload)
    print(f"  [Fig S3 (a/b)] {out_path}")


# ---------------------------------------------------------------------------
# (2) synthetic_optimization.npz — synthetic uniform-μ recovery (heavy)
# ---------------------------------------------------------------------------

def build_synthetic_optimization(out_path, device):
    """Generate a synthetic uniform-μ target with the SS forward model and
    invert it with both SS and MS forward models. Heavy GPU run."""
    from src.inverse import optimize, build_noise_floor_mask
    cfg = OmegaConf.load(os.path.join(PHANTOM_ROOT, "configs", "config.yaml"))
    opt_config = OmegaConf.load(
        os.path.join(PHANTOM_ROOT, "configs", "optimization_config_fast.yaml"))
    cfg.forward_model_system.depth_samples = DEPTH
    cfg.forward_model_system.a_lines       = A_LINES
    sc_config = OmegaConf.load(
        os.path.join(PHANTOM_ROOT, "configs", "scatterer_config.yaml"))

    mu_gt = torch.zeros((DEPTH, A_LINES), dtype=torch.float32, device=device)
    mu_gt[TISSUE_START:TISSUE_END, :] = MU_VAL
    alpha_gt = torch.ones_like(mu_gt)

    # Generate target via SS forward (paper convention)
    cfg.forward_model_system.forward_model_type = "ss"
    fwd_target = RasterScanOCTForward(cfg, device=device)
    fwd_target.gain_dB.fill_(100.0)
    with torch.no_grad():
        I_target = fwd_target(mu_gt, alpha_gt).cpu().numpy()

    noise_floor_dB = float(cfg.forward_model_system.noise_floor_dB)
    target_t = torch.tensor(I_target, dtype=torch.float32, device=device)
    nf_mask  = build_noise_floor_mask(target_t, noise_floor_dB, device)

    def _invert(fwd):
        mu = torch.nn.Parameter(torch.full_like(mu_gt, 5.0))
        alpha = torch.nn.Parameter(torch.ones_like(mu_gt), requires_grad=False)
        fwd.gain_dB.fill_(0.0)
        mu_best, _, _, _, _, _, _ = optimize(
            fwd, mu, alpha, target_t, opt_config,
            prune_threshold=float(opt_config.optimization.mu_min_mm_inv),
            device=device, out_dir="/tmp",
            noise_floor_mask=nf_mask,
            recompute_gain_mse_optimal_each_iter=True,
        )
        return mu_best.detach().cpu().numpy()

    cfg.forward_model_system.forward_model_type = "ss"
    mu_ss_2d = _invert(RasterScanOCTForward(cfg, device=device))
    cfg.forward_model_system.forward_model_type = "ms"
    cfg.forward_model_system.g                  = 0.7
    mu_ms_2d = _invert(RasterScanOCTForwardMS(cfg, device=device))

    np.savez(
        out_path,
        mu_ss_2d=mu_ss_2d.astype(np.float32),
        mu_ms_2d=mu_ms_2d.astype(np.float32),
        mu_ss_aline=mu_ss_2d[:, SHOWCASE_ALINE].astype(np.float32),
        mu_ms_aline=mu_ms_2d[:, SHOWCASE_ALINE].astype(np.float32),
        I_target=I_target.astype(np.float32),
        mu_val=np.float64(MU_VAL),
        tissue_start=np.int64(TISSUE_START),
        tissue_end=np.int64(TISSUE_END),
        aline=np.int64(SHOWCASE_ALINE),
    )
    print(f"  [Fig S3 (synthetic optim)] {out_path}")


# ---------------------------------------------------------------------------
# (3) real_phantom_panel_c.npz — SS + MS recovered μ for showcase real phantom
# ---------------------------------------------------------------------------

def _phantom_label(phantom):
    return f"IL_{phantom['concentration']}pct_rep{phantom['replicate']}"


def _ss_ms_out_dirs():
    """Locate per-model phantom_validation_ss_or_ms.py output directories."""
    ss = os.path.join(PHANTOM_ROOT, "outputs", "phantom_validation_ss")
    ms = os.path.join(PHANTOM_ROOT, "outputs", "phantom_validation_ms")
    return (ss if os.path.isdir(ss) else None,
            ms if os.path.isdir(ms) else None)


def build_real_phantom_panel_c(out_path, mat_dir):
    """Stitch SS + MS recovered μ maps for IL_5.0pct_rep2, B-scan 256."""
    input_dir = os.path.join(PHANTOM_ROOT, "inputs", "IntralipidPhantoms_20250206")
    phantoms = discover_despeckled_phantoms(input_dir)
    showcase = phantoms[SHOWCASE_PHANTOM_INDEX]
    label = _phantom_label(showcase)

    ss_dir, ms_dir = _ss_ms_out_dirs()
    mu_ss_path = os.path.join(ss_dir or "", label, f"bscan_{SHOWCASE_BSCAN:03d}",
                               "ss", "mu_final_mm.npy")
    mu_ms_path = os.path.join(ms_dir or "", label, f"bscan_{SHOWCASE_BSCAN:03d}",
                               "ms", "mu_final_mm.npy")
    if not (os.path.isfile(mu_ss_path) and os.path.isfile(mu_ms_path)):
        print(f"  [Fig S3 (c)] missing SS or MS μ for {label} bscan {SHOWCASE_BSCAN}; "
              f"run phantom_validation_ss_or_ms.py with MODEL_TYPE=ss and =ms first")
        return False

    mu_ss = np.load(mu_ss_path)
    mu_ms = np.load(mu_ms_path)
    intmask = load_intmask(SHOWCASE_PHANTOM_INDEX, SHOWCASE_BSCAN, mat_dir=mat_dir)

    np.savez(
        out_path,
        mu_ss_real=mu_ss.astype(np.float32),
        mu_ms_real=mu_ms.astype(np.float32),
        intmask=intmask.astype(bool),
        phantom=label,
        bscan=f"bscan_{SHOWCASE_BSCAN:03d}",
        aline=np.int64(SHOWCASE_ALINE),
        t_start=np.int64(133),
        t_end=np.int64(212),
    )
    print(f"  [Fig S3 (c)] {out_path}")
    return True


# ---------------------------------------------------------------------------
# (4) phantom_boxplot_data.npz — per-frame tissue-mean μ for SS + MS
# ---------------------------------------------------------------------------

def build_phantom_boxplot(out_path):
    """Merge per-pixel μ pools from outputs/phantom_validation_{ss,ms}/
    boxplot_pixel_data.npz into a single npz keyed by ss_<label> / ms_<label>.

    Matches the schema Figure 3 panel (l) uses, so Figure S3 panel (d) can plot
    wide whisker boxplots over the same per-pixel populations (compared SS vs MS
    instead of rep1 vs rep2).
    """
    ss_dir, ms_dir = _ss_ms_out_dirs()
    if ss_dir is None or ms_dir is None:
        print(f"  [Fig S3 (d)] need both SS and MS phantom_validation outputs; "
              f"got ss={ss_dir}, ms={ms_dir}")
        return False

    ss_npz = os.path.join(ss_dir, "boxplot_pixel_data.npz")
    ms_npz = os.path.join(ms_dir, "boxplot_pixel_data.npz")
    if not (os.path.isfile(ss_npz) and os.path.isfile(ms_npz)):
        print(f"  [Fig S3 (d)] missing source npz(s):\n"
              f"    ss: {ss_npz} ({'ok' if os.path.isfile(ss_npz) else 'missing'})\n"
              f"    ms: {ms_npz} ({'ok' if os.path.isfile(ms_npz) else 'missing'})\n"
              f"  Run phantom_validation_ss_or_ms.py with SUMMARIZE=1 first.")
        return False

    ss = np.load(ss_npz, allow_pickle=True)
    ms = np.load(ms_npz, allow_pickle=True)

    payload = dict(conc_order=np.array(CONC_ORDER, dtype="<U3"))
    if "concentrations" in ss.files:
        payload["concentrations"] = np.asarray(ss["concentrations"])

    n_phantoms = 0
    for key in ss.files:
        if not key.startswith("IL_"):
            continue
        ss_pool = ss[key]
        ms_pool = ms[key] if key in ms.files else np.empty(0, dtype=np.float32)
        payload[f"ss_{key}"] = ss_pool.astype(np.float32, copy=False)
        payload[f"ms_{key}"] = ms_pool.astype(np.float32, copy=False)
        print(f"  {key}: SS {ss_pool.size:>10,} px, MS {ms_pool.size:>10,} px")
        n_phantoms += 1

    np.savez(out_path, **payload)
    print(f"  [Fig S3 (d)] {out_path}  ({n_phantoms} phantoms × 2 models)")
    return True


# ---------------------------------------------------------------------------
# Stage to figureS5_ss_vs_ms/supp_ehf_data
# ---------------------------------------------------------------------------

def _stage_to_figure_paper_S3(out_base):
    fig_paper_S3 = os.path.join(PHANTOM_ROOT, "figures", "data",
                                 "figureS5_ss_vs_ms", "supp_ehf_data")
    os.makedirs(fig_paper_S3, exist_ok=True)
    files = ("forward_model_profiles.npz",
             "synthetic_optimization.npz",
             "real_phantom_panel_c.npz",
             "phantom_boxplot_data.npz")
    n = 0
    for f in files:
        src = os.path.join(out_base, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(fig_paper_S3, f))
            n += 1
    print(f"  [stage → figureS5_ss_vs_ms] {n}/{len(files)} files into {fig_paper_S3}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-synthetic", action="store_true",
                        help="skip the heavy synthetic_optimization.npz step")
    parser.add_argument("--stage-only", action="store_true",
                        help="just stage existing npz files into figureS5_ss_vs_ms/")
    args = parser.parse_args()
    stage_only = args.stage_only or int(os.environ.get("STAGE_ONLY", "0")) == 1
    do_synth   = not args.no_synthetic

    os.makedirs(OUT_BASE, exist_ok=True)
    if stage_only:
        _stage_to_figure_paper_S3(OUT_BASE)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    mat_dir = os.path.join(PHANTOM_ROOT, "matlab_CF_scripts_results")

    print("\n[1/4] forward_model_profiles.npz")
    build_forward_model_profiles(
        os.path.join(OUT_BASE, "forward_model_profiles.npz"), device)

    if do_synth:
        print("\n[2/4] synthetic_optimization.npz (heavy)")
        build_synthetic_optimization(
            os.path.join(OUT_BASE, "synthetic_optimization.npz"), device)
    else:
        print("\n[2/4] synthetic_optimization.npz — skipped (--no-synthetic)")

    print("\n[3/4] real_phantom_panel_c.npz")
    build_real_phantom_panel_c(
        os.path.join(OUT_BASE, "real_phantom_panel_c.npz"), mat_dir)

    print("\n[4/4] phantom_boxplot_data.npz")
    build_phantom_boxplot(
        os.path.join(OUT_BASE, "phantom_boxplot_data.npz"))

    print("\n[stage]")
    _stage_to_figure_paper_S3(OUT_BASE)


if __name__ == "__main__":
    main()
