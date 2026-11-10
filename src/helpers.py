"""Shared helper functions for phantom test scripts.

Extracted from test/1_phantom_optimization.py and
test/3_single_phantom_slice_ROI_optimization.py during repo cleanup.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def discover_despeckled_phantoms(input_dir: str) -> list[dict]:
    """Discover DespeckledLogInt.tif phantoms, sorted by (concentration, replicate).

    Args:
        input_dir: Directory containing phantom TIFF files.

    Returns:
        List of dicts with keys: concentration, path, replicate, basename.
    """
    pattern = os.path.join(input_dir, "*_DespeckledLogInt.tif")
    files = sorted(glob.glob(pattern))

    phantoms = []
    for f in files:
        basename = os.path.basename(f)
        match = re.search(r'AgarInIL_(\d+)\.\.(\d+)', basename)
        if match:
            conc = float(f"{match.group(1)}.{match.group(2)}")
            same_conc = [p for p in phantoms if p['concentration'] == conc]
            replicate = len(same_conc) + 1
            phantoms.append({
                'concentration': conc,
                'path': f,
                'replicate': replicate,
                'basename': basename,
            })

    phantoms.sort(key=lambda x: (x['concentration'], x['replicate']))
    return phantoms


def load_phantom_bscan(tiff_path: str, bscan_index: int) -> np.ndarray:
    """Load a B-scan and return it in dB scale.

    Per Metadata.txt the raw 16-bit values encode dB intensities:
        dB = raw_16bit * (130 / 65535)
    with a range of 0-130 dB.

    Args:
        tiff_path: Path to the multi-frame 16-bit TIFF stack.
        bscan_index: Frame index within the TIFF stack.

    Returns:
        B-scan as float32 ndarray in dB scale, shape (500, 500).
    """
    from PIL import Image
    img = Image.open(tiff_path)
    img.seek(bscan_index)
    raw = np.array(img, dtype=np.float32)
    dB = raw * (130.0 / 65535.0)
    return dB


def save_results(
    mu_final: torch.Tensor,
    alpha_final: torch.Tensor,
    mu_init: torch.Tensor,
    alpha_init: torch.Tensor,
    I_target_dB: torch.Tensor,
    forward_model: nn.Module,
    metrics: dict,
    seq_steps: list,
    seq_imgs: list,
    out_dir: str,
    label: str,
    bscan_index: int,
    noise_floor_mask: torch.Tensor | None = None,
) -> None:
    """Save outputs. ``I_target_dB`` is in dB scale (0-130)."""
    os.makedirs(out_dir, exist_ok=True)
    eps = 1e-12
    mu_final_np = mu_final.detach().cpu().numpy()
    mu_init_np = mu_init.cpu().numpy()
    alpha_final_np = alpha_final.detach().cpu().numpy()
    alpha_init_np = alpha_init.cpu().numpy()
    D, A = mu_final_np.shape

    np.save(os.path.join(out_dir, "mu_final_mm.npy"), mu_final_np)
    np.save(os.path.join(out_dir, "alpha_final.npy"), alpha_final_np)
    np.savez(
        os.path.join(out_dir, "metrics.npz"),
        iters=np.array(metrics["iters"], dtype=np.int32),
        mse=np.array(metrics["mse"], dtype=np.float64),
        mu_mean_mm=np.array(metrics["mu_mean_mm"], dtype=np.float64),
        mu_std_mm=np.array(metrics["mu_std_mm"], dtype=np.float64),
        alpha_mean=np.array(metrics.get("alpha_mean", []), dtype=np.float64),
        alpha_std=np.array(metrics.get("alpha_std", []), dtype=np.float64),
        gain_dB=np.array(metrics.get("gain_dB", []), dtype=np.float64),
        lr=np.array(metrics.get("lr", []), dtype=np.float64),
        time_sec=np.array(metrics["time"], dtype=np.float64),
    )

    with torch.no_grad():
        I_pred_dB_tensor = forward_model(mu_final, alpha_final, noise_mask=noise_floor_mask)
        I_pred_dB_np = I_pred_dB_tensor.cpu().numpy()
        I_target_dB_np = I_target_dB.cpu().numpy()

        I_pred_linear = np.power(10.0, I_pred_dB_np / 10.0)
        I_target_linear = np.power(10.0, I_target_dB_np / 10.0)
        linear_max = np.max(I_target_linear) + eps
        pred_n = I_pred_linear / linear_max
        tgt_n = I_target_linear / linear_max

    delta_mu = mu_final_np - mu_init_np
    delta_alpha = alpha_final_np - alpha_init_np
    diff_linear = np.abs(pred_n - tgt_n)

    mse_arr = np.array(metrics["mse"])
    best_idx = int(np.argmin(mse_arr))
    best_mse = float(mse_arr[best_idx])
    best_iter = int(metrics["iters"][best_idx])
    tgt_dB_range = np.max(I_target_dB_np) - np.min(I_target_dB_np) + eps
    psnr_final = 10.0 * np.log10(tgt_dB_range ** 2 / max(best_mse, 1e-12))

    tgt_dB_min = np.min(I_target_dB_np)
    tgt_dB_max = np.max(I_target_dB_np)
    dB_range = tgt_dB_max - tgt_dB_min + eps
    pred_dB_norm = (I_pred_dB_np - tgt_dB_min) / dB_range
    tgt_dB_norm = (I_target_dB_np - tgt_dB_min) / dB_range
    diff_dB = np.abs(np.clip(pred_dB_norm, 0, 1) - tgt_dB_norm)

    # 4x3 panel figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    im1 = axes[0, 0].imshow(mu_init_np, cmap="viridis", aspect="auto", origin="upper")
    axes[0, 0].set_title("Initial mu (mm^-1)")
    axes[0, 0].set_xlabel(f"A-scan (0..{A-1})")
    axes[0, 0].set_ylabel("Depth (px)")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(mu_final_np, cmap="viridis", aspect="auto", origin="upper")
    axes[0, 1].set_title("Final mu (mm^-1)")
    axes[0, 1].set_xlabel(f"A-scan (0..{A-1})")
    axes[0, 1].set_ylabel("Depth (px)")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    vmax_mu = np.max(np.abs(delta_mu)) + eps
    im3 = axes[0, 2].imshow(
        delta_mu, cmap="coolwarm", aspect="auto", origin="upper",
        vmin=-vmax_mu, vmax=vmax_mu,
    )
    axes[0, 2].set_title("Delta mu (mm^-1)")
    axes[0, 2].set_xlabel(f"A-scan (0..{A-1})")
    axes[0, 2].set_ylabel("Depth (px)")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im4 = axes[1, 0].imshow(alpha_init_np, cmap="plasma", aspect="auto", origin="upper")
    axes[1, 0].set_title("Initial alpha (backscatter)")
    axes[1, 0].set_xlabel(f"A-scan (0..{A-1})")
    axes[1, 0].set_ylabel("Depth (px)")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im5 = axes[1, 1].imshow(alpha_final_np, cmap="plasma", aspect="auto", origin="upper")
    axes[1, 1].set_title("Final alpha (backscatter)")
    axes[1, 1].set_xlabel(f"A-scan (0..{A-1})")
    axes[1, 1].set_ylabel("Depth (px)")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    vmax_a = np.max(np.abs(delta_alpha)) + eps
    im6 = axes[1, 2].imshow(
        delta_alpha, cmap="coolwarm", aspect="auto", origin="upper",
        vmin=-vmax_a, vmax=vmax_a,
    )
    axes[1, 2].set_title("Delta alpha")
    axes[1, 2].set_xlabel(f"A-scan (0..{A-1})")
    axes[1, 2].set_ylabel("Depth (px)")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im7 = axes[2, 0].imshow(pred_dB_norm, cmap="gray", aspect="auto", origin="upper", vmin=0, vmax=1)
    axes[2, 0].set_title("Predicted B-scan (dB)")
    axes[2, 0].set_xlabel(f"A-scan (0..{A-1})")
    axes[2, 0].set_ylabel("Depth (px)")
    plt.colorbar(im7, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im8 = axes[2, 1].imshow(tgt_dB_norm, cmap="gray", aspect="auto", origin="upper", vmin=0, vmax=1)
    axes[2, 1].set_title("Target B-scan (dB)")
    axes[2, 1].set_xlabel(f"A-scan (0..{A-1})")
    axes[2, 1].set_ylabel("Depth (px)")
    plt.colorbar(im8, ax=axes[2, 1], fraction=0.046, pad=0.04)

    im9 = axes[2, 2].imshow(diff_dB, cmap="magma", aspect="auto", origin="upper", vmin=0, vmax=1)
    axes[2, 2].set_title("|Pred - Target| (norm dB)")
    axes[2, 2].set_xlabel(f"A-scan (0..{A-1})")
    axes[2, 2].set_ylabel("Depth (px)")
    plt.colorbar(im9, ax=axes[2, 2], fraction=0.046, pad=0.04)

    im10 = axes[3, 0].imshow(pred_n, cmap="gray", aspect="auto", origin="upper", vmin=0, vmax=1)
    axes[3, 0].set_title("Predicted B-scan (linear)")
    axes[3, 0].set_xlabel(f"A-scan (0..{A-1})")
    axes[3, 0].set_ylabel("Depth (px)")
    plt.colorbar(im10, ax=axes[3, 0], fraction=0.046, pad=0.04)

    im11 = axes[3, 1].imshow(tgt_n, cmap="gray", aspect="auto", origin="upper", vmin=0, vmax=1)
    axes[3, 1].set_title("Target B-scan (linear)")
    axes[3, 1].set_xlabel(f"A-scan (0..{A-1})")
    axes[3, 1].set_ylabel("Depth (px)")
    plt.colorbar(im11, ax=axes[3, 1], fraction=0.046, pad=0.04)

    im12 = axes[3, 2].imshow(diff_linear, cmap="magma", aspect="auto", origin="upper", vmin=0, vmax=1)
    axes[3, 2].set_title("|Pred - Target| (linear)")
    axes[3, 2].set_xlabel(f"A-scan (0..{A-1})")
    axes[3, 2].set_ylabel("Depth (px)")
    plt.colorbar(im12, ax=axes[3, 2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{label} | B-scan {bscan_index} | Best iter {best_iter} | "
        f"MSE={best_mse:.3e} | PSNR={psnr_final:.2f} dB"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_panel.png"), dpi=250)
    plt.close(fig)

    if len(seq_imgs) > 1:
        n_cols = len(seq_imgs)
        fig, axes_seq = plt.subplots(2, n_cols, figsize=(2.6 * n_cols + 0.8, 6))
        base_img = seq_imgs[0]
        max_diff = max(np.max(np.abs(img - base_img)) for img in seq_imgs) + eps
        for j in range(n_cols):
            axes_seq[0, j].imshow(
                seq_imgs[j], cmap="gray", aspect="auto", vmin=0.0, vmax=1.0, origin="upper",
            )
            axes_seq[0, j].set_title(f"Iter {seq_steps[j]}", fontsize=8)
            axes_seq[0, j].axis("off")
            axes_seq[1, j].imshow(
                np.abs(seq_imgs[j] - base_img), cmap="magma",
                aspect="auto", vmin=0.0, vmax=max_diff, origin="upper",
            )
            axes_seq[1, j].axis("off")
        fig.suptitle(f"B-scan {bscan_index} | Sequential predictions", fontsize=10)
        plt.savefig(os.path.join(out_dir, "sequential_bscans.png"), dpi=250)
        plt.close(fig)

    # Metrics figure (2x3)
    fig, axes_m = plt.subplots(2, 3, figsize=(18, 8))

    axes_m[0, 0].plot(metrics["iters"], metrics["mse"], "k-")
    axes_m[0, 0].set_xlabel("Iteration")
    axes_m[0, 0].set_ylabel("MSE")
    axes_m[0, 0].set_title("Loss")
    axes_m[0, 0].grid(True, alpha=0.3)

    axes_m[0, 1].plot(metrics["iters"], metrics["mu_mean_mm"], "b-", label="mu mean")
    axes_m[0, 1].fill_between(
        metrics["iters"],
        np.array(metrics["mu_mean_mm"]) - np.array(metrics["mu_std_mm"]),
        np.array(metrics["mu_mean_mm"]) + np.array(metrics["mu_std_mm"]),
        alpha=0.2, color="b",
    )
    axes_m[0, 1].set_xlabel("Iteration")
    axes_m[0, 1].set_ylabel("mu (mm^-1)")
    axes_m[0, 1].set_title("mu mean +/- std")
    axes_m[0, 1].grid(True, alpha=0.3)
    axes_m[0, 1].legend(fontsize=8)

    axes_m[0, 2].plot(metrics["iters"], metrics.get("alpha_mean", []), "m-", label="alpha mean")
    if metrics.get("alpha_std"):
        axes_m[0, 2].fill_between(
            metrics["iters"],
            np.array(metrics["alpha_mean"]) - np.array(metrics["alpha_std"]),
            np.array(metrics["alpha_mean"]) + np.array(metrics["alpha_std"]),
            alpha=0.2, color="m",
        )
    axes_m[0, 2].set_xlabel("Iteration")
    axes_m[0, 2].set_ylabel("alpha")
    axes_m[0, 2].set_title("alpha mean +/- std")
    axes_m[0, 2].grid(True, alpha=0.3)
    axes_m[0, 2].legend(fontsize=8)

    if metrics.get("gain_dB"):
        axes_m[1, 0].plot(metrics["iters"], metrics["gain_dB"], "c-")
        axes_m[1, 0].set_xlabel("Iteration")
        axes_m[1, 0].set_ylabel("gain_dB")
        axes_m[1, 0].set_title("System Gain")
        axes_m[1, 0].grid(True, alpha=0.3)

    if metrics.get("lr"):
        axes_m[1, 1].plot(metrics["iters"], metrics["lr"], "g-")
        axes_m[1, 1].set_xlabel("Iteration")
        axes_m[1, 1].set_ylabel("Learning Rate")
        axes_m[1, 1].set_title("LR Schedule")
        axes_m[1, 1].grid(True, alpha=0.3)

    axes_m[1, 2].plot(metrics["iters"], metrics["time"], color="orange")
    axes_m[1, 2].set_xlabel("Iteration")
    axes_m[1, 2].set_ylabel("Time (s)")
    axes_m[1, 2].set_title("Iter Time")
    axes_m[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "optimization_metrics.png"), dpi=250)
    plt.close(fig)

    with open(os.path.join(out_dir, "psnr.txt"), "w") as f:
        f.write(f"PSNR_dB={psnr_final:.6f}\nMSE={best_mse:.12e}\nbest_iter={best_iter}\n")
