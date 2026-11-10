"""Single-scattering model, alpha=1 and fixed per-A-line gain from CF."""
import math
import torch


def forward_db(mu, transport_mask, dz):
    active = torch.where(transport_mask, mu, torch.zeros_like(mu))
    cumulative = torch.cumsum(active * dz, dim=1)
    tau = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=1)
    local_mu = torch.where(transport_mask, mu, torch.ones_like(mu))
    return 10.0 * torch.log10(local_mu) - (20.0 / math.log(10.0)) * tau


def objective(mu, target, loss_mask, transport_mask, dz, gain, mse_weight=1000.0, tv_weight=1000.0):
    """Return independent frame objectives; callers sum them before backward."""
    counts = loss_mask.sum(dim=(1, 2)).clamp_min(1)
    f0 = forward_db(mu, transport_mask, dz)
    if gain.shape != (mu.shape[0], mu.shape[2]) or gain.requires_grad:
        raise ValueError("Gain must be fixed with shape (batch, A-line)")
    prediction = f0 + gain[:, None, :]
    mse = (((prediction - target) ** 2) * loss_mask).sum(dim=(1, 2)) / counts
    mz = transport_mask[:, 1:] & transport_mask[:, :-1]
    mx = transport_mask[:, :, 1:] & transport_mask[:, :, :-1]
    tvz = ((mu[:, 1:] - mu[:, :-1]).abs() * mz).sum(dim=(1, 2)) / mz.sum(dim=(1, 2)).clamp_min(1)
    tvx = ((mu[:, :, 1:] - mu[:, :, :-1]).abs() * mx).sum(dim=(1, 2)) / mx.sum(dim=(1, 2)).clamp_min(1)
    tv = tvz + tvx
    return mse_weight * mse + tv_weight * tv, mse, gain, tv
