import math
from typing import Optional

import torch
import torch.nn as nn


class RasterScanOCTForward(nn.Module):
    """Raster-scan OCT forward model that outputs directly in dB space.

    The linear Beer-Lambert model is:
        I(z) = G · α(z) · μ(z) · exp(-2 · cumsum_shifted(μ·Δz)) · C(z)

    Taking 10·log10(·) converts this to:
        I_dB(z) = gain_dB + 10·log10(α(z)) + 10·log10(μ(z))
                  + 10·log10(C(z)) − K · cumsum_shifted(μ·Δz) 

    where:
        gain_dB   = 10·log10(G), a frozen scalar system-gain offset
        α(z)      = per-pixel backscattering fraction (learnable, same shape as μ)
        K         = 20·log10(e) = 20/ln(10) ≈ 8.6859

    Both α(z) and μ(z) are optimised externally; gain_dB is auto-initialised
    once and then frozen.
    """

    def __init__(self, config, device='cuda'):
        super().__init__()
        self.device = device

        self.depth_samples = int(config.forward_model_system.depth_samples)
        self.a_lines = int(config.forward_model_system.a_lines)
        self.focal_plane_px = float(config.forward_model_system.focal_plane_position_pixels)
        self.rayleigh_range_px = float(config.forward_model_system.rayleigh_range_pixels)
        self.confocal_correction_on = bool(config.forward_model_system.confocal_correction)
        self.axial_pixel_size_mm = float(config.forward_model_system.axial_pixel_size_mm)
        self.lateral_pixel_size_mm = float(config.forward_model_system.lateral_pixel_size_mm)

        z_px = torch.arange(self.depth_samples, dtype=torch.float32, device=device)

        # --- confocal function (linear and dB) ---
        if self.confocal_correction_on:
            confocal = 1.0 / (((z_px - self.focal_plane_px) / self.rayleigh_range_px) ** 2 + 1.0)
        else:
            confocal = torch.ones(self.depth_samples, dtype=torch.float32, device=device)
        self.register_buffer('confocal', confocal)
        # confocal ∈ (0, 1] 
        self.register_buffer('confocal_dB', 10.0 * torch.log10(confocal))

        # --- system gain in dB (frozen scalar, auto-initialised externally) ---
        self.register_buffer('gain_dB',
                             torch.tensor(0.0, dtype=torch.float32, device=device))

        # --- K = 20·log10(e) = 20/ln(10): converts cumulative OD to dB ---
        self.register_buffer('_K',
                             torch.tensor(20.0 / math.log(10.0),
                                          dtype=torch.float32, device=device))

        self.register_buffer('_zeros_row',
                             torch.zeros(1, self.a_lines, dtype=torch.float32, device=device))

        print(f"RasterScanOCTForward [dB mode]: {self.depth_samples}x{self.a_lines}, "
              f"axial_px={self.axial_pixel_size_mm}mm, lateral_px={self.lateral_pixel_size_mm}mm, "
              f"confocal={'on' if self.confocal_correction_on else 'off'}, "
              f"K={20.0 / math.log(10.0):.4f}")

    def forward(self, mu, alpha, noise_mask=None):
        """Compute predicted OCT B-scan directly in dB scale.

        I_dB(z) = gain_dB + 10·log10(α(z)) + 10·log10(μ(z))
                  + confocal_dB − K · cumsum_shifted(μ·Δz)

        Args:
            mu: Attenuation tensor (depth_samples × a_lines), in mm⁻¹.
            alpha: Backscattering fraction tensor (depth_samples × a_lines).
            noise_mask: Optional binary mask (1 = tissue, 0 = noise floor).
                Noise-floor scatterers are excluded from the computation.

        Returns:
            I_dB: Predicted B-scan in dB scale (same domain as target).
        """
        eps = 1e-12
        mu_active = mu * noise_mask if noise_mask is not None else mu
        alpha_active = alpha * noise_mask if noise_mask is not None else alpha

        # Cumulative optical depth from surface to (but not including) z.
        cumulative = torch.cumsum(mu_active * self.axial_pixel_size_mm, dim=0)
        shifted = torch.cat([self._zeros_row, cumulative[:-1]], dim=0)

        I_dB = (self.gain_dB
                + 10.0 * torch.log10(alpha_active + eps)
                + 10.0 * torch.log10(mu_active + eps)
                + self.confocal_dB.unsqueeze(1)
                - self._K * shifted)

        return I_dB


class RasterScanOCTForwardMS(nn.Module):
    """Raster-scan OCT forward model with Extended Huygens-Fresnel (EHF) multiple scattering.

    Decomposes the OCT transport into three terms (Thrane et al. 2000, Faber et al. 2004):

        T_total(z) = T_SS(z) + T_cross(z) + T_MS(z)

    where:
        T_SS    = exp(-2*tau)                                         [ballistic photons]
        T_cross = 4*exp(-tau)*(1-exp(-tau)) / (1 + wS^2/wH^2)        [single-multi interference]
        T_MS    = (1-exp(-tau))^2 * wH^2 / wS^2                      [multiply scattered photons]

    The dB output becomes:
        I_dB(z) = gain_dB + 10*log10(alpha) + 10*log10(mu)
                  + confocal_dB + 10*log10(T_total)

    When T_cross = T_MS = 0 (pure single scattering at tau=0):
        10*log10(T_SS) = 10*log10(exp(-2*tau)) = -K*tau
    recovering the single-scatter model exactly.

    Note: g=1.0 gives T_total=1 (no attenuation) — physically correct for purely
    forward scattering. To get SS behavior, use forward_model_type='ss' instead.

    Args:
        config: OmegaConf config with forward_model_system parameters.
        device: Torch device ('cuda' or 'cpu').
    """

    def __init__(self, config, device: str = 'cuda') -> None:
        super().__init__()
        self.device = device

        # --- shared parameters (identical to SS model) ---
        self.depth_samples = int(config.forward_model_system.depth_samples)
        self.a_lines = int(config.forward_model_system.a_lines)
        self.focal_plane_px = float(config.forward_model_system.focal_plane_position_pixels)
        self.rayleigh_range_px = float(config.forward_model_system.rayleigh_range_pixels)
        self.confocal_correction_on = bool(config.forward_model_system.confocal_correction)
        self.axial_pixel_size_mm = float(config.forward_model_system.axial_pixel_size_mm)
        self.lateral_pixel_size_mm = float(config.forward_model_system.lateral_pixel_size_mm)

        # --- EHF-specific parameters ---
        g = float(getattr(config.forward_model_system, 'g', 0.7))
        self.n_sample = float(getattr(config.forward_model_system, 'n_sample', 1.35))

        # theta_rms^2 = 2*(1-g) from Henyey-Greenstein phase function
        self._theta_rms_sq = 2.0 * (1.0 - g)
        self.register_buffer('_theta_rms_sq_t',
                             torch.tensor(self._theta_rms_sq, dtype=torch.float32, device=device))

        # Beam waist at focus: w0 = lambda / (pi * NA)
        lambda_mm = float(config.forward_model_system.central_wavelength_nm) * 1e-6  # nm -> mm
        NA = float(config.forward_model_system.numerical_aperture)
        w0_mm = lambda_mm / (math.pi * NA)
        w0_sq = w0_mm ** 2

        z_px = torch.arange(self.depth_samples, dtype=torch.float32, device=device)

        # --- confocal function (identical to SS model) ---
        if self.confocal_correction_on:
            confocal = 1.0 / (((z_px - self.focal_plane_px) / self.rayleigh_range_px) ** 2 + 1.0)
        else:
            confocal = torch.ones(self.depth_samples, dtype=torch.float32, device=device)
        self.register_buffer('confocal', confocal)
        self.register_buffer('confocal_dB', 10.0 * torch.log10(confocal))

        # --- beam waist squared wH^2(z) [mm^2] ---
        # Uses the same Lorentzian shape as the confocal: ((z-zFP)/zR)^2 + 1
        wH_sq = w0_sq * (((z_px - self.focal_plane_px) / self.rayleigh_range_px) ** 2 + 1.0)
        self.register_buffer('_wH_sq', wH_sq)  # [D]

        # --- depth positions in mm for scatter broadening ---
        z_mm = z_px * self.axial_pixel_size_mm
        z_over_n_sq = (z_mm / self.n_sample).pow(2)
        self.register_buffer('_z_over_n_sq', z_over_n_sq)  # [D]

        # --- system gain in dB (frozen scalar, auto-initialised externally) ---
        self.register_buffer('gain_dB',
                             torch.tensor(0.0, dtype=torch.float32, device=device))

        # K = 20/ln(10) — converts natural-log OD to dB
        self.register_buffer('_K',
                             torch.tensor(20.0 / math.log(10.0),
                                          dtype=torch.float32, device=device))

        self.register_buffer('_zeros_row',
                             torch.zeros(1, self.a_lines, dtype=torch.float32, device=device))

        print(f"RasterScanOCTForwardMS [EHF dB mode]: {self.depth_samples}x{self.a_lines}, "
              f"g={g:.3f}, n={self.n_sample:.3f}, w0={w0_mm*1e3:.2f}um, "
              f"confocal={'on' if self.confocal_correction_on else 'off'}")

    def forward(self, mu: torch.Tensor, alpha: torch.Tensor,
                noise_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute predicted OCT B-scan with EHF multiple scattering in dB.

        I_dB(z) = gain_dB + 10*log10(alpha) + 10*log10(mu)
                  + confocal_dB + 10*log10(T_total)

        Args:
            mu: Attenuation tensor (depth_samples x a_lines), in mm^-1.
            alpha: Backscattering fraction tensor (depth_samples x a_lines).
            noise_mask: Optional binary mask (1 = tissue, 0 = noise floor).

        Returns:
            I_dB: Predicted B-scan in dB scale.
        """
        eps = 1e-12
        mu_active = mu * noise_mask if noise_mask is not None else mu
        alpha_active = alpha * noise_mask if noise_mask is not None else alpha

        # 1. Cumulative optical depth (identical to SS model)
        cumulative = torch.cumsum(mu_active * self.axial_pixel_size_mm, dim=0)
        tau = torch.cat([self._zeros_row, cumulative[:-1]], dim=0)  # [D, A]

        # 2. Scatter-broadened beam waist
        #    wS^2 = wH^2 + (1/3) * tau * theta_rms^2 * (z_mm / n)^2
        wH_sq_2d = self._wH_sq.unsqueeze(1)                         # [D, 1]
        wS_sq = wH_sq_2d + (1.0 / 3.0) * tau * self._theta_rms_sq * self._z_over_n_sq.unsqueeze(1)

        # 3. Three EHF transport terms
        exp_neg_tau = torch.exp(-tau)
        T_SS = exp_neg_tau * exp_neg_tau                             # exp(-2*tau)
        T_cross = 4.0 * exp_neg_tau * (1.0 - exp_neg_tau) / (1.0 + wS_sq / (wH_sq_2d + eps))
        T_MS = (1.0 - exp_neg_tau).pow(2) * wH_sq_2d / (wS_sq + eps)
        T_total = T_SS + T_cross + T_MS

        # 4. Assemble dB output
        I_dB = (self.gain_dB
                + 10.0 * torch.log10(alpha_active + eps)
                + 10.0 * torch.log10(mu_active + eps)
                + self.confocal_dB.unsqueeze(1)
                + 10.0 * torch.log10(T_total + eps))
        return I_dB


def create_forward_model(config, device: str = 'cuda') -> nn.Module:
    """Factory function to create the appropriate forward model.

    Args:
        config: OmegaConf config with forward_model_system parameters.
        device: Torch device ('cuda' or 'cpu').

    Returns:
        Forward model instance (RasterScanOCTForward or RasterScanOCTForwardMS).

    Raises:
        ValueError: If forward_model_type is not 'ss' or 'ms'.
    """
    model_type = str(getattr(config.forward_model_system, 'forward_model_type', 'ss'))
    if model_type == 'ms':
        return RasterScanOCTForwardMS(config, device=device)
    elif model_type == 'ss':
        return RasterScanOCTForward(config, device=device)
    else:
        raise ValueError(f"Unknown forward_model_type: {model_type!r}. Use 'ss' or 'ms'.")
