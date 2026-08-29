"""Photodetection noise models."""

from __future__ import annotations

import torch


def shot_noise(
    intensity: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Poisson-like shot-noise approximation using sqrt(intensity).

    The scale is explicitly configurable for simulation studies.
    """
    standard_deviation = torch.sqrt(
        torch.clamp(intensity, min=0.0)
    ) * scale

    return standard_deviation * torch.randn_like(intensity)


def thermal_noise(
    reference: torch.Tensor,
    standard_deviation: float = 1e-3,
) -> torch.Tensor:
    """Additive Gaussian receiver thermal noise."""
    return standard_deviation * torch.randn_like(reference)


def apply_detector_noise(
    intensity: torch.Tensor,
    shot_scale: float = 1.0,
    thermal_std: float = 1e-3,
) -> torch.Tensor:
    """Apply combined shot and thermal noise."""
    noisy = (
        intensity
        + shot_noise(intensity, shot_scale)
        + thermal_noise(intensity, thermal_std)
    )

    return torch.clamp(noisy, min=0.0)
