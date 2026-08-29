"""Optical amplitude normalization utilities."""

from __future__ import annotations

import torch


def l2_normalize_amplitude(
    amplitude: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Normalize real-valued modal amplitudes so that

        sum_m |a_m|^2 = 1.

    Parameters
    ----------
    amplitude:
        Tensor whose final dimension indexes optical modes.
    eps:
        Numerical stabilization constant.

    Returns
    -------
    torch.Tensor
        Unit-power normalized amplitudes.
    """
    power = torch.sum(torch.abs(amplitude) ** 2, dim=-1, keepdim=True)
    denominator = torch.sqrt(power.clamp_min(eps))
    return amplitude / denominator


def normalize_complex_state(
    state: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize a complex optical state along its mode dimension."""
    norm_sq = torch.sum(torch.abs(state) ** 2, dim=-1, keepdim=True)
    norm = torch.sqrt(norm_sq.clamp_min(eps))
    return state / norm


def check_unit_power(
    state: torch.Tensor,
    tolerance: float = 1e-5,
) -> bool:
    """Check whether the modal state satisfies unit total power."""
    power = torch.sum(torch.abs(state) ** 2, dim=-1)
    target = torch.ones_like(power)
    return bool(torch.allclose(power, target, atol=tolerance, rtol=tolerance))
