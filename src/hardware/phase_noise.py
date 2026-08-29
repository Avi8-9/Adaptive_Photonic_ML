"""Stochastic optical phase perturbation."""

from __future__ import annotations

import torch


def sample_phase_noise(
    shape,
    sigma: float = 0.05,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Sample zero-mean Gaussian phase perturbations."""
    return sigma * torch.randn(
        shape,
        device=device,
        dtype=dtype,
    )


def perturb_phase(
    phase: torch.Tensor,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Add Gaussian phase noise."""
    noise = sample_phase_noise(
        phase.shape,
        sigma=sigma,
        device=phase.device,
        dtype=phase.dtype,
    )

    return phase + noise


def apply_complex_phase_noise(
    state: torch.Tensor,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Apply independent phase perturbation to complex modal fields."""
    noise = sample_phase_noise(
        state.shape,
        sigma=sigma,
        device=state.device,
        dtype=torch.float32,
    )

    noise = noise.to(state.real.dtype)

    return state * torch.exp(
        1j * noise.to(state.dtype)
    )
