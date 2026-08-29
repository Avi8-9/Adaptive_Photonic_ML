"""Optical field propagation utilities."""

from __future__ import annotations

import torch


def propagate(
    state: torch.Tensor,
    unitary: torch.Tensor,
) -> torch.Tensor:
    """Apply a global unitary optical transformation."""
    return state @ unitary.T


def modal_power(
    state: torch.Tensor,
) -> torch.Tensor:
    """Return modal optical power |c_m|^2."""
    return torch.abs(state) ** 2


def total_power(
    state: torch.Tensor,
) -> torch.Tensor:
    """Return total modal optical power."""
    return torch.sum(modal_power(state), dim=-1)
