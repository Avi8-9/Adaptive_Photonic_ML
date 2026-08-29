"""Optical insertion-loss model."""

from __future__ import annotations

import math

import torch


def db_to_amplitude(
    loss_db: float | torch.Tensor,
) -> torch.Tensor:
    """
    Convert dB loss to field-amplitude transmission.

    Power transmission is 10^(-L/10).
    Field amplitude transmission is 10^(-L/20).
    """
    loss = torch.as_tensor(loss_db, dtype=torch.float32)

    return torch.pow(
        torch.tensor(10.0),
        -loss / 20.0,
    )


def db_to_power_transmission(
    loss_db: float | torch.Tensor,
) -> torch.Tensor:
    """Convert dB loss into power transmission factor."""
    loss = torch.as_tensor(loss_db, dtype=torch.float32)

    return torch.pow(
        torch.tensor(10.0),
        -loss / 10.0,
    )


def apply_insertion_loss(
    state: torch.Tensor,
    loss_db_per_mode: float = 1.0,
) -> torch.Tensor:
    """Apply a uniform modal amplitude attenuation."""
    factor = db_to_amplitude(loss_db_per_mode).to(
        state.device
    )

    return state * factor.to(state.dtype)
