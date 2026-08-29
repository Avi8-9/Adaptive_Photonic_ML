"""Classification and physical loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Standard multi-class cross entropy."""
    return F.cross_entropy(logits, labels)


def physical_sensitivity_penalty(
    phase_parameters: torch.Tensor,
    loss_function: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize sensitivity with respect to trainable optical phase.

    The scalar loss is differentiated with respect to the phase
    parameters and the squared gradient norm is accumulated.
    """
    gradient = torch.autograd.grad(
        loss_function,
        phase_parameters,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]

    return torch.sum(gradient ** 2)


def combined_objective(
    classification_loss: torch.Tensor,
    physical_penalty: torch.Tensor,
    lambda_physical: float = 1e-3,
) -> torch.Tensor:
    """J = L_CE + lambda * L_phys."""
    return (
        classification_loss
        + lambda_physical * physical_penalty
    )
