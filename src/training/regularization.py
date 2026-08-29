"""Hardware-aware regularization utilities."""

from __future__ import annotations

import torch


def phase_gradient_penalty(
    loss: torch.Tensor,
    parameters,
) -> torch.Tensor:
    """Return squared gradient sensitivity of loss to phase parameters."""
    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    penalty = torch.zeros(
        (),
        dtype=loss.dtype,
        device=loss.device,
    )

    for gradient in gradients:
        if gradient is not None:
            penalty = penalty + torch.sum(gradient ** 2)

    return penalty


def l2_regularization(parameters) -> torch.Tensor:
    """Standard parameter L2 regularization."""
    total = None

    for parameter in parameters:
        term = torch.sum(parameter ** 2)

        if total is None:
            total = term
        else:
            total = total + term

    if total is None:
        return torch.tensor(0.0)

    return total
