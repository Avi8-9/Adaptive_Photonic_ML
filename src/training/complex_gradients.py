"""Complex-valued gradient utilities."""

from __future__ import annotations

import torch


def complex_gradient(
    output: torch.Tensor,
    variable: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a gradient for a real-valued scalar objective with
    respect to a complex-valued variable.

    PyTorch handles complex reverse-mode differentiation directly;
    this wrapper provides a consistent interface for the photonic code.
    """
    if not output.is_complex():
        return torch.autograd.grad(
            output,
            variable,
            create_graph=True,
            retain_graph=True,
        )[0]

    real_objective = output.real

    return torch.autograd.grad(
        real_objective,
        variable,
        create_graph=True,
        retain_graph=True,
    )[0]


def complex_magnitude_squared(
    z: torch.Tensor,
) -> torch.Tensor:
    """Return |z|^2."""
    return z.real ** 2 + z.imag ** 2
