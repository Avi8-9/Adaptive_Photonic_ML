"""Unitary-domain utilities."""

from __future__ import annotations

import torch


def project_to_unitary(
    matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Polar/SVD projection of a square complex matrix onto the
    nearest unitary matrix in Frobenius norm.
    """
    U, _, Vh = torch.linalg.svd(matrix)
    return U @ Vh


def unitary_constraint_penalty(
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute ||U^H U - I||_F^2."""
    m = matrix.shape[-1]

    identity = torch.eye(
        m,
        dtype=matrix.dtype,
        device=matrix.device,
    )

    error = matrix.conj().transpose(-2, -1) @ matrix - identity

    return torch.sum(torch.abs(error) ** 2)
