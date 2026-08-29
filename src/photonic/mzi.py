"""Mach-Zehnder interferometer primitives."""

from __future__ import annotations

import torch


def beam_splitter(theta: torch.Tensor) -> torch.Tensor:
    """
    Real two-mode rotation used for the programmable MZI coupling stage.

        B(theta) =
            [[cos(theta), -sin(theta)],
             [sin(theta),  cos(theta)]]
    """
    c = torch.cos(theta)
    s = torch.sin(theta)

    matrix = torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ],
        dim=-2,
    )

    return matrix.to(torch.complex64)


def phase_shifter(phi: torch.Tensor) -> torch.Tensor:
    """Single-mode complex phase operation."""
    diagonal = torch.exp(1j * phi)

    if diagonal.ndim == 0:
        return diagonal

    return diagonal


def mzi_matrix(
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """
    Construct the two-mode MZI transformation

        U_MZI = P(phi) B(theta).

    This is unitary for real-valued theta and phi.
    """
    B = beam_splitter(theta)

    phase = torch.exp(1j * phi).to(torch.complex64)
    P = torch.diag_embed(
        torch.stack(
            [
                phase,
                torch.ones_like(phase),
            ],
            dim=-1,
        )
    )

    return P @ B


def is_unitary(
    matrix: torch.Tensor,
    atol: float = 1e-5,
) -> bool:
    """Numerically verify U^H U = I."""
    m = matrix.shape[-1]
    identity = torch.eye(
        m,
        dtype=matrix.dtype,
        device=matrix.device,
    )

    product = matrix.conj().transpose(-2, -1) @ matrix

    return bool(torch.allclose(product, identity, atol=atol))
