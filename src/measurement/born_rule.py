"""Born-rule probability mapping."""

from __future__ import annotations

import torch


def born_probabilities(
    state: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Convert modal amplitudes to a normalized probability vector.

        p_i = |c_i|^2 / sum_j |c_j|^2
    """
    intensity = torch.abs(state) ** 2
    normalization = torch.sum(
        intensity,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)

    return intensity / normalization


def probability_entropy(
    probabilities: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Shannon entropy of the output probability distribution."""
    p = probabilities.clamp_min(eps)

    return -torch.sum(
        p * torch.log(p),
        dim=-1,
    )
