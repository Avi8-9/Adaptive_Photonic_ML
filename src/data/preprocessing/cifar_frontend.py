"""Hybrid electronic-photonic feature projection for CIFAR-10."""

from __future__ import annotations

import torch
from torch import nn


class CIFARHybridProjection(nn.Module):
    """
    Compact classical front-end that maps flattened CIFAR-10 input
    to the 32-dimensional optical-mode interface.

    This implements the architectural boundary described in the
    manuscript: a classical front-end projects the high-dimensional
    image into the photonic mode dimension.
    """

    def __init__(
        self,
        input_dim: int = 32 * 32 * 3,
        output_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)
