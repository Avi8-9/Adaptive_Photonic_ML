"""Learnable optical phase encoder."""

from __future__ import annotations

import math

import torch
from torch import nn


class PhaseEncoder(nn.Module):
    """
    Learnable mapping of classical features to bounded optical phases.

    A sigmoid mapping constrains the phase to [0, 2*pi].
    """

    def __init__(
        self,
        input_dim: int,
        num_modes: int = 32,
        hidden_dims: tuple[int, ...] = (128, 128),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_modes = num_modes

        layers = []
        previous = input_dim

        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.GELU(),
                ]
            )
            previous = width

        layers.append(nn.Linear(previous, num_modes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_phase = self.network(x)
        bounded = torch.sigmoid(raw_phase)
        return 2.0 * math.pi * bounded
