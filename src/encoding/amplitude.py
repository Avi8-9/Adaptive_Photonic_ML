"""Learnable optical amplitude encoder."""

from __future__ import annotations

import torch
from torch import nn

from .normalization import l2_normalize_amplitude


class AmplitudeEncoder(nn.Module):
    """
    Learnable amplitude mapping from classical features to M optical modes.

    The network generates an unconstrained modal amplitude vector and then
    applies unit-power normalization.
    """

    def __init__(
        self,
        input_dim: int,
        num_modes: int = 32,
        hidden_dims: tuple[int, ...] = (128, 128),
        positive_output: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_modes = num_modes
        self.positive_output = positive_output

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
        raw_amplitude = self.network(x)

        if self.positive_output:
            raw_amplitude = torch.nn.functional.softplus(raw_amplitude)

        return l2_normalize_amplitude(raw_amplitude)
