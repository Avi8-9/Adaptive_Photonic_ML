"""Differentiable classification readout."""

from __future__ import annotations

import torch
from torch import nn


class ClassificationReadout(nn.Module):
    """
    Maps M optical-mode probabilities to C class logits.

    This layer is intentionally classical/differentiable and represents
    the decision stage following optical measurement.
    """

    def __init__(
        self,
        num_modes: int,
        num_classes: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(num_modes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(probabilities)


def logits_to_probabilities(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Convert class logits to class probabilities."""
    return torch.softmax(logits, dim=-1)
