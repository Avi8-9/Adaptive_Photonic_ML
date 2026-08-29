"""Reproducible dataset splitting."""

from __future__ import annotations

import torch


def train_validation_split(
    features: torch.Tensor,
    labels: torch.Tensor,
    validation_fraction: float = 0.10,
    seed: int = 42,
):
    """
    Split an existing training partition into train and validation.

    The test partition is intentionally handled separately.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(
        len(features),
        generator=generator,
    )

    validation_size = int(
        round(len(features) * validation_fraction)
    )

    validation_indices = permutation[
        :validation_size
    ]

    train_indices = permutation[
        validation_size:
    ]

    return (
        features[train_indices],
        labels[train_indices],
        features[validation_indices],
        labels[validation_indices],
    )
