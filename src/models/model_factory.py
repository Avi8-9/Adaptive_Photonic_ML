"""Centralized model construction."""

from __future__ import annotations

from .classical_baselines import (
    ClassicalMLP,
    CompactCNN,
    TwoChannelSignalMLP,
)

from .fixed_photonic import (
    FixedAmplitudePNN,
    FixedPhasePNN,
    RandomUnitaryPNN,
)

from .photonic_pnn import AdaptivePhotonicPNN


def build_model(
    name: str,
    input_dim: int,
    num_classes: int,
    num_modes: int = 32,
    seed: int = 42,
):
    """
    Construct a model by registry name.
    """

    normalized = name.lower().strip()

    if normalized == "proposed":
        return AdaptivePhotonicPNN(
            input_dim=input_dim,
            num_classes=num_classes,
            num_modes=num_modes,
        )

    if normalized == "fixed_phase":
        return FixedPhasePNN(
            input_dim=input_dim,
            num_classes=num_classes,
            num_modes=num_modes,
            seed=seed,
        )

    if normalized == "fixed_amplitude":
        return FixedAmplitudePNN(
            input_dim=input_dim,
            num_classes=num_classes,
            num_modes=num_modes,
            seed=seed,
        )

    if normalized == "random_unitary":
        return RandomUnitaryPNN(
            input_dim=input_dim,
            num_classes=num_classes,
            num_modes=num_modes,
            seed=seed,
        )

    if normalized == "mlp":
        return ClassicalMLP(
            input_dim=input_dim,
            num_classes=num_classes,
        )

    if normalized == "signal_mlp":
        return TwoChannelSignalMLP(
            input_dim=input_dim,
            num_classes=num_classes,
        )

    raise ValueError(
        f"Unknown model: {name}"
    )
