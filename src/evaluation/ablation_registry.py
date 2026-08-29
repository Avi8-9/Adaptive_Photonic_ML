"""Controlled ablation definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationDefinition:
    name: str
    lambda_physical: float
    include_loss_model: bool
    learnable_encoding: bool
    purpose: str


ABLATIONS = {
    "no_regularization": AblationDefinition(
        name="No physical regularization",
        lambda_physical=0.0,
        include_loss_model=True,
        learnable_encoding=True,
        purpose=(
            "Measure the contribution of the "
            "hardware-aware gradient penalty."
        ),
    ),

    "no_loss_modeling": AblationDefinition(
        name="No insertion-loss modeling",
        lambda_physical=1e-3,
        include_loss_model=False,
        learnable_encoding=True,
        purpose=(
            "Measure the contribution of explicitly "
            "training with optical attenuation."
        ),
    ),

    "no_learnable_encoding": AblationDefinition(
        name="No learnable encoding",
        lambda_physical=1e-3,
        include_loss_model=True,
        learnable_encoding=False,
        purpose=(
            "Measure the contribution of adaptive "
            "optical representation learning."
        ),
    ),
}


def get_ablation(name: str):
    if name not in ABLATIONS:
        raise KeyError(
            f"Unknown ablation: {name}"
        )

    return ABLATIONS[name]
