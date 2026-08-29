"""Baseline registry for controlled comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineDefinition:
    name: str
    category: str
    trainable_encoder: bool
    trainable_processor: bool
    description: str


BASELINES = {
    "fixed_phase": BaselineDefinition(
        name="Fixed phase encoding PNN",
        category="photonic",
        trainable_encoder=False,
        trainable_processor=True,
        description=(
            "Phase mapping held fixed while the photonic "
            "processor remains trainable."
        ),
    ),

    "fixed_amplitude": BaselineDefinition(
        name="Fixed amplitude encoding PNN",
        category="photonic",
        trainable_encoder=False,
        trainable_processor=True,
        description=(
            "Amplitude mapping held fixed while the photonic "
            "processor remains trainable."
        ),
    ),

    "random_unitary": BaselineDefinition(
        name="Random-unitary PNN",
        category="photonic",
        trainable_encoder=False,
        trainable_processor=False,
        description=(
            "Fixed random optical transformation used as a "
            "controlled non-adaptive reference."
        ),
    ),
}


def get_baseline(name: str):
    if name not in BASELINES:
        raise KeyError(
            f"Unknown baseline: {name}"
        )

    return BASELINES[name]
