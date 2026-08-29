"""Thermal phase-drift model."""

from __future__ import annotations

import torch


def thermal_phase_shift(
    temperature_c: torch.Tensor | float,
    reference_temperature_c: float = 25.0,
    thermo_optic_coefficient: float = 1.0e-2,
) -> torch.Tensor:
    """
    Linearized temperature-dependent phase shift.

    This is a configurable simulation model rather than a claim of
    experimentally measured device behavior.
    """
    temperature = torch.as_tensor(
        temperature_c,
        dtype=torch.float32,
    )

    delta_temperature = (
        temperature - reference_temperature_c
    )

    return thermo_optic_coefficient * delta_temperature


def apply_thermal_drift(
    phase: torch.Tensor,
    temperature_c: torch.Tensor | float,
    reference_temperature_c: float = 25.0,
    thermo_optic_coefficient: float = 1.0e-2,
) -> torch.Tensor:
    """Perturb phase values according to temperature."""
    drift = thermal_phase_shift(
        temperature_c,
        reference_temperature_c,
        thermo_optic_coefficient,
    )

    return phase + drift
