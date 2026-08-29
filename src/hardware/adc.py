"""Finite-resolution ADC model."""

from __future__ import annotations

import torch


def quantize_adc(
    signal: torch.Tensor,
    bits: int = 8,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> torch.Tensor:
    """
    Uniform scalar quantization.

    Parameters
    ----------
    signal:
        Continuous input signal.
    bits:
        Number of ADC bits.
    min_value, max_value:
        Quantizer range.
    """
    if bits <= 0:
        raise ValueError("bits must be positive")

    clipped = torch.clamp(
        signal,
        min=min_value,
        max=max_value,
    )

    levels = 2**bits - 1

    normalized = (
        clipped - min_value
    ) / (
        max_value - min_value
    )

    quantized = torch.round(
        normalized * levels
    ) / levels

    return (
        quantized * (max_value - min_value)
        + min_value
    )
