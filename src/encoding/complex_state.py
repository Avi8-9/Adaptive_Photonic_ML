"""Complex optical state construction."""

from __future__ import annotations

import torch


def amplitude_phase_to_complex(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Convert amplitude-phase representation into complex modal coefficients."""
    amplitude = amplitude.to(torch.complex64)
    phase = phase.to(torch.complex64)

    return amplitude * torch.exp(1j * phase)


def complex_to_amplitude_phase(
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover amplitude and phase representation."""
    amplitude = torch.abs(state)
    phase = torch.angle(state)
    return amplitude, phase
