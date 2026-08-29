"""Joint learnable amplitude-phase optical encoder."""

from __future__ import annotations

import torch
from torch import nn

from .amplitude import AmplitudeEncoder
from .phase import PhaseEncoder


class JointAmplitudePhaseEncoder(nn.Module):
    """
    Jointly produces normalized amplitude and bounded phase vectors.

    The encoded complex modal coefficient is

        c_m = a_m exp(j phi_m).

    The resulting state obeys unit total modal power.
    """

    def __init__(
        self,
        input_dim: int,
        num_modes: int = 32,
        hidden_dims: tuple[int, ...] = (128, 128),
    ):
        super().__init__()

        self.amplitude_encoder = AmplitudeEncoder(
            input_dim=input_dim,
            num_modes=num_modes,
            hidden_dims=hidden_dims,
        )

        self.phase_encoder = PhaseEncoder(
            input_dim=input_dim,
            num_modes=num_modes,
            hidden_dims=hidden_dims,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        amplitude = self.amplitude_encoder(x)
        phase = self.phase_encoder(x)

        state = amplitude.to(torch.complex64) * torch.exp(
            1j * phase.to(torch.complex64)
        )

        return amplitude, phase, state
