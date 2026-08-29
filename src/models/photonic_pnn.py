"""End-to-end adaptive photonic neural network."""

from __future__ import annotations

import torch
from torch import nn

from ..encoding.joint_amplitude_phase import (
    JointAmplitudePhaseEncoder,
)
from ..hardware.channel import HardwareModel
from ..measurement.born_rule import born_probabilities
from ..measurement.readout import ClassificationReadout
from ..photonic.clements_mesh import ClementsMesh


class AdaptivePhotonicPNN(nn.Module):
    """
    End-to-end photonic learning model.

    Pipeline

        classical input
            -> learnable amplitude/phase encoder
            -> Clements-mesh unitary transformation
            -> hardware perturbation model
            -> square-law/Born measurement
            -> classification readout.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_modes: int = 32,
        encoder_hidden_dims: tuple[int, ...] = (128, 128),
        insertion_loss_db: float = 1.0,
        phase_noise_sigma: float = 0.05,
        adc_bits: int | None = None,
        enable_hardware_noise: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_modes = num_modes

        self.encoder = JointAmplitudePhaseEncoder(
            input_dim=input_dim,
            num_modes=num_modes,
            hidden_dims=encoder_hidden_dims,
        )

        self.photonic_processor = ClementsMesh(
            num_modes=num_modes,
        )

        self.readout = ClassificationReadout(
            num_modes=num_modes,
            num_classes=num_classes,
        )

        self.hardware = HardwareModel(
            insertion_loss_db=insertion_loss_db,
            phase_noise_sigma=phase_noise_sigma,
            adc_bits=adc_bits,
            enable_detector_noise=enable_hardware_noise,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ):
        amplitude, phase, encoded_state = self.encoder(x)

        optical_state = self.photonic_processor(
            encoded_state
        )

        if self.hardware.enable_detector_noise:
            optical_state = self.hardware.optical_forward(
                optical_state
            )

        probabilities = born_probabilities(
            optical_state
        )

        logits = self.readout(probabilities)

        if not return_intermediate:
            return logits

        return {
            "amplitude": amplitude,
            "phase": phase,
            "encoded_state": encoded_state,
            "optical_state": optical_state,
            "probabilities": probabilities,
            "logits": logits,
        }

    def forward_with_hardware(
        self,
        x: torch.Tensor,
    ):
        """
        Explicit full hardware inference path.

        This path applies optical loss, phase noise, detector noise,
        and optional ADC quantization before classification.
        """
        amplitude, phase, encoded_state = self.encoder(x)

        ideal_output = self.photonic_processor(
            encoded_state
        )

        detected = self.hardware.detected_signal(
            ideal_output
        )

        probabilities = detected / (
            torch.sum(
                detected,
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        logits = self.readout(probabilities)

        return {
            "amplitude": amplitude,
            "phase": phase,
            "encoded_state": encoded_state,
            "ideal_output": ideal_output,
            "detected": detected,
            "probabilities": probabilities,
            "logits": logits,
        }

    def optical_power(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate total modal optical power."""
        return torch.sum(
            torch.abs(state) ** 2,
            dim=-1,
        )
