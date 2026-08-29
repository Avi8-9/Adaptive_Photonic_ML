"""Combined hardware perturbation pipeline."""

from __future__ import annotations

import torch

from .adc import quantize_adc
from .detector_noise import apply_detector_noise
from .insertion_loss import apply_insertion_loss
from .phase_noise import apply_complex_phase_noise
from ..measurement.photon_detection import square_law_detection


class HardwareModel:
    """
    Differentiable/simulation-oriented hardware impairment wrapper.

    Parameters
    ----------
    insertion_loss_db:
        Per-mode insertion-loss value.
    phase_noise_sigma:
        Standard deviation of phase perturbation.
    adc_bits:
        Receiver quantization resolution.
    enable_detector_noise:
        Whether shot/thermal noise is applied at detection.
    """

    def __init__(
        self,
        insertion_loss_db: float = 1.0,
        phase_noise_sigma: float = 0.05,
        adc_bits: int | None = 8,
        enable_detector_noise: bool = True,
    ):
        self.insertion_loss_db = insertion_loss_db
        self.phase_noise_sigma = phase_noise_sigma
        self.adc_bits = adc_bits
        self.enable_detector_noise = enable_detector_noise

    def optical_forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Apply optical impairments to a complex modal state."""
        state = apply_insertion_loss(
            state,
            self.insertion_loss_db,
        )

        state = apply_complex_phase_noise(
            state,
            self.phase_noise_sigma,
        )

        return state

    def detected_signal(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Convert impaired optical state to detected intensity."""
        state = self.optical_forward(state)

        intensity = square_law_detection(state)

        if self.enable_detector_noise:
            intensity = apply_detector_noise(intensity)

        if self.adc_bits is not None:
            intensity = quantize_adc(
                intensity,
                bits=self.adc_bits,
            )

        return intensity
