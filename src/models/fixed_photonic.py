"""Fixed-encoding photonic baseline models."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..photonic.clements_mesh import ClementsMesh
from ..measurement.born_rule import born_probabilities
from ..measurement.readout import ClassificationReadout


class FixedAmplitudeEncoding(nn.Module):
    """
    Non-learnable amplitude projection.

    The projection matrix is initialized once and remains fixed.
    """

    def __init__(
        self,
        input_dim: int,
        num_modes: int,
        seed: int = 42,
    ):
        super().__init__()

        generator = torch.Generator()
        generator.manual_seed(seed)

        matrix = torch.randn(
            input_dim,
            num_modes,
            generator=generator,
        )

        matrix = matrix / (
            torch.linalg.matrix_norm(
                matrix,
                dim=0,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        self.register_buffer(
            "projection",
            matrix,
        )

    def forward(self, x):
        amplitude = torch.abs(
            x @ self.projection
        )

        norm = torch.linalg.vector_norm(
            amplitude,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        amplitude = amplitude / norm

        return amplitude


class FixedPhaseEncoding(nn.Module):
    """
    Non-learnable phase mapping.

    The same deterministic linear projection is used for every sample.
    """

    def __init__(
        self,
        input_dim: int,
        num_modes: int,
        seed: int = 42,
    ):
        super().__init__()

        generator = torch.Generator()
        generator.manual_seed(seed)

        matrix = torch.randn(
            input_dim,
            num_modes,
            generator=generator,
        )

        self.register_buffer(
            "projection",
            matrix,
        )

    def forward(self, x):
        raw = x @ self.projection

        return math.pi * torch.tanh(
            raw
        )


class FixedPhasePNN(nn.Module):
    """
    PNN with fixed phase representation and trainable photonic processor.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_modes: int = 32,
        seed: int = 42,
    ):
        super().__init__()

        self.amplitude = FixedAmplitudeEncoding(
            input_dim,
            num_modes,
            seed=seed,
        )

        self.phase = FixedPhaseEncoding(
            input_dim,
            num_modes,
            seed=seed,
        )

        self.processor = ClementsMesh(
            num_modes=num_modes,
        )

        self.readout = ClassificationReadout(
            num_modes,
            num_classes,
        )

    def forward(self, x):
        amplitude = self.amplitude(x)
        phase = self.phase(x)

        state = (
            amplitude.to(torch.complex64)
            * torch.exp(
                1j * phase.to(torch.complex64)
            )
        )

        state = self.processor(
            state
        )

        probabilities = born_probabilities(
            state
        )

        return self.readout(
            probabilities
        )


class FixedAmplitudePNN(nn.Module):
    """
    PNN with fixed amplitude representation and trainable phase.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_modes: int = 32,
        seed: int = 42,
    ):
        super().__init__()

        self.amplitude = FixedAmplitudeEncoding(
            input_dim,
            num_modes,
            seed=seed,
        )

        self.phase_parameter = nn.Parameter(
            torch.zeros(num_modes)
        )

        self.processor = ClementsMesh(
            num_modes=num_modes,
        )

        self.readout = ClassificationReadout(
            num_modes,
            num_classes,
        )

    def forward(self, x):
        amplitude = self.amplitude(x)

        phase = self.phase_parameter.unsqueeze(0)

        phase = phase.expand(
            x.shape[0],
            -1,
        )

        state = (
            amplitude.to(torch.complex64)
            * torch.exp(
                1j * phase.to(torch.complex64)
            )
        )

        state = self.processor(
            state
        )

        probabilities = born_probabilities(
            state
        )

        return self.readout(
            probabilities
        )


class RandomUnitaryPNN(nn.Module):
    """
    Fixed random optical representation and fixed optical processor.

    This acts as a non-adaptive photonic reference.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_modes: int = 32,
        seed: int = 42,
    ):
        super().__init__()

        generator = torch.Generator()
        generator.manual_seed(seed)

        projection = torch.randn(
            input_dim,
            num_modes,
            generator=generator,
        )

        self.register_buffer(
            "projection",
            projection,
        )

        random_complex = (
            torch.randn(
                num_modes,
                num_modes,
                generator=generator,
            )
            + 1j
            * torch.randn(
                num_modes,
                num_modes,
                generator=generator,
            )
        ).to(torch.complex64)

        q, r = torch.linalg.qr(
            random_complex
        )

        diagonal = torch.diagonal(r)

        phase = diagonal / (
            torch.abs(diagonal)
            + 1e-12
        )

        q = q * phase.conj()

        self.register_buffer(
            "unitary",
            q,
        )

        self.readout = ClassificationReadout(
            num_modes,
            num_classes,
        )

        # Random-unitary baseline has no adaptive photonic parameters.
        for parameter in self.readout.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        real_embedding = x @ self.projection

        amplitude = torch.abs(
            real_embedding
        )

        amplitude = amplitude / (
            torch.linalg.vector_norm(
                amplitude,
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        state = amplitude.to(
            torch.complex64
        )

        state = state @ self.unitary.T

        probabilities = born_probabilities(
            state
        )

        return self.readout(
            probabilities
        )
