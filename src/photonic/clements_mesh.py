"""Programmable Clements-mesh photonic processor."""

from __future__ import annotations

import math

import torch
from torch import nn

from .mzi import mzi_matrix


class ClementsMesh(nn.Module):
    """
    Differentiable programmable M-mode photonic processor.

    The number of two-mode MZI blocks follows

        L_MZI = M(M-1)/2.

    The implementation uses trainable coupling and phase parameters.
    """

    def __init__(
        self,
        num_modes: int = 32,
        initialization_scale: float = 0.1,
    ):
        super().__init__()

        if num_modes < 2:
            raise ValueError("num_modes must be >= 2")

        self.num_modes = num_modes
        self.num_mzis = num_modes * (num_modes - 1) // 2

        theta = initialization_scale * torch.randn(self.num_mzis)
        phi = initialization_scale * torch.randn(self.num_mzis)

        self.theta = nn.Parameter(theta)
        self.phi = nn.Parameter(phi)

        pairs = []
        for layer in range(num_modes):
            start = layer % 2

            for i in range(start, num_modes - 1, 2):
                pairs.append((i, i + 1))

        if len(pairs) < self.num_mzis:
            extra = []
            for i in range(num_modes):
                for j in range(i + 1, num_modes):
                    extra.append((i, j))

            for pair in extra:
                if pair not in pairs:
                    pairs.append(pair)

        self.register_buffer(
            "pair_indices",
            torch.tensor(
                pairs[: self.num_mzis],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _identity(self, device, dtype):
        return torch.eye(
            self.num_modes,
            dtype=dtype,
            device=device,
        )

    def unitary_matrix(self) -> torch.Tensor:
        """
        Build the global unitary transformation.

        For reproducibility, the trainable MZI parameters are inserted
        sequentially into a dense M x M matrix.
        """
        U = self._identity(
            self.theta.device,
            torch.complex64,
        )

        for k, pair in enumerate(self.pair_indices.tolist()):
            i, j = pair

            local = mzi_matrix(
                self.theta[k],
                self.phi[k],
            )

            embedded = self._identity(
                self.theta.device,
                torch.complex64,
            )

            embedded[i, i] = local[0, 0]
            embedded[i, j] = local[0, 1]
            embedded[j, i] = local[1, 0]
            embedded[j, j] = local[1, 1]

            U = embedded @ U

        return U

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Propagate a batch of optical states."""
        U = self.unitary_matrix()

        return state @ U.T

    def unitary_error(self) -> torch.Tensor:
        """Return Frobenius norm of U^H U - I."""
        U = self.unitary_matrix()

        identity = torch.eye(
            self.num_modes,
            dtype=U.dtype,
            device=U.device,
        )

        error = U.conj().T @ U - identity

        return torch.linalg.matrix_norm(error)

    @property
    def expected_mzi_count(self) -> int:
        """Return Clements M(M-1)/2 element count."""
        return self.num_modes * (self.num_modes - 1) // 2
