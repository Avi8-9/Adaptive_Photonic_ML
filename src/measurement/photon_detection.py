"""Square-law photon-number detection."""

from __future__ import annotations

import torch


def photon_number_expectation(
    state: torch.Tensor,
) -> torch.Tensor:
    """
    Mean photon-number/intensity proxy for each output mode.

    For the complex coefficient representation used in the model,

        n_i = |c_i|^2.
    """
    return torch.abs(state) ** 2


def square_law_detection(
    state: torch.Tensor,
) -> torch.Tensor:
    """Alias for intensity-based square-law detection."""
    return photon_number_expectation(state)
