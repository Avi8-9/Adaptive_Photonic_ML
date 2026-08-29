"""Feature normalization and standardization."""

from __future__ import annotations

import torch


class Standardizer:
    """
    Feature-wise standardization using training statistics only.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor):
        self.mean = x.mean(
            dim=0,
            keepdim=True,
        )

        self.std = x.std(
            dim=0,
            keepdim=True,
            unbiased=False,
        ).clamp_min(self.eps)

        return self

    def transform(self, x: torch.Tensor):
        if self.mean is None:
            raise RuntimeError(
                "Standardizer must be fitted first."
            )

        return (x - self.mean) / self.std

    def fit_transform(self, x: torch.Tensor):
        return self.fit(x).transform(x)
