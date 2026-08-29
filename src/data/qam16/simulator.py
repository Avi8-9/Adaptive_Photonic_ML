"""
Configurable simulated 16-QAM signal dataset.

IMPORTANT:
The manuscript identifies the 16-QAM benchmark as a simulated optical
communication impairment dataset, but does not fully specify a unique
channel-generation recipe. Therefore this module exposes the channel
assumptions explicitly instead of presenting them as experimentally
measured data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass
class QAM16Config:
    samples_per_class: int = 12500
    noise_std: float = 0.08
    phase_rotation_std: float = 0.03
    amplitude_jitter_std: float = 0.03
    seed: int = 42


def constellation():
    """Return normalized square 16-QAM constellation points."""
    levels = np.array(
        [-3.0, -1.0, 1.0, 3.0],
        dtype=np.float64,
    )

    points = []

    for i in levels:
        for q in levels:
            points.append(
                complex(i, q)
            )

    points = np.asarray(points)

    normalization = np.sqrt(
        np.mean(
            np.abs(points) ** 2
        )
    )

    return points / normalization


def generate_qam16(
    config: QAM16Config | None = None,
):
    """
    Generate a configurable four-class aggregate-label benchmark.

    Labels identify four coarse signal groups while the underlying
    sample contains the in-phase and quadrature components.

    This generator is intended for repository reproducibility and
    sensitivity studies; replace its channel model with the exact
    source-data generator when that source is available.
    """
    if config is None:
        config = QAM16Config()

    rng = np.random.default_rng(
        config.seed
    )

    constellation_points = constellation()

    total = (
        4 * config.samples_per_class
    )

    selected = rng.choice(
        len(constellation_points),
        size=total,
        replace=True,
    )

    clean = constellation_points[selected]

    phase_noise = rng.normal(
        0.0,
        config.phase_rotation_std,
        total,
    )

    amplitude_noise = rng.normal(
        0.0,
        config.amplitude_jitter_std,
        total,
    )

    noisy = (
        clean
        * (1.0 + amplitude_noise)
        * np.exp(1j * phase_noise)
    )

    additive = (
        rng.normal(
            0.0,
            config.noise_std,
            total,
        )
        + 1j
        * rng.normal(
            0.0,
            config.noise_std,
            total,
        )
    )

    received = noisy + additive

    # Two real channels: I and Q.
    features = np.stack(
        [
            received.real,
            received.imag,
        ],
        axis=1,
    )

    # Four coarse groups.
    labels = (
        np.arange(total)
        // config.samples_per_class
    )

    # Scale features for numerical stability.
    scale = np.max(
        np.abs(features),
        axis=0,
        keepdims=True,
    )

    features = features / np.maximum(
        scale,
        1e-8,
    )

    x = torch.tensor(
        features,
        dtype=torch.float32,
    )

    y = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return TensorDataset(x, y), x, y


def save_qam16(
    dataset: TensorDataset,
    features: torch.Tensor,
    labels: torch.Tensor,
    path,
):
    """Save simulated 16-QAM data."""
    path = str(path)

    torch.save(
        {
            "features": features,
            "labels": labels,
        },
        path,
    )
