"""Synthetic nonlinear Circles benchmark."""

from __future__ import annotations

from pathlib import Path

import torch
from sklearn.datasets import make_circles
from torch.utils.data import TensorDataset

from .common import save_metadata, save_tensor_dataset


def build_circles(
    n_samples: int = 6000,
    noise: float = 0.08,
    factor: float = 0.5,
    seed: int = 42,
):
    """
    Generate a two-class nonlinear concentric-circle dataset.

    The paper uses a 2-D Circles benchmark to evaluate nonlinear
    decision-boundary learning.
    """
    features, labels = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=seed,
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


def prepare_circles(
    output_dir: str | Path,
    n_samples: int = 6000,
    noise: float = 0.08,
    factor: float = 0.5,
    seed: int = 42,
):
    """Generate and save the Circles benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset, x, y = build_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        seed=seed,
    )

    save_tensor_dataset(
        x,
        y,
        output_dir / "circles_full.pt",
    )

    save_metadata(
        {
            "dataset": "Circles",
            "samples": int(len(y)),
            "input_dimension": 2,
            "classes": 2,
            "noise": noise,
            "factor": factor,
            "seed": seed,
            "status": "synthetically generated",
        },
        output_dir / "metadata.json",
    )

    return dataset
