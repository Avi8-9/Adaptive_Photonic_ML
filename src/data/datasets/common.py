"""Common dataset and tensor utilities."""

from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset


def set_global_seed(seed: int = 42):
    """Set reproducible Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_tensor_dataset(
    features: torch.Tensor,
    labels: torch.Tensor,
    path: str | Path,
):
    """Persist a tensor dataset to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "features": features.cpu(),
            "labels": labels.cpu(),
        },
        path,
    )


def load_tensor_dataset(
    path: str | Path,
) -> TensorDataset:
    """Load a tensor dataset previously saved by save_tensor_dataset."""
    payload = torch.load(
        path,
        map_location="cpu",
    )

    return TensorDataset(
        payload["features"],
        payload["labels"],
    )


def save_metadata(
    metadata: dict,
    path: str | Path,
):
    """Write machine-readable dataset metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
