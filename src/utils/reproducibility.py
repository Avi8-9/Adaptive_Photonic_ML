"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
):
    """Set Python, NumPy and PyTorch random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def environment_summary() -> dict[str, str]:
    """Return basic runtime information."""
    return {
        "python": __import__("sys").version,
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": str(
            torch.cuda.is_available()
        ),
    }
