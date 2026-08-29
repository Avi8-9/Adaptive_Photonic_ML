"""Experiment file I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_json(
    data: dict,
    path: str | Path,
):
    """Save a dictionary as readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    path: str | Path,
    extra: dict | None = None,
):
    """Save model and optimizer state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    map_location="cpu",
):
    """Load a previously saved checkpoint."""
    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint
