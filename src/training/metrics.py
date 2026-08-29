"""Training and classification metrics."""

from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Classification accuracy."""
    predictions = torch.argmax(logits, dim=-1)

    return float(
        (predictions == labels).float().mean().item()
    )


@torch.no_grad()
def batch_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Return common batch-level metrics."""
    return {
        "accuracy": accuracy(logits, labels),
    }
