"""Experiment-level classification metrics."""

from __future__ import annotations

import math

import torch


def classification_accuracy(
    predictions,
    labels,
):
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    return float(
        (
            predictions == labels
        ).float().mean().item()
    )


def confusion_matrix(
    predictions,
    labels,
    num_classes: int,
):
    matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int64,
    )

    for true, predicted in zip(
        labels.tolist(),
        predictions.tolist(),
    ):
        matrix[
            true,
            predicted,
        ] += 1

    return matrix


def macro_precision_recall_f1(
    predictions,
    labels,
    num_classes: int,
):
    cm = confusion_matrix(
        predictions,
        labels,
        num_classes,
    )

    precision_values = []
    recall_values = []
    f1_values = []

    for c in range(num_classes):

        tp = cm[c, c].float()

        fp = (
            cm[:, c].sum()
            - tp
        ).float()

        fn = (
            cm[c, :].sum()
            - tp
        ).float()

        precision = (
            tp / (tp + fp).clamp_min(1e-12)
        )

        recall = (
            tp / (tp + fn).clamp_min(1e-12)
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision + recall
            ).clamp_min(1e-12)
        )

        precision_values.append(
            precision.item()
        )

        recall_values.append(
            recall.item()
        )

        f1_values.append(
            f1.item()
        )

    return {
        "macro_precision": sum(
            precision_values
        ) / num_classes,

        "macro_recall": sum(
            recall_values
        ) / num_classes,

        "macro_f1": sum(
            f1_values
        ) / num_classes,
    }


def robustness_degradation(
    clean_accuracy: float,
    noisy_accuracy: float,
):
    """Relative accuracy decrease between clean and perturbed inference."""
    if clean_accuracy <= 0:
        return float("nan")

    return (
        (clean_accuracy - noisy_accuracy)
        / clean_accuracy
    )


def mean_confidence(
    probabilities,
):
    confidence = probabilities.max(
        dim=-1
    ).values

    return float(
        confidence.mean().item()
    )
