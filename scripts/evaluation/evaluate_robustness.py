"""Evaluate a trained model across hardware perturbation grids."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.dataset_adapter import (
    load_prepared_dataset,
)

from src.models.model_factory import (
    build_model,
)

from src.measurement.born_rule import (
    born_probabilities,
)

from src.hardware.insertion_loss import (
    apply_insertion_loss,
)

from src.hardware.phase_noise import (
    apply_complex_phase_noise,
)


def evaluate_phase_noise(
    model,
    state,
    labels,
    values,
):

    rows = []

    clean_logits = model(
        state.real
    )

    clean_predictions = (
        clean_logits.argmax(
            dim=-1
        )
    )

    clean_accuracy = (
        clean_predictions == labels
    ).float().mean().item()

    for sigma in values:

        noisy_state = (
            apply_complex_phase_noise(
                state,
                sigma=sigma,
            )
        )

        probabilities = (
            born_probabilities(
                noisy_state
            )
        )

        # Reuse model readout when available.
        if hasattr(
            model,
            "readout"
        ):
            logits = model.readout(
                probabilities
            )
        else:
            logits = model(
                noisy_state.real
            )

        predictions = (
            logits.argmax(
                dim=-1
            )
        )

        accuracy = (
            predictions == labels
        ).float().mean().item()

        rows.append(
            {
                "sigma_phi_rad": sigma,
                "clean_accuracy": (
                    clean_accuracy
                ),
                "noisy_accuracy": (
                    accuracy
                ),
            }
        )

    return rows


def save_rows(
    rows,
    output,
):

    output = Path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fields = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)


if __name__ == "__main__":

    print(
        "Use this module from an experiment script "
        "with a loaded model and prepared test tensors."
    )
