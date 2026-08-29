"""Prepare data for the manuscript learning-dynamics figure."""

    from __future__ import annotations

import csv
from pathlib import Path


def load_history(path):

    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def create_figure5_dataset(
    history_path,
    output_path=(
        "results/figures/"
        "figure_05_learning_dynamics/"
        "learning_dynamics.csv"
    ),
):

    history = load_history(
        history_path
    )

    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    ]

    rows = []

    for row in history:

        rows.append(
            {
                field: row.get(
                    field,
                    "",
                )
                for field in fields
            }
        )

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
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
        "Provide an actual training-history CSV "
        "to generate Figure 5 data."
    )
