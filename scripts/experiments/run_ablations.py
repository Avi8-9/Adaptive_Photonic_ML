"""Controlled ablation experiment definitions and runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ABLATION_MATRIX = [

    {
        "name": "complete_framework",
        "learnable_encoding": True,
        "physical_regularization": True,
        "loss_modeling": True,
    },

    {
        "name": "no_regularization",
        "learnable_encoding": True,
        "physical_regularization": False,
        "loss_modeling": True,
    },

    {
        "name": "no_loss_modeling",
        "learnable_encoding": True,
        "physical_regularization": True,
        "loss_modeling": False,
    },

    {
        "name": "no_learnable_encoding",
        "learnable_encoding": False,
        "physical_regularization": True,
        "loss_modeling": True,
    },
]


def save_ablation_matrix(
    output="results/metrics/ablation_configurations.csv",
):

    path = Path(output)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        fieldnames = [
            "name",
            "learnable_encoding",
            "physical_regularization",
            "loss_modeling",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in ABLATION_MATRIX:
            writer.writerow(row)


if __name__ == "__main__":
    save_ablation_matrix()
    print(
        "Ablation configuration matrix saved."
    )
