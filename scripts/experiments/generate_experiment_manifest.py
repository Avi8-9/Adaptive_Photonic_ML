"""Generate a reproducible large-scale experiment manifest.

The manifest describes experiments that can subsequently be executed.
It does not fabricate scientific results.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path


DATASETS = {
    "circles": {
        "input_dim": 2,
        "classes": 2,
        "optical_modes": 32,
    },
    "mnist": {
        "input_dim": 784,
        "classes": 10,
        "optical_modes": 32,
    },
    "fashion_mnist": {
        "input_dim": 784,
        "classes": 10,
        "optical_modes": 32,
    },
    "cifar10": {
        "input_dim": 3072,
        "classes": 10,
        "optical_modes": 32,
    },
    "qam16": {
        "input_dim": 2,
        "classes": 4,
        "optical_modes": 32,
    },
}


MODELS = [
    "proposed",
    "fixed_phase",
    "fixed_amplitude",
    "random_unitary",
    "mlp",
]


CONDITIONS = [
    "clean",
    "hardware_noise",
]


SEEDS = [
    42,
    123,
    256,
    512,
    1024,
    2024,
    4096,
    8192,
]


LEARNING_RATES = [
    1e-3,
    5e-4,
    1e-4,
]


BATCH_SIZES = [
    32,
    64,
    128,
]


def build_manifest():

    manifest = []

    experiment_id = 0

    for dataset, dataset_info in DATASETS.items():

        for model in MODELS:

            for condition in CONDITIONS:

                for seed in SEEDS:

                    for learning_rate in LEARNING_RATES:

                        for batch_size in BATCH_SIZES:

                            experiment_id += 1

                            manifest.append(
                                {
                                    "experiment_id": (
                                        f"EXP-{experiment_id:07d}"
                                    ),
                                    "dataset": dataset,
                                    "model": model,
                                    "condition": condition,
                                    "seed": seed,
                                    "input_dim": dataset_info[
                                        "input_dim"
                                    ],
                                    "num_classes": dataset_info[
                                        "classes"
                                    ],
                                    "num_modes": dataset_info[
                                        "optical_modes"
                                    ],
                                    "learning_rate": learning_rate,
                                    "batch_size": batch_size,
                                    "epochs": 100,
                                    "validation_fraction": 0.10,
                                    "early_stopping_patience": 20,
                                    "insertion_loss_db_per_mode": (
                                        1.0
                                    ),
                                    "phase_noise_sigma_rad": (
                                        0.05
                                    ),
                                    "adc_bits": 8,
                                }
                            )

    return manifest


def save_manifest(
    manifest,
    output="results/experiment_manifests/full_matrix.json",
):

    path = Path(output)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():

    manifest = build_manifest()

    save_manifest(manifest)

    print(
        "Experiments in manifest:",
        len(manifest),
    )


if __name__ == "__main__":
    main()
