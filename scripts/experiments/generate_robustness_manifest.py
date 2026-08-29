"""Generate phase-noise and insertion-loss sweep manifests."""

from __future__ import annotations

import json
from pathlib import Path


PHASE_NOISE_VALUES = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10,
    0.12,
    0.15,
]


LOSS_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
]


ADC_BITS = [
    4,
    6,
    8,
    10,
    12,
    16,
]


DATASETS = [
    "mnist",
    "fashion_mnist",
    "cifar10",
    "qam16",
]


MODELS = [
    "proposed",
    "fixed_phase",
    "fixed_amplitude",
    "random_unitary",
]


SEEDS = [
    42,
    123,
    2024,
]


def generate():

    records = []

    experiment_id = 0

    # ----------------------------------------------------
    # Phase noise
    # ----------------------------------------------------

    for dataset in DATASETS:

        for model in MODELS:

            for seed in SEEDS:

                for sigma in PHASE_NOISE_VALUES:

                    experiment_id += 1

                    records.append(
                        {
                            "experiment_id": (
                                f"PN-{experiment_id:06d}"
                            ),
                            "experiment_type": "phase_noise",
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "sigma_phi_rad": sigma,
                        }
                    )

    # ----------------------------------------------------
    # Insertion loss
    # ----------------------------------------------------

    experiment_id = 0

    for dataset in DATASETS:

        for model in MODELS:

            for seed in SEEDS:

                for loss in LOSS_VALUES:

                    experiment_id += 1

                    records.append(
                        {
                            "experiment_id": (
                                f"IL-{experiment_id:06d}"
                            ),
                            "experiment_type": (
                                "insertion_loss"
                            ),
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "loss_db_per_mode": loss,
                        }
                    )

    # ----------------------------------------------------
    # ADC resolution
    # ----------------------------------------------------

    experiment_id = 0

    for dataset in DATASETS:

        for model in MODELS:

            for seed in SEEDS:

                for bits in ADC_BITS:

                    experiment_id += 1

                    records.append(
                        {
                            "experiment_id": (
                                f"ADC-{experiment_id:06d}"
                            ),
                            "experiment_type": (
                                "adc_quantization"
                            ),
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "adc_bits": bits,
                        }
                    )

    return records


def save(
    records,
    output="results/experiment_manifests/robustness_matrix.json",
):

    path = Path(output)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            records,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Robustness experiments:",
        len(records),
    )


if __name__ == "__main__":
    save(generate())
