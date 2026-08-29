"""Run the controlled model/dataset experiment matrix."""

from __future__ import annotations

from pathlib import Path
import csv
import json

from scripts.experiments.train_model import train


EXPERIMENTS = {

    "mnist": {
        "input_dim": 784,
        "classes": 10,
    },

    "fashion_mnist": {
        "input_dim": 784,
        "classes": 10,
    },

    "cifar10": {
        "input_dim": 3072,
        "classes": 10,
    },

    "circles": {
        "input_dim": 2,
        "classes": 2,
    },

    "qam16": {
        "input_dim": 2,
        "classes": 4,
    },
}


MODELS = [
    "proposed",
    "fixed_phase",
    "fixed_amplitude",
    "random_unitary",
    "mlp",
]


SEEDS = [
    42,
    123,
    2024,
]


def run_all():

    summaries = []

    for dataset_name, info in (
        EXPERIMENTS.items()
    ):

        for model_name in MODELS:

            for seed in SEEDS:

                try:

                    summary = train(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        input_dim=info[
                            "input_dim"
                        ],
                        num_classes=info[
                            "classes"
                        ],
                        seed=seed,
                        epochs=100,
                        batch_size=64,
                        learning_rate=1e-3,
                    )

                    summaries.append(
                        summary
                    )

                except Exception as exc:

                    print(
                        f"FAILED: "
                        f"{dataset_name} / "
                        f"{model_name} / "
                        f"seed={seed}"
                    )

                    summaries.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "seed": seed,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

    output = Path(
        "results/metrics"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output
        / "full_experiment_matrix.json"
    )

    json_path.write_text(
        json.dumps(
            summaries,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = (
        output
        / "full_experiment_matrix.csv"
    )

    keys = sorted(
        {
            key
            for row in summaries
            for key in row
        }
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=keys,
        )

        writer.writeheader()

        for row in summaries:
            writer.writerow(row)

    print(
        "Complete experiment matrix finished."
    )


if __name__ == "__main__":
    run_all()
