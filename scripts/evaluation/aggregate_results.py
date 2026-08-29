"""Aggregate experiment summaries into manuscript-oriented tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def load_summaries(
    root="results/experiments",
):

    rows = []

    root = Path(root)

    if not root.exists():
        return rows

    for summary_path in root.rglob(
        "summary.json"
    ):

        try:
            rows.append(
                json.loads(
                    summary_path.read_text(
                        encoding="utf-8"
                    )
                )
            )

        except json.JSONDecodeError:
            continue

    return rows


def aggregate(
    rows,
):

    grouped = defaultdict(list)

    for row in rows:

        if "test_accuracy" not in row:
            continue

        key = (
            row["dataset"],
            row["model"],
        )

        grouped[key].append(
            row["test_accuracy"]
        )

    output = []

    for (
        dataset,
        model,
    ), values in grouped.items():

        mean = (
            sum(values)
            / len(values)
        )

        if len(values) > 1:

            variance = sum(
                (
                    value - mean
                ) ** 2
                for value in values
            ) / (
                len(values) - 1
            )

            std = (
                variance ** 0.5
            )

        else:
            std = 0.0

        output.append(
            {
                "dataset": dataset,
                "model": model,
                "n_runs": len(values),
                "accuracy_mean": mean,
                "accuracy_std": std,
            }
        )

    return output


def save(
    rows,
    path="results/metrics/aggregated_results.csv",
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "dataset",
        "model",
        "n_runs",
        "accuracy_mean",
        "accuracy_std",
    ]

    with path.open(
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

    summaries = load_summaries()

    aggregated = aggregate(
        summaries
    )

    save(aggregated)

    print(
        f"Aggregated {len(aggregated)} "
        "dataset/model combinations."
    )
