"""Aggregate actual robustness-sweep outputs."""

    from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def load_rows(path):

    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def mean(values):

    values = [
        float(value)
        for value in values
    ]

    return (
        sum(values)
        / len(values)
        if values
        else float("nan")
    )


def aggregate_by_parameter(
    rows,
    parameter,
    accuracy_field="noisy_accuracy",
):

    grouped = defaultdict(list)

    for row in rows:

        grouped[
            float(
                row[parameter]
            )
        ].append(
            float(
                row[accuracy_field]
            )
        )

    output = []

    for parameter_value in sorted(
        grouped
    ):

        values = grouped[
            parameter_value
        ]

        output.append(
            {
                parameter: parameter_value,
                "mean_accuracy": mean(
                    values
                ),
                "n": len(values),
            }
        )

    return output


if __name__ == "__main__":

    print(
        "This module aggregates measured outputs "
        "from actual robustness experiments."
    )
