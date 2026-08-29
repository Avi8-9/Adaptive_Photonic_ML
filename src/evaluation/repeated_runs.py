"""Repeated-run statistical analysis."""

from __future__ import annotations

import math
from collections import defaultdict


def mean(values):
    if not values:
        return float("nan")

    return sum(values) / len(values)


def sample_std(values):

    if len(values) < 2:
        return 0.0

    average = mean(values)

    variance = sum(
        (x - average) ** 2
        for x in values
    ) / (len(values) - 1)

    return math.sqrt(variance)


def confidence_interval_95(values):

    if len(values) < 2:
        average = mean(values)

        return {
            "mean": average,
            "std": 0.0,
            "lower": average,
            "upper": average,
        }

    average = mean(values)
    std = sample_std(values)

    standard_error = (
        std / math.sqrt(len(values))
    )

    margin = (
        1.96
        * standard_error
    )

    return {
        "mean": average,
        "std": std,
        "lower": average - margin,
        "upper": average + margin,
    }


def aggregate_accuracy(
    rows,
    group_fields=(
        "dataset",
        "model",
    ),
    metric_field="test_accuracy",
):

    groups = defaultdict(list)

    for row in rows:

        key = tuple(
            row[field]
            for field in group_fields
        )

        value = row.get(
            metric_field
        )

        if value is None:
            continue

        groups[key].append(
            float(value)
        )

    results = []

    for key, values in groups.items():

        statistics = (
            confidence_interval_95(
                values
            )
        )

        output = {
            field: key[index]
            for index, field
            in enumerate(group_fields)
        }

        output.update(
            {
                "n_runs": len(values),
                "mean": statistics["mean"],
                "std": statistics["std"],
                "ci95_lower": statistics[
                    "lower"
                ],
                "ci95_upper": statistics[
                    "upper"
                ],
            }
        )

        results.append(
            output
        )

    return results
