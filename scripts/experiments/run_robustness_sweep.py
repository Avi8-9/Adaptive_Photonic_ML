"""Generate controlled hardware-robustness experiment configurations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def write_phase_noise_grid(
    output: Path,
    values,
):
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(
            [
                "experiment_id",
                "sigma_phi_rad",
            ]
        )

        for index, value in enumerate(values):

            writer.writerow(
                [
                    f"phase_noise_{index:04d}",
                    value,
                ]
            )


def write_loss_grid(
    output: Path,
    values,
):
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(
            [
                "experiment_id",
                "loss_db_per_mode",
            ]
        )

        for index, value in enumerate(values):

            writer.writerow(
                [
                    f"loss_{index:04d}",
                    value,
                ]
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="results/robustness/grids",
    )

    args = parser.parse_args()

    output = Path(args.output)

    phase_values = [
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

    loss_values = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    write_phase_noise_grid(
        output / "phase_noise_grid.csv",
        phase_values,
    )

    write_loss_grid(
        output / "insertion_loss_grid.csv",
        loss_values,
    )

    print(
        f"Robustness grids saved in {output}"
    )


if __name__ == "__main__":
    main()
