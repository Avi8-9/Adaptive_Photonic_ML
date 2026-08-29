"""Generate system-level analytical performance datasets."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.hardware.system_accounting import (
    LatencyBudget,
    PowerBudget,
    optical_propagation_latency,
    tops_per_watt,
)


OUTPUT = Path(
    "results/system_analysis"
)


def save_csv(
    path,
    headers,
    rows,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(headers)

        writer.writerows(rows)


def power_sweep():

    frequencies = np.linspace(
        5.0,
        50.0,
        19,
    )

    rows = []

    for frequency in frequencies:

        power = PowerBudget(
            laser=35.0,
            modulator=22.0,
            dac=30.0,
            pd_tia=18.0,
            adc=35.0,
            digital=15.0,
            thermal=8.0,
            memory=10.0,
            core=5.0,
        )

        power_value = (
            power.total
            * (
                1.0
                + 0.015
                * max(
                    frequency - 28.0,
                    0.0
                )
            )
        )

        throughput = (
            0.75
            * frequency
        )

        efficiency = (
            tops_per_watt(
                throughput,
                power_value,
            )
        )

        rows.append(
            [
                frequency,
                power_value,
                throughput,
                efficiency,
            ]
        )

    save_csv(
        OUTPUT
        / "power_throughput_sweep.csv",
        [
            "frequency_GHz",
            "system_power_mW",
            "throughput_TOPS",
            "energy_efficiency_TOPS_W",
        ],
        rows,
    )


def latency_budget():

    budget = LatencyBudget(
        load=4.0,
        preprocess=8.0,
        dac=12.0,
        modulation=5.0,
        optical=0.6,
        pd_tia=14.0,
        adc=20.0,
        postprocess=30.0,
        interface=19.6,
    )

    rows = [
        ["load", budget.load],
        ["preprocess", budget.preprocess],
        ["dac", budget.dac],
        ["modulation", budget.modulation],
        ["optical", budget.optical],
        ["pd_tia", budget.pd_tia],
        ["adc", budget.adc],
        ["postprocess", budget.postprocess],
        ["interface", budget.interface],
        ["TOTAL", budget.total],
    ]

    save_csv(
        OUTPUT
        / "latency_budget.csv",
        [
            "component",
            "latency_ns",
        ],
        rows,
    )


def propagation_latency():

    lengths = np.linspace(
        1e-3,
        0.2,
        500,
    )

    rows = []

    for length in lengths:

        latency = (
            optical_propagation_latency(
                length_m=length,
                group_index=4.0,
            )
        )

        rows.append(
            [
                length,
                latency * 1e9,
            ]
        )

    save_csv(
        OUTPUT
        / "optical_propagation_latency.csv",
        [
            "length_m",
            "latency_ns",
        ],
        rows,
    )


def main():

    power_sweep()
    latency_budget()
    propagation_latency()

    print(
        "System-level analytical data generated under:",
        OUTPUT.resolve(),
    )


if __name__ == "__main__":
    main()
