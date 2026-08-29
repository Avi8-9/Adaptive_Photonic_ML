"""Generate numerical device-characterization datasets."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.hardware.device_models import (
    eye_diagram_signal,
    mzi_transfer_function,
    optical_spectrum,
    phase_modulation_curve,
    te0_te1_profiles,
)


ROOT = Path("results/device_simulation")


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


def generate_modes():

    axis, x, y, te0, te1 = (
        te0_te1_profiles()
    )

    output = (
        ROOT
        / "te_modes"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output / "axis.npy",
        axis,
    )

    np.save(
        output / "te0_field.npy",
        te0,
    )

    np.save(
        output / "te1_field.npy",
        te1,
    )

    rows = []

    center = len(axis) // 2

    for index in range(
        len(axis)
    ):

        rows.append(
            [
                axis[index],
                te0[
                    center,
                    index
                ],
                te1[
                    center,
                    index
                ],
            ]
        )

    save_csv(
        output / "mode_centerline.csv",
        [
            "coordinate",
            "TE0",
            "TE1",
        ],
        rows,
    )


def generate_mzi():

    phases = np.linspace(
        0.0,
        2.0 * np.pi,
        5001,
    )

    t0, t1 = (
        mzi_transfer_function(
            phases
        )
    )

    rows = zip(
        phases,
        t0,
        t1,
    )

    save_csv(
        ROOT
        / "mzi_interference"
        / "transfer_curve.csv",
        [
            "phase_rad",
            "port0_transmission",
            "port1_transmission",
        ],
        rows,
    )


def generate_spectrum():

    wavelength, response = (
        optical_spectrum()
    )

    rows = zip(
        wavelength,
        response,
    )

    save_csv(
        ROOT
        / "optical_spectrum"
        / "spectrum.csv",
        [
            "wavelength_nm",
            "normalized_power",
        ],
        rows,
    )


def generate_phase_modulation():

    command = np.linspace(
        0.0,
        1.0,
        2001,
    )

    response = (
        phase_modulation_curve(
            command
        )
    )

    rows = zip(
        command,
        response,
    )

    save_csv(
        ROOT
        / "phase_modulation"
        / "phase_response.csv",
        [
            "control_command",
            "phase_response",
        ],
        rows,
    )


def generate_eye():

    time, waveform = (
        eye_diagram_signal()
    )

    rows = zip(
        time,
        waveform,
    )

    save_csv(
        ROOT
        / "eye_diagram"
        / "waveform.csv",
        [
            "sample_index",
            "signal",
        ],
        rows,
    )


def main():

    generate_modes()
    generate_mzi()
    generate_spectrum()
    generate_phase_modulation()
    generate_eye()

    print(
        "Device simulation datasets generated under:",
        ROOT.resolve(),
    )


if __name__ == "__main__":
    main()
