"""Numerical device-level photonic simulation models.

These are simulation models. They are not experimental measurements.
"""

from __future__ import annotations

import math

import numpy as np


def gaussian_mode(
    x,
    y,
    x0=0.0,
    y0=0.0,
    sigma_x=1.0,
    sigma_y=1.0,
):
    """Normalized 2-D Gaussian mode approximation."""
    return np.exp(
        -(
            ((x - x0) ** 2)
            / (2.0 * sigma_x**2)
            +
            ((y - y0) ** 2)
            / (2.0 * sigma_y**2)
        )
    )


def normalize_field(field):
    power = np.sum(
        np.abs(field) ** 2
    )

    return field / math.sqrt(
        max(power, 1e-15)
    )


def te0_te1_profiles(
    grid_size=256,
    spatial_extent=8.0,
):
    """
    Generate idealized TE0/TE1-like transverse field profiles.

    The output is suitable for visualization and numerical
    characterization, not as a substitute for full-wave FDTD.
    """
    axis = np.linspace(
        -spatial_extent,
        spatial_extent,
        grid_size,
    )

    x, y = np.meshgrid(
        axis,
        axis,
    )

    fundamental = gaussian_mode(
        x,
        y,
        sigma_x=1.1,
        sigma_y=0.7,
    )

    first_order = (
        x
        * gaussian_mode(
            x,
            y,
            sigma_x=1.3,
            sigma_y=0.8,
        )
    )

    te0 = normalize_field(
        fundamental
    )

    te1 = normalize_field(
        first_order
    )

    return axis, x, y, te0, te1


def mzi_transfer_function(
    phase,
    coupling_ratio=0.5,
):
    """
    Idealized two-port MZI intensity transfer.

    This is a compact transfer-function model for numerical studies.
    """
    k = np.clip(
        coupling_ratio,
        0.0,
        1.0,
    )

    visibility = (
        2.0
        * math.sqrt(k * (1.0 - k))
    )

    transmission_0 = (
        0.5
        * (
            1.0
            + visibility
            * np.cos(phase)
        )
    )

    transmission_1 = (
        1.0
        - transmission_0
    )

    return transmission_0, transmission_1


def optical_spectrum(
    wavelengths_nm=None,
    center_nm=1550.0,
    bandwidth_nm=30.0,
    insertion_loss_db=1.0,
):
    """Generate a smooth numerical spectral response model."""

    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(
            center_nm - 50.0,
            center_nm + 50.0,
            2001,
        )

    wavelengths_nm = np.asarray(
        wavelengths_nm
    )

    sigma = (
        bandwidth_nm
        / (
            2.0
            * np.sqrt(
                2.0 * np.log(2.0)
            )
        )
    )

    response = np.exp(
        -0.5
        * (
            (
                wavelengths_nm
                - center_nm
            )
            / sigma
        ) ** 2
    )

    loss = (
        10.0
        ** (
            -insertion_loss_db
            / 10.0
        )
    )

    return (
        wavelengths_nm,
        response * loss,
    )


def phase_modulation_curve(
    phase_command,
    phase_gain=1.0,
    offset=0.0,
    nonlinearity=0.0,
):
    """Thermo/electro-optic phase response model."""

    phase_command = np.asarray(
        phase_command
    )

    normalized = (
        phase_command
        - np.min(phase_command)
    ) / (
        np.ptp(phase_command)
        + 1e-12
    )

    response = (
        offset
        + phase_gain * normalized
        + nonlinearity
        * normalized**2
    )

    return response


def eye_diagram_signal(
    samples=10000,
    samples_per_symbol=32,
    amplitude=1.0,
    noise_std=0.03,
    jitter_std=0.02,
    seed=42,
):
    """
    Generate a numerical eye-diagram waveform.

    This is a signal-integrity simulation rather than an experimental
    oscilloscope trace.
    """
    rng = np.random.default_rng(
        seed
    )

    symbols = rng.choice(
        [-1.0, 1.0],
        size=max(
            2,
            samples
            // samples_per_symbol,
        ),
    )

    waveform = np.repeat(
        symbols,
        samples_per_symbol,
    )

    waveform = (
        amplitude
        * waveform
        + noise_std
        * rng.standard_normal(
            len(waveform)
        )
    )

    jitter = (
        jitter_std
        * rng.standard_normal(
            len(waveform)
        )
    )

    time = (
        np.arange(
            len(waveform)
        )
        + jitter
    )

    return time, waveform
