"""System-level optical/electronic power and latency accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PowerBudget:

    laser: float = 0.0
    modulator: float = 0.0
    dac: float = 0.0
    pd_tia: float = 0.0
    adc: float = 0.0
    digital: float = 0.0
    thermal: float = 0.0
    memory: float = 0.0
    core: float = 0.0

    @property
    def total(self):
        return (
            self.laser
            + self.modulator
            + self.dac
            + self.pd_tia
            + self.adc
            + self.digital
            + self.thermal
            + self.memory
            + self.core
        )

    @property
    def core_power(self):
        return self.core


@dataclass
class LatencyBudget:

    load: float = 0.0
    preprocess: float = 0.0
    dac: float = 0.0
    modulation: float = 0.0
    optical: float = 0.0
    pd_tia: float = 0.0
    adc: float = 0.0
    postprocess: float = 0.0
    interface: float = 0.0

    @property
    def total(self):
        return (
            self.load
            + self.preprocess
            + self.dac
            + self.modulation
            + self.optical
            + self.pd_tia
            + self.adc
            + self.postprocess
            + self.interface
        )


def tops_per_watt(
    throughput_tops: float,
    power_mw: float,
):
    """Compute TOPS/W from TOPS and mW."""
    if power_mw <= 0:
        raise ValueError(
            "Power must be positive."
        )

    # TOPS / (mW * 1e-3 W/mW)
    return throughput_tops / (
        power_mw * 1e-3
    )


def optical_propagation_latency(
    length_m: float,
    group_index: float = 4.0,
    c: float = 299792458.0,
):
    """Compute n_g L / c."""
    return (
        group_index
        * length_m
        / c
    )
