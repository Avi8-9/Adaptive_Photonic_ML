# Results Directory

Results are separated into:

    runs/
    experiment_manifests/
    device_simulation/
    system_analysis/
    robustness/
    ablations/
    figures/
    metrics/
    tables/

`experiment_manifests/` contains configurations describing experiments.

`runs/` contains outputs from experiments that have actually been
executed.

`device_simulation/` contains numerical device models and their outputs.
These are simulation results, not fabricated measurements.

`system_analysis/` contains analytical power and latency calculations.

`figures/` contains data products used to generate manuscript figures.

A result must never be entered manually simply to reproduce a
manuscript percentage. Final reported values should be traceable to
an executable experiment, a documented analytical calculation, or
an explicitly labeled literature/reference value.
