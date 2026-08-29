# Experiment Engine

The experiment layer is organized around the manuscript's controlled
comparisons and hardware-aware evaluation.

## Core model families

- proposed adaptive amplitude-phase PNN
- fixed-phase PNN
- fixed-amplitude PNN
- random-unitary PNN
- classical MLP
- signal-domain MLP

## Default training configuration

- optimizer: Adam
- batch size: 64
- maximum epochs: 100
- validation fraction: 10%
- early stopping patience: 20
- random seed: 42

## Output organization

Each independent run is stored under:

    results/experiments/<dataset>/<model>/seed_<XXXX>/

The directory contains:

    best_model.pt
    records.json
    records.csv
    summary.json

Aggregated results are written under:

    results/metrics/

including:

    full_experiment_matrix.csv
    aggregated_results.csv
    adaptive_photonic_ml_results.xlsx
