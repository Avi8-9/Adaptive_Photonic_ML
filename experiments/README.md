# Experiment Runs

This directory contains experiment-specific outputs.

Each run should contain, where applicable:

    config.json
    seed.txt
    training_history.csv
    validation_history.csv
    predictions.csv
    metrics.json
    model.pt
    hardware_parameters.json

Recommended naming:

    <dataset>/<model>/<condition>/<seed>/

Example:

    mnist/proposed/clean/seed_0042/
    mnist/proposed/noisy/seed_0042/
    cifar10/proposed/phase_noise/seed_0042/
    qam16/proposed/insertion_loss/seed_0042/

This organization allows raw experiment outputs to remain
distinguishable from aggregated manuscript tables.
