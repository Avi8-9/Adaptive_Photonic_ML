# Contributing

Thank you for contributing to the Adaptive Photonic ML repository.

## Code Contributions

New code should:

- include clear documentation;
- preserve the existing package organization;
- use explicit and stable interfaces;
- include tests for important numerical behavior;
- avoid undocumented changes to scientific assumptions.

## Numerical Experiments

Changes that affect numerical results should document:

- dataset;
- model;
- configuration;
- random seed;
- hardware condition;
- software version;
- generating script.

## Data and Results

Externally sourced data should retain its original provenance.

Generated numerical simulation data should be clearly identified as
simulation output.

Synthetic reproducibility artifacts should not be represented as physical
measurements.

## Pull Requests

Please describe:

1. What changed.
2. Why it changed.
3. Which research component is affected.
4. How the change was validated.
5. Whether numerical results changed.
