# Reproducibility Guide

## Project

Adaptive Amplitude-Phase Encoding and Physically Constrained Optimization for High-Performance Photonic Machine Learning

## Repository

https://github.com/Avi8-9/Adaptive_Photonic_ML

## Scope

This repository contains numerical implementation, simulation, experiment
configuration, data-processing, analysis, and reproducibility resources for
the proposed adaptive amplitude-phase photonic neural processing framework.

## Default Optical Configuration

| Parameter | Default |
|---|---:|
| Optical modes | 32 |
| Clements-mesh MZI elements | 496 |
| Insertion-loss parameter | 1 dB/mode |
| Phase-noise parameter | 0.05 rad |

The insertion-loss and phase-noise quantities are configurable simulation
parameters.

## Training Configuration

- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 64
- Maximum epochs: 100
- Validation fraction: 10%
- Early-stopping infrastructure: enabled

## Benchmark Problems

- Circles
- MNIST
- Fashion-MNIST
- CIFAR-10
- Simulated 16-QAM

## Model Families

- Adaptive amplitude-phase PNN
- Fixed-phase PNN
- Fixed-amplitude PNN
- Random-unitary PNN
- Classical MLP
- Compact CNN infrastructure

## Hardware-Aware Analysis

The repository includes configurable models for:

- optical insertion loss,
- phase uncertainty,
- thermal phase drift,
- detector noise,
- ADC quantization.

## Ablation Studies

- Complete framework
- No physical regularization
- No insertion-loss modeling
- No learnable encoding

## Reproduction Workflow

1. Install the software dependencies.
2. Prepare or obtain the benchmark datasets.
3. Inspect the selected experiment configuration.
4. Fix the random seed.
5. Execute the selected experiment.
6. Save raw results and metadata.
7. Aggregate numerical metrics.
8. Generate figure-ready datasets.
9. Export manuscript-oriented tables.
10. Preserve provenance information.

## Provenance

Repository artifacts should remain distinguishable as:

- numerical simulation,
- analytical calculation,
- literature/configuration-derived parameter,
- synthetic reproducibility artifact,
- laboratory measurement.

Simulation and synthetic reproducibility outputs must not be described as direct
laboratory measurements.

## Directory Organization

```text
data/
src/
configs/
scripts/
experiments/
results/
figures/
tables/
reproducibility/
tests/
supplementary/
```
