# Adaptive Amplitude-Phase Encoding and Physically Constrained Optimization for High-Performance Photonic Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Code%20%26%20Simulation-brightgreen)](REPRODUCIBILITY.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

This repository contains the software, numerical simulation framework,
experiment configurations, analysis utilities, and reproducibility resources
associated with the manuscript:

**Adaptive Amplitude-Phase Encoding and Physically Constrained Optimization for High-Performance Photonic Machine Learning**

The proposed framework jointly learns amplitude and phase optical
representations and a programmable photonic processing transformation within
an end-to-end differentiable pipeline.

The complete computational flow is:

```text
Classical Input
      |
      v
Learnable Amplitude Encoder
      |
      +---- Learnable Phase Encoder
      |
      v
Complex Optical State
      |
      v
Programmable Clements-Mesh PPU
      |
      v
Hardware Non-Idealities
      |
      v
Optical Measurement
      |
      v
Probability Mapping
      |
      v
Classification Readout
      |
      v
End-to-End Optimization
```

## Main Contributions

The repository implements the principal components of the proposed
representation--processor co-adaptation framework:

- learnable amplitude encoding;
- learnable phase encoding;
- complex-valued optical-state construction;
- unit-power optical normalization;
- programmable Clements-mesh photonic processing;
- Mach-Zehnder interferometer primitives;
- square-law optical measurement;
- Born-rule probability mapping;
- hardware-aware optimization;
- insertion-loss modeling;
- phase-uncertainty modeling;
- detector-noise modeling;
- ADC quantization;
- thermal phase-drift modeling;
- controlled baseline comparisons;
- ablation studies;
- hardware-robustness sweeps;
- device-level numerical characterization;
- system-level power, latency, throughput, and scalability analysis;
- reproducibility and figure/table generation utilities.

## Photonic Architecture

The default photonic configuration used by the repository is:

| Parameter | Value |
|---|---:|
| Optical modes | 32 |
| Clements-mesh MZI elements | 496 |
| Optical representation | Amplitude + phase |
| Phase domain | 0 to 2π |
| Default insertion-loss parameter | 1 dB/mode |
| Default phase-noise parameter | 0.05 rad |

For an \(M\)-mode Clements mesh, the number of two-mode interferometric
elements is

\[
L_{\mathrm{MZI}}=\frac{M(M-1)}{2}.
\]

Therefore, for \(M=32\),

\[
L_{\mathrm{MZI}}=496.
\]

The insertion-loss and phase-noise values are configurable simulation
parameters.

## Optical Representation

The optical coefficient of mode \(m\) is represented as

\[
c_m=a_m e^{j\phi_m},
\]

where \(a_m\) is the normalized optical amplitude and \(\phi_m\) is the
optical phase.

The amplitude encoder enforces a normalized optical-power representation:

\[
\sum_m |a_m|^2=1.
\]

The encoded state is subsequently transformed by the programmable photonic
processor.

## Measurement

At the output of the photonic processor, square-law detection provides a
modal intensity representation:

\[
I_m=|c_m^{\mathrm{out}}|^2.
\]

The measured intensities are normalized to form a probability representation
for classification.

## Benchmark Datasets

The repository supports five benchmark categories:

| Dataset | Classes | Primary purpose |
|---|---:|---|
| Circles | 2 | Nonlinear decision-boundary analysis |
| MNIST | 10 | Handwritten digit recognition |
| Fashion-MNIST | 10 | Multi-class visual classification |
| CIFAR-10 | 10 | High-dimensional natural-image classification |
| 16-QAM | 4 | Optical-communication signal classification |

For CIFAR-10, the repository includes a hybrid electronic-photonic
projection interface that reduces the high-dimensional input to the optical
mode dimension.

The 16-QAM component is implemented as a configurable numerical simulation.
It should therefore be interpreted as simulated communication data rather
than direct laboratory measurement data.

## Baseline Models

The repository contains controlled comparison infrastructure for:

1. Adaptive amplitude-phase PNN
2. Fixed-phase PNN
3. Fixed-amplitude PNN
4. Random-unitary PNN
5. Classical multilayer perceptron (MLP)
6. Compact convolutional neural network (CNN)

## Ablation Studies

The principal component-level ablations are:

- Complete framework
- No physical regularization
- No insertion-loss modeling
- No learnable encoding

These configurations are intended to isolate the contribution of adaptive
optical representation learning and hardware-aware optimization.

## Hardware Robustness

The repository provides configurable studies for:

### Phase noise

```text
0.00
0.01
0.02
0.03
0.04
0.05
0.06
0.08
0.10
0.12
0.15 rad
```

### Insertion loss

```text
0
0.5
1.0
1.5
2.0
3.0
4.0
5.0 dB/mode
```

### ADC resolution

```text
4
6
8
10
12
16 bits
```

Additional hardware components include thermal drift and detector-noise
models.

## Device-Level Numerical Analysis

The device simulation layer contains numerical models for:

- TE-like optical mode profiles;
- MZI interference;
- optical spectral response;
- phase modulation;
- thermal phase drift;
- eye-diagram signal behavior.

These are numerical simulation outputs and should not be interpreted as
direct measurements from fabricated photonic hardware.

## System-Level Analysis

The repository separates intrinsic photonic-core performance from the
complete optoelectronic inference chain.

Power accounting includes:

```text
Laser
Modulator
DAC
Photodetector / TIA
ADC
Digital processing
Thermal subsystem
Memory / interface
Photonic core
```

Latency accounting includes:

```text
Data loading
Preprocessing
DAC
Modulation
Optical propagation
Photodetection
ADC
Postprocessing
Interface
```

This distinction is used to prevent the intrinsic optical propagation
latency from being interpreted as complete end-to-end system latency.

## Repository Structure

```text
Adaptive_Photonic_ML/
|
|-- configs/
|   |-- datasets/
|   |-- models/
|   |-- training/
|   |-- hardware/
|   |-- experiments/
|   `-- ablations/
|
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- splits/
|
|-- docs/
|   |-- architecture/
|   |-- mathematical_model/
|   |-- experimental_protocol/
|   |-- hardware_model/
|   |-- reproducibility/
|   `-- figures/
|
|-- experiments/
|-- baselines/
|-- ablations/
|-- robustness/
|
|-- figures/
|-- tables/
|-- notebooks/
|
|-- reproducibility/
|
|-- results/
|   |-- raw/
|   |-- processed/
|   |-- metrics/
|   |-- robustness/
|   |-- ablations/
|   |-- device_simulation/
|   |-- system_analysis/
|   |-- figures/
|   `-- tables/
|
|-- scripts/
|   |-- data/
|   |-- experiments/
|   |-- evaluation/
|   |-- device_simulation/
|   |-- system_analysis/
|   |-- figures/
|   `-- reproduction/
|
|-- src/
|   |-- data/
|   |-- encoding/
|   |-- photonic/
|   |-- measurement/
|   |-- hardware/
|   |-- models/
|   |-- training/
|   |-- evaluation/
|   `-- utils/
|
|-- supplementary/
|-- system_analysis/
|-- tests/
|
|-- README.md
|-- REPRODUCIBILITY.md
|-- CITATION.cff
|-- CONTRIBUTING.md
|-- LICENSE
|-- requirements.txt
`-- environment.yml
```

## Core Source Code

The main implementation is organized as follows.

### Encoding

```text
src/encoding/
```

Contains the amplitude encoder, phase encoder, joint amplitude-phase encoder,
normalization routines, and complex-state utilities.

### Photonic processing

```text
src/photonic/
```

Contains MZI primitives, the programmable Clements mesh, unitary utilities,
and optical propagation functions.

### Measurement

```text
src/measurement/
```

Contains photon-number/square-law detection, probability mapping, and
classification readout.

### Hardware

```text
src/hardware/
```

Contains optical-loss, phase-noise, thermal-drift, detector-noise, ADC,
device-level, and system-level models.

### Models

```text
src/models/
```

Contains the proposed PNN and controlled baseline implementations.

### Training

```text
src/training/
```

Contains classification losses, hardware-aware regularization,
complex-gradient utilities, metrics, and training infrastructure.

### Evaluation

```text
src/evaluation/
```

Contains experiment logging, baseline/ablation definitions, metrics,
statistical aggregation, and result processing.

## Installation

Install the Python dependencies using:

```bash
pip install -r requirements.txt
```

The principal dependencies include:

- PyTorch
- Torchvision
- NumPy
- SciPy
- scikit-learn
- Pandas
- Matplotlib
- PyYAML
- openpyxl
- pytest

A Conda environment specification is also provided:

```text
environment.yml
```

## Dataset Preparation

The main dataset-preparation workflow is:

```bash
python scripts/data/prepare_datasets.py
```

Prepared datasets are organized under:

```text
data/processed/
```

## Experiment Configuration

Important configuration files include:

```text
configs/models/proposed_pnn.yaml
configs/training/default.yaml
configs/experiments/base.yaml
configs/experiments/large_scale_matrix.yaml
configs/ablations/robustness_sweeps.yaml
```

## Running the Proposed Model

A representative experiment can be launched with:

```bash
python scripts/experiments/train_model.py \
    --dataset mnist \
    --model proposed \
    --input-dim 784 \
    --classes 10 \
    --seed 42 \
    --epochs 100
```

Outputs from executed experiments are organized under the results hierarchy.

## Experiment Manifests

The repository separates experiment definitions from experiment outputs.

Manifest generators include:

```text
scripts/experiments/generate_experiment_manifest.py
scripts/experiments/generate_robustness_manifest.py
```

These files define experiment combinations without implying that every
configured experiment has been executed.

## Robustness Evaluation

Hardware-robustness analyses are organized under:

```text
results/robustness/
scripts/evaluation/
configs/ablations/
```

The framework supports phase-noise, insertion-loss, ADC-resolution, detector,
and thermal perturbation studies.

## Device Simulation

The device-level numerical workflow is organized under:

```text
scripts/device_simulation/
results/device_simulation/
```

The main generated categories include mode profiles, MZI transfer behavior,
optical spectrum, phase modulation, thermal drift, and eye-diagram data.

## System Analysis

System-level analytical utilities are located under:

```text
src/hardware/system_accounting.py
scripts/system_analysis/
results/system_analysis/
```

These utilities evaluate power, latency, throughput, energy efficiency, and
scaling behavior.

## Figure and Table Reproduction

Figure-data and visualization utilities are provided under:

```text
scripts/figures/
results/figures_data/
results/figures/
```

Manuscript-oriented table data and Excel exports are maintained under:

```text
tables/
results/tables/
scripts/evaluation/
```

## Reproducibility

The reproducibility package records model, dataset, hardware, seed, and
artifact-provenance information.

See:

```text
REPRODUCIBILITY.md
```

Machine-readable metadata are maintained under:

```text
reproducibility/
```

## Scientific Provenance

The repository distinguishes among:

1. Executed numerical simulations
2. Analytical estimates
3. Literature- or configuration-derived parameters
4. Synthetic reproducibility artifacts
5. Laboratory measurements

Simulation results and synthetic reproducibility artifacts must not be
presented as direct laboratory measurements.

The device and system-level modules in this repository are designed to make
this provenance explicit.

## Testing

Run the available tests with:

```bash
pytest tests/
```

Core tests include checks of:

- model output dimensions;
- probability normalization;
- photonic mesh configuration;
- basic pipeline behavior.

## Citation

Please cite the associated manuscript when using this repository.

Machine-readable citation metadata are provided in:

```text
CITATION.cff
```

## License

This repository is released under the MIT License.

See:

```text
LICENSE
```

for the complete license text.

## Repository

GitHub:

https://github.com/Avi8-9/Adaptive_Photonic_ML
