"""Classical MLP and compact CNN baselines."""

from __future__ import annotations

import torch
from torch import nn


class ClassicalMLP(nn.Module):
    """
    Three-layer dense electronic baseline.

    The network is intentionally compact and is designed to provide
    a reproducible electronic reference for the photonic models.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        previous = input_dim

        for width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.GELU(),
                ]
            )

            if dropout > 0:
                layers.append(
                    nn.Dropout(dropout)
                )

            previous = width

        layers.append(
            nn.Linear(previous, num_classes)
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.network(x)


class CompactCNN(nn.Module):
    """
    Compact convolutional baseline.

    This model is intended for image-shaped input and provides the
    CNN comparison described for the image benchmarks.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                (4, 4)
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 4 * 4,
                128,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):
        return self.classifier(
            self.features(x)
        )


class TwoChannelSignalMLP(nn.Module):
    """
    MLP for complex communication signals represented as I/Q channels.
    """

    def __init__(
        self,
        input_dim: int = 2,
        num_classes: int = 4,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                64,
            ),
            nn.GELU(),

            nn.Linear(
                64,
                64,
            ),
            nn.GELU(),

            nn.Linear(
                64,
                num_classes,
            ),
        )

    def forward(self, x):
        return self.network(x)
