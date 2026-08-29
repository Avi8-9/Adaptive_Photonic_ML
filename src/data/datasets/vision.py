"""Vision benchmark download and preparation utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms


DATASET_INFO = {
    "mnist": {
        "classes": 10,
        "shape": (28, 28),
    },
    "fashion_mnist": {
        "classes": 10,
        "shape": (28, 28),
    },
    "cifar10": {
        "classes": 10,
        "shape": (32, 32, 3),
    },
}


def _dataset(
    name: str,
    root: str | Path,
    train: bool,
    download: bool = True,
):
    root = Path(root)

    transform = transforms.ToTensor()

    if name.lower() == "mnist":
        return datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

    if name.lower() == "fashion_mnist":
        return datasets.FashionMNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

    if name.lower() == "cifar10":
        return datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

    raise ValueError(
        f"Unsupported dataset: {name}"
    )


def load_vision_dataset(
    name: str,
    root: str | Path = "data/raw",
    download: bool = True,
):
    """
    Load the requested benchmark.

    Returns
    -------
    train_dataset, test_dataset
    """
    name = name.lower()

    train = _dataset(
        name,
        root,
        train=True,
        download=download,
    )

    test = _dataset(
        name,
        root,
        train=False,
        download=download,
    )

    return train, test


def flatten_dataset(dataset):
    """
    Convert an image dataset into a flattened float tensor.

    Pixel values remain in [0, 1].
    """
    features = []
    labels = []

    for image, label in dataset:
        features.append(
            image.reshape(-1)
        )

        labels.append(
            int(label)
        )

    x = torch.stack(features)
    y = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return x, y
