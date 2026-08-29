"""Common interface for prepared repository datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset


class DatasetBundle:
    """Container holding train, validation, and test tensors."""

    def __init__(
        self,
        train_x,
        train_y,
        validation_x,
        validation_y,
        test_x,
        test_y,
    ):
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.test_x = test_x
        self.test_y = test_y

    @property
    def input_dimension(self):
        return int(
            self.train_x.shape[-1]
        )

    @property
    def num_classes(self):
        return int(
            torch.unique(
                self.train_y
            ).numel()
        )


def _load(path):
    payload = torch.load(
        path,
        map_location="cpu",
    )

    return (
        payload["features"].float(),
        payload["labels"].long(),
    )


def make_validation_split(
    x,
    y,
    validation_fraction=0.10,
    seed=42,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(
        len(x),
        generator=generator,
    )

    validation_size = int(
        len(x) * validation_fraction
    )

    validation_idx = permutation[
        :validation_size
    ]

    train_idx = permutation[
        validation_size:
    ]

    return (
        x[train_idx],
        y[train_idx],
        x[validation_idx],
        y[validation_idx],
    )


def load_prepared_dataset(
    dataset_name: str,
    root: str | Path = "data/processed",
    validation_fraction: float = 0.10,
    seed: int = 42,
):
    root = Path(root)

    name = dataset_name.lower()

    train_path = (
        root
        / name
        / "train.pt"
    )

    test_path = (
        root
        / name
        / "test.pt"
    )

    if name == "circles":
        full_path = (
            root
            / "circles"
            / "circles_full.pt"
        )

        x, y = _load(
            full_path
        )

        (
            train_x,
            train_y,
            validation_x,
            validation_y,
        ) = make_validation_split(
            x,
            y,
            validation_fraction,
            seed,
        )

        test_count = min(
            1000,
            len(validation_x),
        )

        test_x = validation_x[
            :test_count
        ]

        test_y = validation_y[
            :test_count
        ]

        validation_x = validation_x[
            test_count:
        ]

        validation_y = validation_y[
            test_count:
        ]

    elif name == "qam16":
        full_path = (
            root
            / "qam16"
            / "full.pt"
        )

        x, y = _load(
            full_path
        )

        (
            train_x,
            train_y,
            validation_x,
            validation_y,
        ) = make_validation_split(
            x,
            y,
            validation_fraction,
            seed,
        )

        test_count = min(
            10000,
            len(validation_x),
        )

        test_x = validation_x[
            :test_count
        ]

        test_y = validation_y[
            :test_count
        ]

        validation_x = validation_x[
            test_count:
        ]

        validation_y = validation_y[
            test_count:
        ]

    else:
        if (
            not train_path.exists()
            or not test_path.exists()
        ):
            raise FileNotFoundError(
                "Prepared dataset files not found. "
                "Run scripts/data/prepare_datasets.py first."
            )

        train_x, train_y = _load(
            train_path
        )

        test_x, test_y = _load(
            test_path
        )

        (
            train_x,
            train_y,
            validation_x,
            validation_y,
        ) = make_validation_split(
            train_x,
            train_y,
            validation_fraction,
            seed,
        )

    return DatasetBundle(
        train_x,
        train_y,
        validation_x,
        validation_y,
        test_x,
        test_y,
    )
