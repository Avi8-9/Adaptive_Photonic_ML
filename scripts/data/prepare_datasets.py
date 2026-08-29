"""Prepare all benchmark datasets used by the repository."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.datasets.circles import prepare_circles
from src.data.datasets.common import (
    save_metadata,
    save_tensor_dataset,
)
from src.data.datasets.vision import (
    load_vision_dataset,
    flatten_dataset,
)
from src.data.qam16.simulator import (
    QAM16Config,
    generate_qam16,
)


def prepare_vision(
    name: str,
    raw_root: Path,
    output_root: Path,
):
    print(f"Preparing {name} ...")

    train, test = load_vision_dataset(
        name,
        root=raw_root,
        download=True,
    )

    train_x, train_y = flatten_dataset(
        train
    )

    test_x, test_y = flatten_dataset(
        test
    )

    dataset_dir = output_root / name
    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_tensor_dataset(
        train_x,
        train_y,
        dataset_dir / "train.pt",
    )

    save_tensor_dataset(
        test_x,
        test_y,
        dataset_dir / "test.pt",
    )

    save_metadata(
        {
            "dataset": name,
            "train_samples": len(train_y),
            "test_samples": len(test_y),
            "classes": 10,
            "flattened_input_dimension": train_x.shape[1],
        },
        dataset_dir / "metadata.json",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        default="data/raw",
    )

    parser.add_argument(
        "--output-root",
        default="data/processed",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)

    # Circles benchmark.
    prepare_circles(
        output_root / "circles",
        n_samples=6000,
        seed=args.seed,
    )

    # Vision benchmarks.
    for name in [
        "mnist",
        "fashion_mnist",
        "cifar10",
    ]:
        prepare_vision(
            name,
            raw_root,
            output_root,
        )

    # Simulated 16-QAM.
    qam_config = QAM16Config(
        seed=args.seed
    )

    _, qam_x, qam_y = generate_qam16(
        qam_config
    )

    qam_dir = output_root / "qam16"
    qam_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_tensor_dataset(
        qam_x,
        qam_y,
        qam_dir / "full.pt",
    )

    save_metadata(
        {
            "dataset": "16-QAM",
            "samples": len(qam_y),
            "input_dimension": 2,
            "classes": 4,
            "generator": "configurable simulated channel",
            "seed": args.seed,
            "note": (
                "Simulation parameters are explicitly configurable "
                "because the manuscript does not uniquely specify "
                "the original source-data generation procedure."
            ),
        },
        qam_dir / "metadata.json",
    )

    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()
