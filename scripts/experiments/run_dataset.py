"""Generic experiment launcher for a prepared tensor dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.photonic_pnn import AdaptivePhotonicPNN
from src.training.trainer import PhotonicTrainer
from src.utils.reproducibility import set_seed


def load_pt_dataset(path):
    payload = torch.load(
        path,
        map_location="cpu",
    )

    return TensorDataset(
        payload["features"].float(),
        payload["labels"].long(),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
    )

    parser.add_argument(
        "--test",
        required=True,
    )

    parser.add_argument(
        "--input-dim",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--classes",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--experiment-name",
        default="experiment",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--lambda-physical",
        type=float,
        default=1e-3,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    train_dataset = load_pt_dataset(
        args.train
    )

    test_dataset = load_pt_dataset(
        args.test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = AdaptivePhotonicPNN(
        input_dim=args.input_dim,
        num_classes=args.classes,
        num_modes=32,
        insertion_loss_db=1.0,
        phase_noise_sigma=0.05,
        enable_hardware_noise=False,
    )

    trainer = PhotonicTrainer(
        model=model,
        learning_rate=args.learning_rate,
        lambda_physical=args.lambda_physical,
    )

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=None,
        epochs=args.epochs,
    )

    output_dir = (
        Path("results")
        / "experiments"
        / args.experiment_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        output_dir / "model.pt",
    )

    with open(
        output_dir / "training_history.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "train_loss": history.train_loss,
                "train_accuracy": history.train_accuracy,
            },
            handle,
            indent=2,
        )

    print(
        f"\nExperiment saved to {output_dir}"
    )


if __name__ == "__main__":
    main()
