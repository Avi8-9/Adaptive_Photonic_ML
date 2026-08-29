"""Train one model/dataset configuration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.dataset_adapter import (
    load_prepared_dataset,
)

from src.evaluation.metrics import (
    classification_accuracy,
)

from src.evaluation.logger import (
    ExperimentLogger,
)

from src.models.model_factory import (
    build_model,
)

from src.utils.reproducibility import (
    set_seed,
)


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            total_loss += (
                loss.item()
                * len(y)
            )

            predictions = (
                logits.argmax(
                    dim=-1
                )
            )

            correct += int(
                (
                    predictions == y
                ).sum().item()
            )

            total += len(y)

    return (
        total_loss
        / max(total, 1),
        correct
        / max(total, 1),
    )


def train(
    dataset_name: str,
    model_name: str,
    input_dim: int,
    num_classes: int,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_modes: int = 32,
    output_root="results/experiments",
    patience: int = 20,
):

    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    bundle = load_prepared_dataset(
        dataset_name,
        validation_fraction=0.10,
        seed=seed,
    )

    train_dataset = TensorDataset(
        bundle.train_x,
        bundle.train_y,
    )

    validation_dataset = TensorDataset(
        bundle.validation_x,
        bundle.validation_y,
    )

    test_dataset = TensorDataset(
        bundle.test_x,
        bundle.test_y,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_model(
        name=model_name,
        input_dim=input_dim,
        num_classes=num_classes,
        num_modes=num_modes,
        seed=seed,
    )

    model = model.to(device)

    criterion = (
        torch.nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=learning_rate,
    )

    run_directory = (
        Path(output_root)
        / dataset_name
        / model_name
        / f"seed_{seed:04d}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = ExperimentLogger(
        run_directory
    )

    best_validation_accuracy = -1.0
    best_epoch = 0
    patience_counter = 0

    start_time = time.perf_counter()

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            epoch_loss += (
                loss.item()
                * len(y)
            )

            epoch_correct += int(
                (
                    logits.argmax(
                        dim=-1
                    ) == y
                ).sum().item()
            )

            epoch_total += len(y)

        train_loss = (
            epoch_loss
            / max(epoch_total, 1)
        )

        train_accuracy = (
            epoch_correct
            / max(epoch_total, 1)
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        logger.log(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            learning_rate=learning_rate,
            seed=seed,
            dataset=dataset_name,
            model=model_name,
        )

        print(
            f"[{dataset_name}] "
            f"[{model_name}] "
            f"Epoch {epoch:03d} | "
            f"train={train_accuracy:.4f} | "
            f"val={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = (
                validation_accuracy
            )

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "epoch": epoch,
                    "validation_accuracy": (
                        validation_accuracy
                    ),
                },
                run_directory
                / "best_model.pt",
            )

        else:
            patience_counter += 1

            if (
                patience_counter
                >= patience
            ):
                print(
                    "Early stopping."
                )
                break

    elapsed = (
        time.perf_counter()
        - start_time
    )

    checkpoint = torch.load(
        run_directory
        / "best_model.pt",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    logger.save_json()
    logger.save_csv()

    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_time_seconds": elapsed,
        "device": str(device),
        "num_modes": num_modes,
    }

    (
        run_directory
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Finished {dataset_name} / "
        f"{model_name} / seed={seed}"
    )
    print(
        f"Test accuracy: "
        f"{test_accuracy:.4f}"
    )

    return summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--model",
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

    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        model_name=args.model,
        input_dim=args.input_dim,
        num_classes=args.classes,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
