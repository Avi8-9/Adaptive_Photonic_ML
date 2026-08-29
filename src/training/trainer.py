"""Training engine for the proposed photonic model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from .losses import cross_entropy_loss
from .metrics import accuracy
from .regularization import phase_gradient_penalty


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    validation_accuracy: list[float] = field(default_factory=list)


class PhotonicTrainer:
    """
    Mini-batch training loop with optional hardware-aware regularization.
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        lambda_physical: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.lambda_physical = lambda_physical

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
        )

        self.history = TrainingHistory()

    def _batch(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        training: bool = True,
    ):
        features = features.to(self.device)
        labels = labels.to(self.device)

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(features)

        classification_loss = cross_entropy_loss(
            logits,
            labels,
        )

        physical_penalty = torch.zeros(
            (),
            device=self.device,
            dtype=classification_loss.dtype,
        )

        if self.lambda_physical != 0.0:

            physical_penalty = phase_gradient_penalty(
                classification_loss,
                [
                    self.model.photonic_processor.theta,
                    self.model.photonic_processor.phi,
                ],
            )

        total_loss = (
            classification_loss
            + self.lambda_physical * physical_penalty
        )

        if training:
            total_loss.backward()
            self.optimizer.step()

        batch_accuracy = accuracy(
            logits.detach(),
            labels,
        )

        return (
            float(total_loss.detach().item()),
            batch_accuracy,
        )

    def train_epoch(
        self,
        loader: DataLoader,
    ) -> tuple[float, float]:

        self.model.train()

        losses = []
        accuracies = []

        for features, labels in loader:
            loss, acc = self._batch(
                features,
                labels,
                training=True,
            )

            losses.append(loss)
            accuracies.append(acc)

        return (
            sum(losses) / max(len(losses), 1),
            sum(accuracies) / max(len(accuracies), 1),
        )

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
    ) -> tuple[float, float]:

        self.model.eval()

        losses = []
        accuracies = []

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(features)

            loss = cross_entropy_loss(
                logits,
                labels,
            )

            losses.append(float(loss.item()))
            accuracies.append(
                accuracy(logits, labels)
            )

        return (
            sum(losses) / max(len(losses), 1),
            sum(accuracies) / max(len(accuracies), 1),
        )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
        epochs: int = 100,
    ) -> TrainingHistory:

        for epoch in range(1, epochs + 1):

            train_loss, train_acc = self.train_epoch(
                train_loader
            )

            self.history.train_loss.append(
                train_loss
            )

            self.history.train_accuracy.append(
                train_acc
            )

            if validation_loader is not None:
                val_loss, val_acc = self.validate(
                    validation_loader
                )

                self.history.validation_loss.append(
                    val_loss
                )

                self.history.validation_accuracy.append(
                    val_acc
                )

                print(
                    f"Epoch {epoch:03d} | "
                    f"train loss={train_loss:.5f} | "
                    f"train acc={train_acc:.4f} | "
                    f"val loss={val_loss:.5f} | "
                    f"val acc={val_acc:.4f}"
                )

            else:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train loss={train_loss:.5f} | "
                    f"train acc={train_acc:.4f}"
                )

        return self.history
