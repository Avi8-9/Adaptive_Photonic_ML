"""Basic functional tests for the photonic pipeline."""

import torch

from src.models.photonic_pnn import AdaptivePhotonicPNN


def test_forward_shape():
    model = AdaptivePhotonicPNN(
        input_dim=32,
        num_classes=10,
        num_modes=32,
    )

    x = torch.randn(8, 32)

    logits = model(x)

    assert logits.shape == (8, 10)


def test_probability_normalization():
    model = AdaptivePhotonicPNN(
        input_dim=32,
        num_classes=10,
        num_modes=32,
    )

    x = torch.randn(4, 32)

    outputs = model(
        x,
        return_intermediate=True,
    )

    probabilities = outputs["probabilities"]

    sums = probabilities.sum(dim=-1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5,
    )


def test_clements_mzi_count():
    model = AdaptivePhotonicPNN(
        input_dim=32,
        num_classes=10,
        num_modes=32,
    )

    assert (
        model.photonic_processor.expected_mzi_count
        == 496
    )
