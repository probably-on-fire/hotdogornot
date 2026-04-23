import pytest
import torch
from rfconnectorai.training.triplet_loss import batch_hard_triplet_loss


def test_zero_loss_when_classes_perfectly_separated():
    # Two classes placed at opposite unit vectors — perfectly separable.
    embeddings = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [-1.0, 0.0],
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.5)
    # Easy-positive (distance 0) + easy-negative (distance 2) + margin 0.5 → loss = max(0 - 2 + 0.5, 0) = 0
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_positive_loss_when_classes_overlap():
    # Same-class points are farther apart than cross-class points — loss must be > 0.
    embeddings = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],    # same class as above, but far
        [0.1, 0.0],    # different class, close to first point
        [0.9, 0.0],    # different class, close to second point
    ], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.2)
    assert loss.item() > 0.0


def test_loss_is_gradient_bearing():
    embeddings = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.3)
    loss.backward()
    assert embeddings.grad is not None
    assert embeddings.grad.abs().sum() > 0.0


def test_requires_at_least_two_classes():
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 0, 0, 0])  # single class → no valid negatives
    with pytest.raises(ValueError):
        batch_hard_triplet_loss(embeddings, labels, margin=0.2)


def test_requires_at_least_two_samples_per_class():
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])  # one sample per class → no valid positives
    with pytest.raises(ValueError):
        batch_hard_triplet_loss(embeddings, labels, margin=0.2)
