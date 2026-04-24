import pytest
import torch
import torch.nn.functional as F

from rfconnectorai.training.arcface_loss import ArcFaceLoss


def test_arcface_runs_and_returns_scalar():
    loss = ArcFaceLoss(embedding_dim=8, num_classes=4, margin=0.5, scale=30.0)
    embeddings = F.normalize(torch.randn(16, 8), dim=1)
    labels = torch.randint(0, 4, (16,))
    val = loss(embeddings, labels)
    assert val.dim() == 0
    assert torch.isfinite(val)
    assert val.item() >= 0.0


def test_arcface_separated_classes_lower_loss_than_random():
    """A model whose embeddings line up with class weights should incur lower
    loss than embeddings drawn at random."""
    torch.manual_seed(123)
    loss = ArcFaceLoss(embedding_dim=4, num_classes=4, margin=0.5, scale=30.0)

    # Force the class weights to be the standard basis (one axis per class).
    with torch.no_grad():
        loss.weight.copy_(torch.eye(4))

    # Perfectly separated embeddings: each sample exactly matches its class axis.
    separated = torch.eye(4)
    labels = torch.arange(4)
    val_sep = loss(separated, labels).item()

    random_emb = F.normalize(torch.randn(4, 4), dim=1)
    val_rand = loss(random_emb, labels).item()

    assert val_sep < val_rand


def test_arcface_gradients_flow_to_embeddings():
    loss = ArcFaceLoss(embedding_dim=8, num_classes=4)
    embeddings = F.normalize(torch.randn(16, 8), dim=1).requires_grad_(True)
    labels = torch.randint(0, 4, (16,))
    val = loss(embeddings, labels)
    val.backward()
    assert embeddings.grad is not None
    assert embeddings.grad.abs().sum().item() > 0.0


def test_arcface_gradients_flow_to_class_weights():
    loss = ArcFaceLoss(embedding_dim=8, num_classes=4)
    embeddings = F.normalize(torch.randn(16, 8), dim=1)
    labels = torch.randint(0, 4, (16,))
    val = loss(embeddings, labels)
    val.backward()
    assert loss.weight.grad is not None
    assert loss.weight.grad.abs().sum().item() > 0.0


def test_arcface_rejects_invalid_labels():
    loss = ArcFaceLoss(embedding_dim=4, num_classes=3)
    embeddings = F.normalize(torch.randn(2, 4), dim=1)
    bad_labels = torch.tensor([0, 5])  # 5 is out of range
    with pytest.raises((IndexError, RuntimeError)):
        loss(embeddings, bad_labels)
