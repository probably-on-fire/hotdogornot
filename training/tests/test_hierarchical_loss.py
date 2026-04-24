import torch
import torch.nn.functional as F
from rfconnectorai.training.hierarchical_loss import HierarchicalAuxLoss


def test_hierarchical_aux_loss_shape_and_signal():
    loss_mod = HierarchicalAuxLoss(embedding_dim=16, n_families=2, n_genders=2)
    emb = F.normalize(torch.randn(8, 16), dim=1)
    family_labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    gender_labels = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1])

    loss = loss_mod(emb, family_labels, gender_labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_hierarchical_aux_loss_has_trainable_params():
    loss_mod = HierarchicalAuxLoss(embedding_dim=8, n_families=2, n_genders=2)
    assert any(p.requires_grad for p in loss_mod.parameters())


def test_hierarchical_aux_loss_gradients_flow_to_embedding():
    loss_mod = HierarchicalAuxLoss(embedding_dim=8, n_families=2, n_genders=2)
    emb = F.normalize(torch.randn(4, 8), dim=1).requires_grad_(True)
    family = torch.tensor([0, 1, 0, 1])
    gender = torch.tensor([0, 0, 1, 1])
    loss = loss_mod(emb, family, gender)
    loss.backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0
