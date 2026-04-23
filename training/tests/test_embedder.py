import pytest
import torch
from rfconnectorai.models.embedder import RGBDEmbedder


def test_forward_shape_is_128d():
    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    x = torch.randn(2, 4, 384, 384)
    out = model(x)
    assert out.shape == (2, 128)


def test_output_is_l2_normalized():
    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    model.eval()
    x = torch.randn(4, 4, 384, 384)
    with torch.no_grad():
        out = model(x)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_rejects_3channel_input():
    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    x = torch.randn(1, 3, 384, 384)
    with pytest.raises(RuntimeError):
        model(x)


def test_gradients_flow_through_depth_channel():
    model = RGBDEmbedder(embedding_dim=128, pretrained=False)
    x = torch.randn(1, 4, 384, 384, requires_grad=True)
    out = model(x)
    out.sum().backward()
    # Gradient must be non-zero on the depth channel (channel index 3)
    assert x.grad[:, 3].abs().sum().item() > 0.0


def test_custom_embedding_dim():
    model = RGBDEmbedder(embedding_dim=64, pretrained=False)
    x = torch.randn(1, 4, 384, 384)
    out = model(x)
    assert out.shape == (1, 64)
