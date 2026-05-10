"""Multi-head image classifier for RF connector attributes.

Wraps a torchvision-style backbone with one linear head per attribute. The
backbone is pluggable so the cloud bake-off can compare ResNet-18,
ResNet-50, EfficientNetV2-S, MobileNetV3, MobileViT, and ConvNeXt-Tiny.

Tests construct a tiny stub backbone so the model can be exercised on the
CPU without downloading weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

try:  # pragma: no cover - torch is required only in cloud training
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


SUPPORTED_BACKBONES = (
    "resnet18",
    "resnet50",
    "efficientnet_v2_s",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "mobilevit_s",
    "convnext_tiny",
)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    embedding_dim: int


def list_supported_backbones() -> tuple[str, ...]:
    return SUPPORTED_BACKBONES


def _build_backbone(name: str):  # pragma: no cover - cloud only
    """Build a torchvision backbone stripped of its classification head."""
    import torchvision.models as tvm  # type: ignore[import-not-found]

    if name == "resnet18":
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        emb = net.fc.in_features
        net.fc = nn.Identity()
        return net, emb
    if name == "resnet50":
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        emb = net.fc.in_features
        net.fc = nn.Identity()
        return net, emb
    if name == "efficientnet_v2_s":
        net = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.DEFAULT)
        emb = net.classifier[-1].in_features
        net.classifier[-1] = nn.Identity()
        return net, emb
    if name == "mobilenet_v3_small":
        net = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
        emb = net.classifier[-1].in_features
        net.classifier[-1] = nn.Identity()
        return net, emb
    if name == "mobilenet_v3_large":
        net = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT)
        emb = net.classifier[-1].in_features
        net.classifier[-1] = nn.Identity()
        return net, emb
    if name == "convnext_tiny":
        net = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        emb = net.classifier[-1].in_features
        net.classifier[-1] = nn.Identity()
        return net, emb
    if name == "mobilevit_s":
        try:
            import timm  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "timm is required for mobilevit_s backbone"
            ) from exc
        net = timm.create_model("mobilevit_s", pretrained=True, num_classes=0)
        return net, net.num_features
    raise ValueError(f"unsupported backbone {name!r}")


class MultiHeadClassifier(nn.Module if nn is not None else object):
    """Backbone + one linear head per attribute.

    The factory function :func:`build_multihead_classifier` is the normal
    entry. Tests inject a stub backbone via :func:`new_with_stub_backbone`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        head_sizes: Mapping[str, int],
    ) -> None:
        if nn is None:
            raise RuntimeError("torch is required to instantiate MultiHeadClassifier")
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.heads = nn.ModuleDict(
            {name: nn.Linear(embedding_dim, size) for name, size in head_sizes.items()}
        )

    def forward(self, x):
        features = self.backbone(x)
        return {name: head(features) for name, head in self.heads.items()}


def build_multihead_classifier(
    backbone_name: str,
    head_sizes: Mapping[str, int],
) -> MultiHeadClassifier:  # pragma: no cover - cloud only
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"backbone {backbone_name!r} not in supported {SUPPORTED_BACKBONES}"
        )
    backbone, emb = _build_backbone(backbone_name)
    return MultiHeadClassifier(backbone, emb, head_sizes)


def new_with_stub_backbone(
    head_sizes: Mapping[str, int],
    embedding_dim: int = 8,
    feature_factory: Callable[[int], "nn.Module"] | None = None,
) -> MultiHeadClassifier:
    """Build a tiny CPU-friendly model for unit tests.

    The default stub flattens the input and runs it through a single linear
    layer, so a tiny ``(B, 3, 8, 8)`` input is enough to exercise the heads.
    """
    if nn is None:
        raise RuntimeError("torch is required for new_with_stub_backbone")

    if feature_factory is None:
        class _Stub(nn.Module):
            def __init__(self, in_features: int, out_features: int) -> None:
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear = nn.Linear(in_features, out_features)

            def forward(self, x):  # type: ignore[override]
                return self.linear(self.flatten(x))

        in_features = 3 * 8 * 8
        backbone = _Stub(in_features=in_features, out_features=embedding_dim)
    else:
        backbone = feature_factory(embedding_dim)

    return MultiHeadClassifier(backbone, embedding_dim, head_sizes)
