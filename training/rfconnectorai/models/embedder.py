from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# Recommended input image size per backbone family.
# DINOv2 uses patch_size=14, so image must be a multiple of 14 (224 = 14×16).
# MobileViT was pretrained at 256 and works well up to 384 on RF crops.
_RECOMMENDED_IMAGE_SIZE = {
    "mobilevitv2_100": 384,
    "vit_small_patch14_dinov2.lvd142m": 224,
    "vit_base_patch14_dinov2.lvd142m": 224,
}


def recommended_image_size(backbone: str) -> int:
    """Suggested input resolution for a given backbone."""
    return _RECOMMENDED_IMAGE_SIZE.get(backbone, 384)


class RGBDEmbedder(nn.Module):
    """
    Backbone-agnostic embedder for 4-channel RGBD input.

    - Uses timm's `in_chans=4` to expand the first conv to 4 channels and (when
      `pretrained=True`) replicate the pretrained RGB filter statistics into
      the depth channel.
    - Adds `dynamic_img_size=True` for ViT-family backbones (DINOv2, plain
      ViT) so the model accepts arbitrary input sizes that are multiples of
      patch size, not just the pretraining size.
    - Replaces the classification head with a projection to `embedding_dim`.
    - LayerNorm in the projection head (vs BatchNorm1d) is stable with the
      small/variable batch sizes typical in metric-learning setups.
    - L2-normalizes outputs for cosine-similarity matching and batch-hard
      triplet loss.

    Tested backbones:
      - "mobilevitv2_100"                       (default; ~5M params, fast)
      - "vit_small_patch14_dinov2.lvd142m"      (~21M; strong fine-grained features)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        backbone: str = "mobilevitv2_100",
    ) -> None:
        super().__init__()

        # Try to construct with dynamic_img_size (pure ViT-family models support
        # this and need it to accept arbitrary input sizes); fall back to a
        # plain construction for backbones that don't accept the kwarg
        # (MobileViT, Swin, MaxViT, etc., have their own size handling).
        common = dict(pretrained=pretrained, num_classes=0, in_chans=4)
        try:
            self.backbone = timm.create_model(backbone, **common, dynamic_img_size=True)
        except TypeError:
            self.backbone = timm.create_model(backbone, **common)

        feat_dim = self.backbone.num_features
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)           # (B, feat_dim)
        emb = self.projection(feats)       # (B, embedding_dim)
        return F.normalize(emb, p=2, dim=1)
