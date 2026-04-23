from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBDEmbedder(nn.Module):
    """
    MobileViT-v2 backbone adapted for 4-channel RGBD input.

    - Relies on timm's `in_chans=4` support, which handles the first-conv
      input-channel expansion and, when loading pretrained weights, replicates
      the RGB filter statistics into the depth channel. This avoids manually
      walking the backbone's module tree looking for the "first" 3-channel
      Conv2d (fragile across backbones).
    - Replaces the classification head with a projection to `embedding_dim`.
    - Uses LayerNorm (rather than BatchNorm1d) so training is stable with
      small or single-sample batches common in metric-learning setups.
    - L2-normalizes outputs for cosine-similarity matching and batch-hard
      triplet loss.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        backbone: str = "mobilevitv2_100",
    ) -> None:
        super().__init__()

        # `in_chans=4` lets timm construct a 4-channel first conv and, if
        # `pretrained=True`, copy + replicate the pretrained RGB weights.
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            in_chans=4,
        )

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
