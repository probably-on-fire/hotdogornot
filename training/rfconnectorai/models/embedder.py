from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBDEmbedder(nn.Module):
    """
    MobileViT-v2 backbone adapted for 4-channel RGBD input.

    - Replaces the first conv layer (3 -> out_channels) with a 4-channel version.
    - When pretrained=True, initializes the depth channel's conv weights to the
      mean of the pretrained RGB weights. This gives training a reasonable start
      rather than random init on the depth filter.
    - Replaces the classification head with a projection to `embedding_dim`.
    - L2-normalizes outputs for use with cosine-similarity-based matching and
      the batch-hard triplet loss.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        backbone: str = "mobilevitv2_100",
    ) -> None:
        super().__init__()

        # Create backbone without a classification head (num_classes=0 returns features).
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        # Replace the first conv layer to accept 4 input channels.
        original_first = self._find_first_conv(self.backbone)
        new_first = nn.Conv2d(
            in_channels=4,
            out_channels=original_first.out_channels,
            kernel_size=original_first.kernel_size,
            stride=original_first.stride,
            padding=original_first.padding,
            bias=(original_first.bias is not None),
        )
        with torch.no_grad():
            new_first.weight[:, :3] = original_first.weight
            # Depth channel: mean of RGB filter weights.
            new_first.weight[:, 3:4] = original_first.weight.mean(dim=1, keepdim=True)
            if original_first.bias is not None:
                new_first.bias.copy_(original_first.bias)
        self._replace_first_conv(self.backbone, new_first)

        # Projection head from backbone features to embedding_dim.
        # LayerNorm rather than BatchNorm so training works with small/variable
        # batch sizes (common with balanced PK sampling in metric learning).
        feat_dim = self.backbone.num_features
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, embedding_dim),
        )

    @staticmethod
    def _find_first_conv(module: nn.Module) -> nn.Conv2d:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("No Conv2d found in backbone")

    @staticmethod
    def _replace_first_conv(module: nn.Module, new_conv: nn.Conv2d) -> None:
        for name, child in module.named_modules():
            for attr_name, attr_val in list(vars(child).get("_modules", {}).items()):
                if isinstance(attr_val, nn.Conv2d) and attr_val.in_channels == 3:
                    setattr(child, attr_name, new_conv)
                    return
        raise RuntimeError("Failed to replace first Conv2d")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)           # (B, feat_dim)
        emb = self.projection(feats)       # (B, embedding_dim)
        return F.normalize(emb, p=2, dim=1)
