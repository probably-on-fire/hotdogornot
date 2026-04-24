"""
ArcFace loss for fine-grained metric learning.

Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
Deng et al. 2019, https://arxiv.org/abs/1801.07698

ArcFace augments standard softmax-cross-entropy by adding a fixed additive
angular margin m to the angle between an embedding and its target class's
weight vector. This forces same-class embeddings to compress more tightly
than vanilla softmax requires, sharpening inter-class boundaries — which is
exactly what we want for fine-grained connector identification where some
classes (2.4 / 2.92 / 3.5 mm) are visually near-identical.

Used in the training loop instead of (or alongside) batch-hard triplet loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    Computes ArcFace loss given L2-normalized embeddings and integer labels.

    The class-prototype weights are learned. Embeddings are expected to be
    L2-normalized already (RGBDEmbedder ensures this).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize the class weight matrix (rows are unit prototypes).
        W = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity matrix (B, C).
        cos_theta = embeddings @ W.T
        # Numerical safety for acos near ±1.
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Add the angular margin only to the target-class entry per sample.
        theta = torch.acos(cos_theta)
        theta_with_margin = theta + self.margin
        cos_theta_target = torch.cos(theta_with_margin)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(cos_theta.dtype)
        logits = one_hot * cos_theta_target + (1.0 - one_hot) * cos_theta
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)
