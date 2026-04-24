"""
Auxiliary hierarchical classification loss for the embedder.

At inference the embedder produces a single 128-d vector and all class decisions
come from nearest-neighbor matching. During training we can make that embedding
space easier to cluster by adding two auxiliary linear classifiers on top of
the embedding and training them jointly: one for "family" (SMA vs precision)
and one for "gender" (male vs female). These are high-contrast signals the
model will nail; the auxiliary gradient helps the embedding space organize
itself along those axes, which makes the hard 2.4/2.92/3.5mm subspace cleaner
for the main ArcFace loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalAuxLoss(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_families: int = 2,
        n_genders: int = 2,
        family_weight: float = 0.3,
        gender_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.family_head = nn.Linear(embedding_dim, n_families)
        self.gender_head = nn.Linear(embedding_dim, n_genders)
        self.family_weight = family_weight
        self.gender_weight = gender_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        family_labels: torch.Tensor,
        gender_labels: torch.Tensor,
    ) -> torch.Tensor:
        fam_logits = self.family_head(embeddings)
        gen_logits = self.gender_head(embeddings)
        fam_loss = F.cross_entropy(fam_logits, family_labels)
        gen_loss = F.cross_entropy(gen_logits, gender_labels)
        return self.family_weight * fam_loss + self.gender_weight * gen_loss
