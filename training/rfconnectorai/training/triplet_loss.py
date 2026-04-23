from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Batch-hard triplet loss using squared Euclidean distance on L2-normalized embeddings.

    For each anchor a in the batch:
      - hardest positive  = max_{p : label(p) = label(a), p != a} d(a, p)
      - hardest negative  = min_{n : label(n) != label(a)}       d(a, n)
      - triplet loss      = max(hardest_pos - hardest_neg + margin, 0)

    Returns the mean triplet loss across the batch.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be 1-D and match batch size")

    unique = torch.unique(labels)
    if unique.numel() < 2:
        raise ValueError("batch-hard triplet loss requires at least two distinct classes")

    # Check that at least one class has >=2 samples (otherwise no valid positive).
    counts = torch.stack([(labels == c).sum() for c in unique])
    if counts.max().item() < 2:
        raise ValueError("batch-hard triplet loss requires at least two samples of some class")

    # Normalize so distances are in a stable range.
    emb = F.normalize(embeddings, p=2, dim=1)

    # Pairwise squared Euclidean distances: d_ij = 2 - 2 * cos_sim (since vectors unit-norm)
    # but compute from first principles to keep it explicit.
    diff = emb.unsqueeze(0) - emb.unsqueeze(1)          # (N, N, D)
    dist = (diff ** 2).sum(dim=-1)                       # (N, N)

    # Masks
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (N, N)
    labels_ne = ~labels_eq
    eye = torch.eye(emb.size(0), dtype=torch.bool, device=emb.device)

    positive_mask = labels_eq & ~eye  # same class, not self
    negative_mask = labels_ne

    # For each anchor, find hardest positive (max distance) within same-class samples.
    # Replace invalid (non-positive) slots with -inf so they are ignored by max.
    pos_dist = dist.clone()
    pos_dist[~positive_mask] = float("-inf")
    hardest_pos = pos_dist.max(dim=1).values  # (N,)

    # For each anchor, find hardest negative (min distance) across different-class samples.
    neg_dist = dist.clone()
    neg_dist[~negative_mask] = float("inf")
    hardest_neg = neg_dist.min(dim=1).values  # (N,)

    # Anchors that have no valid positive (class with a single sample) will have -inf.
    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    if not valid.any():
        raise ValueError("No anchors with both a valid positive and negative were found.")

    losses = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return losses.mean()
