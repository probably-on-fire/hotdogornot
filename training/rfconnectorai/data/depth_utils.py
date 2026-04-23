from __future__ import annotations

import numpy as np


def synthesize_depth_from_mask(
    mask: np.ndarray,
    focal_length_px: float,
    object_depth_m: float,
    background_depth_m: float = 2.0,
    noise_std_m: float = 0.002,
    seed: int | None = None,
) -> np.ndarray:
    """
    Produce a plausible depth map for an RGB-only image given a foreground mask.

    Foreground pixels get depth ~object_depth_m with small Gaussian noise.
    Background pixels get background_depth_m with larger noise.

    This exists so the RGBD embedder can train on scraped catalog images by
    treating the object as an approximately flat disc at a typical hand-held
    distance. Real LiDAR captures replace this in Phase 1.
    """
    if mask.dtype != bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")

    rng = np.random.default_rng(seed)
    depth = np.full(mask.shape, background_depth_m, dtype=np.float32)
    depth[mask] = object_depth_m

    # Add noise to prevent the model from learning a trivial "flat depth = object" shortcut.
    depth += rng.normal(0.0, noise_std_m, size=mask.shape).astype(np.float32)

    # Clamp to positive values (depth must be non-negative).
    depth = np.maximum(depth, 0.01)

    _ = focal_length_px  # reserved for future per-pixel geometric variation
    return depth
