"""
Physically-based material specs for connector rendering.

Each material entry describes base color + metallic + roughness + (for dielectrics)
refractive index. Values are chosen to look plausible under common lab lighting;
the goal is not physically-exact replication, it's enough variance for a
model-trained-on-CAD to generalize to real-world photos.
"""

from __future__ import annotations

import numpy as np


BRASS = "brass"
STAINLESS_STEEL = "stainless_steel"
GOLD_PLATED = "gold_plated"
PTFE = "ptfe"


MATERIAL_LIBRARY: dict[str, dict] = {
    BRASS: {
        "base_color": (0.92, 0.82, 0.55, 1.0),
        "metallic": 1.0,
        "roughness": 0.35,
    },
    STAINLESS_STEEL: {
        "base_color": (0.78, 0.78, 0.80, 1.0),
        "metallic": 1.0,
        "roughness": 0.30,
    },
    GOLD_PLATED: {
        "base_color": (1.00, 0.85, 0.50, 1.0),
        "metallic": 1.0,
        "roughness": 0.20,
    },
    PTFE: {
        "base_color": (0.95, 0.95, 0.94, 1.0),
        "metallic": 0.0,
        "roughness": 0.45,
        "ior": 1.38,           # Teflon / PTFE refractive index
    },
}


def make_material(
    name: str,
    roughness_jitter: float = 0.0,
    color_jitter: float = 0.0,
    seed: int | None = None,
) -> dict:
    """
    Return a material spec dict with small randomizations applied.
    Values clamp into a plausible physical range.
    """
    if name not in MATERIAL_LIBRARY:
        raise ValueError(f"Unknown material {name!r}")

    rng = np.random.default_rng(seed)
    spec = dict(MATERIAL_LIBRARY[name])  # copy
    if roughness_jitter > 0.0:
        delta = float(rng.uniform(-roughness_jitter, roughness_jitter))
        spec["roughness"] = max(0.05, min(0.95, spec["roughness"] + delta))
    if color_jitter > 0.0:
        r, g, b, a = spec["base_color"]
        spec["base_color"] = (
            max(0.0, min(1.0, r + float(rng.uniform(-color_jitter, color_jitter)))),
            max(0.0, min(1.0, g + float(rng.uniform(-color_jitter, color_jitter)))),
            max(0.0, min(1.0, b + float(rng.uniform(-color_jitter, color_jitter)))),
            a,
        )
    return spec
