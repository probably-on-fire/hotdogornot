"""
Scene-setup helpers for bpy rendering. Keeps scene logic separate from
the render function so it's easier to test and tweak.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RenderConfig:
    image_size: int = 384
    samples: int = 32                       # Cycles samples per pixel
    camera_distance_m: float = 0.15         # default camera distance from connector center
    camera_distance_jitter_m: float = 0.05
    camera_fov_deg: float = 40.0
    camera_fov_jitter_deg: float = 8.0
    rotation_range_rad: float = 0.8         # how far to rotate the connector on each axis
    depth_of_field: bool = True             # enable DoF for realism


@dataclass
class DomainRandomizationConfig:
    hdri_dir: Path | None = None            # path to directory of .exr/.hdr files
    hdri_energy_range: tuple[float, float] = (0.5, 3.0)
    background_strength: float = 1.0
    motion_blur: bool = False
    jpeg_quality_range: tuple[int, int] = (70, 95)
    noise_std_range: tuple[float, float] = (0.0, 0.02)
