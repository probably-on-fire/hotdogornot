"""
Family detector: SMA vs precision (air-dielectric).

Discriminates via the brightness of the annular region between the aperture
edge and the inner-conductor pin at the mating face.

  SMA:       PTFE dielectric fills the bore → annular region is bright (whitish)
  Precision: air-dielectric → annular region is dark (hollow)

This is a robust, interpretable signal that requires no training. The only
parameter is a brightness threshold, which we'll tune once real photos exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FamilyDetection:
    family: str            # "sma" | "precision"
    dielectric_brightness: float   # 0..255 mean luminance in the annular bore region
    reason: str = ""


# Tuned on synthetic renders; revisit with real photos.
BRIGHTNESS_THRESHOLD = 120.0


def detect_family(
    image: np.ndarray,
    aperture_center: tuple[float, float],
    aperture_radius_px: float,
    pin_radius_px: float,
) -> FamilyDetection:
    """
    Measure mean brightness in the annular region (between pin_radius_px and
    aperture_radius_px from the aperture center) and classify the family.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    h, w = gray.shape
    cx, cy = aperture_center

    # Build an annular mask: inside aperture, outside pin.
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    annulus = (r2 <= aperture_radius_px ** 2) & (r2 >= pin_radius_px ** 2)

    if not annulus.any():
        return FamilyDetection(
            family="precision",  # default when the annulus is degenerate
            dielectric_brightness=0.0,
            reason="annular region degenerate",
        )

    mean_brightness = float(gray[annulus].mean())
    family = "sma" if mean_brightness >= BRIGHTNESS_THRESHOLD else "precision"
    return FamilyDetection(
        family=family,
        dielectric_brightness=mean_brightness,
    )
