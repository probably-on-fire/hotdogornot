"""
Gender detector: male vs female.

The male plug has a pin protruding along the axis; when viewed head-on at the
mating face, the pin appears as a bright metallic disc in the aperture center.
The female jack has a recessed socket; the aperture center reads as dark.

We measure mean luminance in a small central disc (inside the aperture but
on-axis with the pin) and threshold.

Like the family detector, this is no-training and interpretable; the only
parameter is a brightness threshold that we tune against real photos.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GenderDetection:
    gender: str                 # "male" | "female"
    center_brightness: float    # 0..255 mean luminance of the central disc
    reason: str = ""


# Tuned on synthetic renders; revisit with real photos.
BRIGHTNESS_THRESHOLD = 110.0
CENTER_DISC_FRACTION = 0.3   # fraction of aperture radius used for the central disc


def detect_gender(
    image: np.ndarray,
    aperture_center: tuple[float, float],
    aperture_radius_px: float,
) -> GenderDetection:
    """
    Measure mean brightness in a small disc at the aperture center and
    classify as male (bright → pin present) or female (dark → socket).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    h, w = gray.shape
    cx, cy = aperture_center
    r_center = max(2.0, aperture_radius_px * CENTER_DISC_FRACTION)

    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = r2 <= r_center ** 2

    if not mask.any():
        return GenderDetection(
            gender="female",
            center_brightness=0.0,
            reason="central disc mask degenerate",
        )

    mean_brightness = float(gray[mask].mean())
    gender = "male" if mean_brightness >= BRIGHTNESS_THRESHOLD else "female"
    return GenderDetection(
        gender=gender,
        center_brightness=mean_brightness,
    )
