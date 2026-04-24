"""
Aperture (inner bore) detector for female RF connector images.

Once the hex is located (see hex_detector) and we have pixels-per-mm via its
known flat-to-flat size, we can measure the inner aperture in pixels and
convert to mm. The aperture diameter *is* the connector class:

  3.5 mm female → ~3.5 mm aperture
  2.92 mm female → ~2.92 mm aperture
  2.4 mm female → ~2.4 mm aperture

Approach: look for the darkest circular region near a provided search center
(which will be the hex centroid from the hex detector). Hough circles works
well here since the aperture presents as a high-contrast dark-on-metal ring.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ApertureDetection:
    diameter_px: float
    center: tuple[float, float]


def detect_aperture(
    image: np.ndarray,
    search_center: tuple[float, float],
    search_radius_px: float,
    min_diameter_frac: float = 0.1,
    max_diameter_frac: float = 1.6,
) -> ApertureDetection | None:
    """
    Find the aperture (dark central circle) within search_radius_px of
    search_center.

    min_diameter_frac / max_diameter_frac bound the detector's search range
    as fractions of search_radius_px (half-width of the search ROI).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    cx, cy = int(search_center[0]), int(search_center[1])
    r = int(search_radius_px)
    h, w = gray.shape
    x0 = max(0, cx - r)
    x1 = min(w, cx + r)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r)
    if x1 <= x0 or y1 <= y0:
        return None

    roi = gray[y0:y1, x0:x1]

    # Two-step detection:
    # 1. HoughCircles for a geometric circle fit
    # 2. If Hough fails, fall back to dark-threshold + largest dark contour
    min_r = int(r * min_diameter_frac / 2)
    max_r = int(r * max_diameter_frac / 2)
    min_r = max(min_r, 3)
    max_r = max(max_r, min_r + 2)

    blurred = cv2.medianBlur(roi, 5)

    # Invert so the aperture (dark) becomes bright — HoughCircles finds bright
    # circles, and we want to prefer the dark central region here.
    inverted = 255 - blurred

    circles = cv2.HoughCircles(
        inverted,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min_r * 2, 10),
        param1=80,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Pick the circle closest to the ROI center.
        roi_cx = (x1 - x0) // 2
        roi_cy = (y1 - y0) // 2
        best = min(
            circles,
            key=lambda c: (c[0] - roi_cx) ** 2 + (c[1] - roi_cy) ** 2,
        )
        bx, by, br = best
        return ApertureDetection(
            diameter_px=float(br * 2),
            center=(float(x0 + bx), float(y0 + by)),
        )

    # Fallback: fixed-threshold the very-dark region (< ~25% intensity), which
    # captures the aperture but rejects the hex face (typically mid-gray). This
    # is more robust than Otsu when the ROI is dominated by a gray hex body.
    dark_threshold = 60
    _, dark_mask = cv2.threshold(blurred, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the contour closest to ROI center, not the largest — the aperture is
    # central in the ROI by construction.
    roi_cx = (x1 - x0) // 2
    roi_cy = (y1 - y0) // 2
    def _dist_to_center(c):
        m = cv2.moments(c)
        if m["m00"] == 0:
            return float("inf")
        cxp = m["m10"] / m["m00"]
        cyp = m["m01"] / m["m00"]
        return (cxp - roi_cx) ** 2 + (cyp - roi_cy) ** 2

    contour = min(contours, key=_dist_to_center)
    area = cv2.contourArea(contour)
    if area < 20:
        return None

    (bx, by), br = cv2.minEnclosingCircle(contour)
    return ApertureDetection(
        diameter_px=float(br * 2),
        center=(float(x0 + bx), float(y0 + by)),
    )
