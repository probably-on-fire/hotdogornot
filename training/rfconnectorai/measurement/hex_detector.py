"""
Hex (coupling nut) detector for RF connector images.

The coupling nut on SMA, 3.5mm, 2.92mm, and 2.4mm precision connectors has a
known-size hex flat-to-flat dimension standardized by the connector series:

  3.5 mm and 2.92 mm female:  5/16 inch (7.94 mm) hex
  2.4 mm female:              1/4 inch  (6.35 mm) hex

Detecting the hex in image pixels gives us a known physical reference, which
we can use to scale the *inner aperture* measurement into millimeters. The
aperture then maps directly to class.

Approach: find the dominant 6-sided polygon in the image by thresholding +
contour detection + polygon approximation. We measure flat-to-flat as the
shorter side of the minimum-area bounding rectangle — for a regular hexagon
oriented in any direction, this is the true flat-to-flat dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HexDetection:
    """Result of a hex detection on an image."""

    flat_to_flat_px: float
    center: tuple[float, float]
    vertices_px: np.ndarray  # (6, 2) — approximated vertex positions
    bounding_box: tuple[int, int, int, int]  # x, y, w, h


def detect_hex(image: np.ndarray) -> HexDetection | None:
    """
    Find the largest 6-sided polygon in `image`.

    Input: BGR or RGB uint8 array of shape (H, W, 3), or grayscale (H, W).
    Returns a HexDetection or None if no hex is found.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Bilateral + Otsu gives robust thresholding for product-photo backgrounds.
    blurred = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Morphological close fills in small bright holes (specular highlights, etc.)
    # so the hex contour traces the true outer boundary instead of detouring
    # around bright spots. Real metallic connectors have specular reflections
    # in nearly every photo, so this matters even more on real captures.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Search the top few largest contours for a hex; the largest isn't always
    # the right one (photo borders etc. can dominate).
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    best: HexDetection | None = None
    best_area = 0.0
    for contour in contours_sorted:
        area = cv2.contourArea(contour)
        # Reject contours that are too small to be the connector body.
        if area < 200:
            continue

        peri = cv2.arcLength(contour, closed=True)
        # Try a range of epsilon values — the right one depends on shape
        # complexity and image noise.
        for epsilon_factor in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(contour, epsilon_factor * peri, closed=True)
            if len(approx) == 6 and cv2.isContourConvex(approx):
                # Measure flat-to-flat via minAreaRect's short side, which is
                # invariant to rotation and gives the apothem × 2 for a regular
                # hexagon regardless of pointy-top vs flat-top orientation.
                rect = cv2.minAreaRect(contour)
                (cx, cy), (w, h), angle = rect
                flat_to_flat = min(w, h)

                x, y, bw, bh = cv2.boundingRect(contour)
                det = HexDetection(
                    flat_to_flat_px=float(flat_to_flat),
                    center=(float(cx), float(cy)),
                    vertices_px=approx.reshape(-1, 2).astype(np.float32),
                    bounding_box=(int(x), int(y), int(bw), int(bh)),
                )
                if area > best_area:
                    best_area = area
                    best = det
                break  # Found hex at this contour; no need to try other epsilons

    return best
