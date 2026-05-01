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
    Find a hex coupling-nut contour in `image`.

    Two-stage approach for robustness on phone shots with cluttered
    backgrounds:

      1. Localize the connector face circle via Hough — the bright,
         well-defined mating face is the most reliable anchor.
      2. Crop to ~3x the face radius around the circle's center, then
         run hex detection on that small clean crop with adaptive
         thresholding (handles wood-grain backgrounds the original
         Otsu approach would lump together with the connector).

    Falls back to whole-image Otsu detection if Hough finds no circle
    (preserves the original behavior for clean bench shots).

    Input: BGR or RGB uint8 array of shape (H, W, 3), or grayscale (H, W).
    Returns a HexDetection or None if no hex is found.
    """
    if image.ndim == 3:
        bgr = image  # treat any 3-channel input as BGR for cv2 functions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image

    # Stage 1: try to localize via Hough Circle.
    h, w = gray.shape[:2]
    short = min(h, w)
    blurred_for_hough = cv2.medianBlur(gray, 7)
    min_r = max(15, int(short * 0.03))   # smaller min radius — phone shots
    max_r = max(min_r + 1, int(short * 0.45))
    circles = cv2.HoughCircles(
        blurred_for_hough, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=max_r,
        param1=80, param2=22,
        minRadius=min_r, maxRadius=max_r,
    )

    candidate_anchors: list[tuple[int, int, int]] = []
    if circles is not None:
        for cx, cy, r in circles[0][:4]:
            candidate_anchors.append((int(round(cx)), int(round(cy)), int(round(r))))

    # Always include the "no anchor" fallback so a clean bench shot
    # without obvious face circle still passes through Otsu.
    # When we DO have a face anchor, also enforce the physical
    # constraint: hex must encircle the face, so flat_to_flat must
    # be at least ~1.4x the face diameter and at most ~3.0x.
    best: HexDetection | None = None
    best_score = -1.0

    for anchor in candidate_anchors + [None]:
        if anchor is None:
            crop_gray = gray
            x_off, y_off = 0, 0
        else:
            cx, cy, r = anchor
            # 3x face radius gives plenty of room for the hex coupling nut.
            half = int(r * 3)
            x0 = max(0, cx - half); x1 = min(w, cx + half)
            y0 = max(0, cy - half); y1 = min(h, cy + half)
            crop_gray = gray[y0:y1, x0:x1]
            x_off, y_off = x0, y0
            if crop_gray.size == 0:
                continue

        det = _detect_hex_on_crop(crop_gray)
        if det is None:
            continue

        # Translate the detection back to the original image coords.
        if x_off or y_off:
            det = HexDetection(
                flat_to_flat_px=det.flat_to_flat_px,
                center=(det.center[0] + x_off, det.center[1] + y_off),
                vertices_px=det.vertices_px + np.array([[x_off, y_off]], dtype=np.float32),
                bounding_box=(det.bounding_box[0] + x_off, det.bounding_box[1] + y_off,
                              det.bounding_box[2], det.bounding_box[3]),
            )

        # Hard physical constraints when we have a face anchor:
        # the hex must encircle the face. Reject any candidate whose
        # flat_to_flat is < 1.3x face diameter (impossible — hex is
        # bigger than the face) or > 3.0x (way too big — likely a
        # background object or photo border).
        if anchor is not None:
            cx, cy, r = anchor
            offset = ((det.center[0] - cx) ** 2 + (det.center[1] - cy) ** 2) ** 0.5
            ratio = det.flat_to_flat_px / max(1.0, 2 * r)
            if ratio < 1.3 or ratio > 3.0:
                continue
            if offset > r * 1.0:
                continue   # not concentric — different object
            score = det.flat_to_flat_px * 2.0   # prefer hex anchored to a face
        else:
            score = det.flat_to_flat_px

        if score > best_score:
            best_score = score
            best = det

    return best


def _detect_hex_on_crop(gray: np.ndarray) -> HexDetection | None:
    """The original Otsu + adaptive-fallback hex detection, run on either
    the full image or a localized crop around a Hough-found face circle."""
    blurred = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # Try Otsu first (works on bench shots), then adaptive threshold (works
    # on phone shots with cluttered backgrounds).
    thresholds = []
    _, otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    thresholds.append(otsu)
    block = max(11, (min(gray.shape[:2]) // 8) | 1)   # odd
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, 5,
    )
    thresholds.append(adaptive)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    best: HexDetection | None = None
    best_area = 0.0
    for binary in thresholds:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

        for contour in contours_sorted:
            area = cv2.contourArea(contour)
            if area < 200:
                continue

            peri = cv2.arcLength(contour, closed=True)
            for epsilon_factor in (0.02, 0.03, 0.04, 0.05, 0.06):
                approx = cv2.approxPolyDP(contour, epsilon_factor * peri, closed=True)
                # Accept 6-vertex polygons strictly, but also 5- and 7-vertex
                # ones if they're "hex-like" (high circularity, convex).
                # Phone shots often produce one extra/missing vertex from
                # noise on a single edge.
                n_vertices = len(approx)
                if not cv2.isContourConvex(approx):
                    continue
                if n_vertices == 6:
                    pass  # ideal
                elif n_vertices in (5, 7):
                    # Hex circularity ≈ 0.907; reject anything below 0.78.
                    circularity = 4.0 * np.pi * area / max(1.0, peri * peri)
                    if circularity < 0.78:
                        continue
                else:
                    continue

                rect = cv2.minAreaRect(contour)
                (cx, cy), (w, h), _ = rect
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
                break

    return best
