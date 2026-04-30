"""
Auto-detect connector positions in a video frame and return tight crops.

Most of our M+F training videos show 1-2 connectors on a wood surface. The
connectors are bright metallic blobs on a much darker matte background, so
a simple Otsu-thresholded contour search picks them out reliably without
any ML.

Public surface:

    detect_connector_crops(frame_bgr, max_crops=4) -> list[CropResult]

Each CropResult contains the bounding box, a tight crop (with padding),
and a center coordinate. Used by the Streamlit Video Labeler to give the
user one labeling decision per detected connector instead of per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CropResult:
    crop: np.ndarray           # BGR uint8, padded square crop around the connector
    bbox: tuple[int, int, int, int]   # x, y, w, h on the source frame
    center: tuple[int, int]    # cx, cy on the source frame
    area_px: int


def detect_connector_crops(
    frame_bgr: np.ndarray,
    min_area_frac: float = 0.001,
    max_area_frac: float = 0.10,
    pad_frac: float = 0.35,
    max_crops: int = 4,
) -> list[CropResult]:
    """
    Find bright metallic blobs on the frame and return padded square crops.

    `min_area_frac` / `max_area_frac` filter contours by their fraction of the
    total frame area — drops noise (too small) and continuous bright regions
    like the wood surface (too big). Defaults are tuned for typical phone
    captures of small connectors on a wood bench.

    Crops are square with the connector centered, padded by `pad_frac * size`
    so the user sees enough context to identify M vs F at a glance.
    """
    h, w = frame_bgr.shape[:2]
    total = h * w
    min_area = total * min_area_frac
    max_area = total * max_area_frac

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu finds the metal-vs-wood split fairly reliably.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Dilate-then-erode (close) to bridge the small dark gaps inside the
    # connector body (e.g., the dark hex shadow lines).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[CropResult] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        # Drop highly-elongated contours (likely wood-grain lines).
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if aspect > 2.5:
            continue
        # Build a square padded crop centered on the contour.
        side = int(max(cw, ch) * (1 + 2 * pad_frac))
        cx = x + cw // 2
        cy = y + ch // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        # Re-anchor if we hit an edge so the crop stays square if possible.
        if x1 - x0 < side: x0 = max(0, x1 - side)
        if y1 - y0 < side: y0 = max(0, y1 - side)
        crop = frame_bgr[y0:y1, x0:x1].copy()
        candidates.append(CropResult(
            crop=crop, bbox=(x, y, cw, ch), center=(cx, cy), area_px=int(area),
        ))

    # Keep the largest N crops (likely the actual connectors, not background blobs).
    candidates.sort(key=lambda r: -r.area_px)
    return candidates[:max_crops]
