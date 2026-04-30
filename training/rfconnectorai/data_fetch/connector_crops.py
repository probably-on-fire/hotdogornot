"""
Auto-detect connector positions in a video frame and return tight crops.

Connectors on a wood bench (our typical capture environment) defeat naive
brightness thresholding because the wood is bright too — Otsu lumps wood
and metal into one giant white blob. We use **edge density** instead:
metallic connectors have sharp hex outlines, dark shadow grooves, and
specular highlights that all produce strong local edges, while wood
surfaces are smooth with low local-edge magnitude.

Public surface:

    detect_connector_crops(frame_bgr, max_crops=4) -> list[CropResult]

Each CropResult contains the bounding box, a tight padded square crop,
and a center coordinate. Used by the Streamlit Video Labeler at training
time and the /predict endpoint at inference time, so the same detector
produces the data the classifier sees in both phases.
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
    min_area_frac: float = 0.0005,
    max_area_frac: float = 0.10,
    pad_frac: float = 0.35,
    max_crops: int = 4,
    edge_threshold_std: float = 2.0,
    min_circularity: float = 0.0,
) -> list[CropResult]:
    """
    Find connectors via local edge density. Returns padded square crops.

    Algorithm:
      1. Laplacian magnitude per pixel — highlights edges (hex outline,
         shadow grooves, specular highlights).
      2. Local sum of edge magnitudes via box filter — region-density map.
      3. Threshold at mean + 2*std of the density map; morphologically
         close so a connector's clustered edges merge into one blob.
      4. Filter contours by area + aspect ratio (drops wood-grain lines
         and noise).

    Window sizes scale with frame dimensions so the same detector works
    on phone video (1080p) and high-res photos (4K) without retuning.
    """
    h, w = frame_bgr.shape[:2]
    total = h * w
    min_area = total * min_area_frac
    max_area = total * max_area_frac

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge_mag = np.abs(lap).astype(np.float32)

    # Box-filter the edge magnitude — high values mean edge-dense regions.
    # Window scales with frame size; |1 makes it odd as box filters expect.
    window = max(11, int(min(h, w) * 0.02) | 1)
    local_edge = cv2.boxFilter(edge_mag, -1, (window, window))

    mean_e = float(local_edge.mean())
    std_e = float(local_edge.std())
    threshold = mean_e + edge_threshold_std * std_e
    mask = (local_edge > threshold).astype(np.uint8) * 255

    close_size = max(7, int(min(h, w) * 0.012) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[CropResult] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if aspect > 2.5:
            continue
        # Circularity = 4π·area / perimeter² — perfect circle is 1.0,
        # connectors are 0.6–0.9, wood-grain blobs are usually <0.4.
        if min_circularity > 0.0:
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue
        side = int(max(cw, ch) * (1 + 2 * pad_frac))
        cx = x + cw // 2
        cy = y + ch // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        if x1 - x0 < side: x0 = max(0, x1 - side)
        if y1 - y0 < side: y0 = max(0, y1 - side)
        crop = frame_bgr[y0:y1, x0:x1].copy()
        candidates.append(CropResult(
            crop=crop, bbox=(x, y, cw, ch), center=(cx, cy), area_px=int(area),
        ))

    candidates.sort(key=lambda r: -r.area_px)
    return candidates[:max_crops]
