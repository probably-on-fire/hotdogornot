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


def detect_connector_crops_hough(
    frame_bgr: np.ndarray,
    min_radius_frac: float = 0.04,
    max_radius_frac: float = 0.25,
    pad_frac: float = 0.6,
    max_crops: int = 4,
    accumulator_threshold: int = 35,
) -> list[CropResult]:
    """
    Find connectors by Hough circle detection. RF connector mating faces
    are explicitly circular, so looking for circles beats edge-density
    heuristics on textured backgrounds (wood grain has linear/random edges,
    not circular ones).

    pad_frac defaults to 0.6 — wider than the edge-density detector's 0.35
    so the crop shows the coupling nut and a bit of context around the
    mating face, not just a tight crop of the face itself.
    """
    h, w = frame_bgr.shape[:2]
    short_side = min(h, w)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    min_r = max(8, int(short_side * min_radius_frac))
    max_r = max(min_r + 1, int(short_side * max_radius_frac))

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max_r,            # don't double-count overlapping circles
        param1=80,                # upper Canny edge threshold
        param2=accumulator_threshold,  # center-vote threshold; lower = more circles
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []

    # circles shape: (1, N, 3) — (x, y, r). Sort by accumulator strength
    # (HoughCircles returns strongest first already).
    circles = circles[0]
    candidates: list[CropResult] = []
    for cx, cy, r in circles[:max_crops]:
        cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
        side = int(2 * r * (1 + 2 * pad_frac))
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        if x1 - x0 < side: x0 = max(0, x1 - side)
        if y1 - y0 < side: y0 = max(0, y1 - side)
        crop = frame_bgr[y0:y1, x0:x1].copy()
        candidates.append(CropResult(
            crop=crop,
            bbox=(cx - r, cy - r, 2 * r, 2 * r),
            center=(cx, cy),
            area_px=int(np.pi * r * r),
        ))
    return candidates


def detect_connector_crops(
    frame_bgr: np.ndarray,
    min_area_frac: float = 0.0005,
    max_area_frac: float = 0.10,
    pad_frac: float = 0.35,
    max_crops: int = 4,
    edge_threshold_std: float = 2.0,
    min_circularity: float = 0.0,
    detect_max_dim: int = 1280,
) -> list[CropResult]:
    """
    Find connectors via local edge density. Returns padded square crops
    from the original full-resolution frame.

    Algorithm:
      1. Downsample frame to <= detect_max_dim on the long edge so
         Laplacian + boxFilter + contour work scales with megapixels,
         not just gigabytes of RAM. Crops still come from the
         original-resolution frame so the classifier gets sharp input.
      2. Laplacian magnitude per pixel — highlights edges (hex outline,
         shadow grooves, specular highlights).
      3. Local sum of edge magnitudes via box filter — region-density map.
      4. Threshold at mean + 2*std of the density map; morphologically
         close so a connector's clustered edges merge into one blob.
      5. Filter contours by area + aspect ratio (drops wood-grain lines
         and noise).

    On a 12 MP phone shot the downsample-for-detection step cuts this
    function from ~1700ms to ~200ms with no measurable accuracy
    change on the holdout (crops are still cut from the original
    frame at full resolution).
    """
    full_h, full_w = frame_bgr.shape[:2]
    long_edge = max(full_h, full_w)
    if long_edge > detect_max_dim:
        scale = detect_max_dim / long_edge
        det_frame = cv2.resize(
            frame_bgr,
            (int(round(full_w * scale)), int(round(full_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        det_frame = frame_bgr

    h, w = det_frame.shape[:2]
    total = h * w
    min_area = total * min_area_frac
    max_area = total * max_area_frac

    gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
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
        # Detection ran on the downsampled frame; map bbox back to the
        # original frame so crops are cut at full resolution.
        if scale != 1.0:
            x = int(round(x / scale))
            y = int(round(y / scale))
            cw = int(round(cw / scale))
            ch = int(round(ch / scale))
        side = int(max(cw, ch) * (1 + 2 * pad_frac))
        cx = x + cw // 2
        cy = y + ch // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(full_w, x0 + side)
        y1 = min(full_h, y0 + side)
        if x1 - x0 < side: x0 = max(0, x1 - side)
        if y1 - y0 < side: y0 = max(0, y1 - side)
        crop = frame_bgr[y0:y1, x0:x1].copy()
        # Area_px reported in the *original* frame so callers comparing
        # to image area get the right number.
        original_area = int(area / (scale * scale)) if scale != 1.0 else int(area)
        candidates.append(CropResult(
            crop=crop, bbox=(x, y, cw, ch), center=(cx, cy),
            area_px=original_area,
        ))

    candidates.sort(key=lambda r: -r.area_px)
    return candidates[:max_crops]


def detect_connector_crops_yolo(
    frame_bgr: np.ndarray,
    yolo_model,
    pad_frac: float = 0.3,
    max_crops: int = 4,
    conf: float = 0.20,
) -> list[CropResult]:
    """Fallback detector for cases where Hough finds no circles —
    YOLO sees connectors at non-perpendicular angles where the face
    isn't a clean circle. Wider crops than Hough since YOLO already
    captures the full connector.

    `yolo_model` is an already-loaded `ultralytics.YOLO` instance.
    Caller is responsible for the lifecycle (load once at service
    startup, reuse per request).
    """
    h, w = frame_bgr.shape[:2]
    results = yolo_model.predict(frame_bgr, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    candidates: list[CropResult] = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        bw, bh = x2 - x1, y2 - y1
        # Pad outward to a square, same shape contract as Hough crops.
        side = int(max(bw, bh) * (1 + 2 * pad_frac))
        cx = x1 + bw // 2
        cy = y1 + bh // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x_end = min(w, x0 + side)
        y_end = min(h, y0 + side)
        if x_end - x0 < side: x0 = max(0, x_end - side)
        if y_end - y0 < side: y0 = max(0, y_end - side)
        crop = frame_bgr[y0:y_end, x0:x_end].copy()
        candidates.append(CropResult(
            crop=crop,
            bbox=(x1, y1, bw, bh),
            center=(cx, cy),
            area_px=int(bw * bh),
        ))
    candidates.sort(key=lambda r: -r.area_px)
    return candidates[:max_crops]

