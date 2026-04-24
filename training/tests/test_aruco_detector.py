import math

import cv2
import numpy as np
import pytest
from PIL import Image

from rfconnectorai.measurement.aruco_detector import detect_aruco_marker, ArucoDetection


def _make_aruco_image(image_size: int = 400, marker_size_px: int = 200, marker_id: int = 0,
                      cx: int | None = None, cy: int | None = None) -> np.ndarray:
    """Render a synthetic ArUco marker centered (or off-centered) on a white background."""
    cx = cx if cx is not None else image_size // 2
    cy = cy if cy is not None else image_size // 2

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)

    img = np.full((image_size, image_size, 3), 250, dtype=np.uint8)
    half = marker_size_px // 2
    y0 = cy - half
    x0 = cx - half
    img[y0:y0 + marker_size_px, x0:x0 + marker_size_px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return img


def test_detects_marker_at_known_size():
    img = _make_aruco_image(image_size=400, marker_size_px=200)
    result = detect_aruco_marker(img, marker_size_mm=25.0)
    assert isinstance(result, ArucoDetection)
    # marker_size_px is the side length of the rendered marker; detector should
    # recover this within ~5%.
    assert abs(result.edge_px - 200) < 200 * 0.05
    # pixels_per_mm = edge_px / marker_size_mm = 200 / 25 = 8.0
    assert abs(result.pixels_per_mm - 8.0) < 8.0 * 0.05


def test_detects_marker_off_center():
    img = _make_aruco_image(image_size=600, marker_size_px=120, cx=400, cy=200)
    result = detect_aruco_marker(img, marker_size_mm=25.0)
    assert result is not None
    assert abs(result.edge_px - 120) < 120 * 0.06


def test_returns_none_when_no_marker():
    img = np.full((400, 400, 3), 250, dtype=np.uint8)
    result = detect_aruco_marker(img, marker_size_mm=25.0)
    assert result is None


def test_pixels_per_mm_scales_with_marker_size():
    small = _make_aruco_image(image_size=400, marker_size_px=80)
    large = _make_aruco_image(image_size=400, marker_size_px=320)
    rs = detect_aruco_marker(small, marker_size_mm=25.0)
    rl = detect_aruco_marker(large, marker_size_mm=25.0)
    # Larger pixel size at same physical size = more pixels per mm
    assert rl.pixels_per_mm > rs.pixels_per_mm
    assert abs(rl.pixels_per_mm / rs.pixels_per_mm - 4.0) < 0.5
