import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from rfconnectorai.measurement.aperture_detector import (
    detect_aperture,
    ApertureDetection,
)


def _make_hex_with_aperture(
    image_size: int = 400,
    hex_flat_to_flat_px: int = 160,
    aperture_diameter_px: int = 60,
    center: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Render a hex with a central dark circular aperture — the shape the real
    connector face presents.
    """
    img = Image.new("RGB", (image_size, image_size), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    cx, cy = center if center else (image_size // 2, image_size // 2)
    apothem = hex_flat_to_flat_px / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)

    verts = []
    for i in range(6):
        a = math.radians(60 * i + 30)
        x = cx + circumradius * math.cos(a)
        y = cy + circumradius * math.sin(a)
        verts.append((x, y))
    draw.polygon(verts, fill=(180, 180, 180))

    # Aperture — dark circle in the middle.
    r = aperture_diameter_px / 2.0
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 15, 15))

    return np.array(img)


def test_detect_aperture_recovers_known_diameter():
    img = _make_hex_with_aperture(
        image_size=400, hex_flat_to_flat_px=200, aperture_diameter_px=80
    )
    det = detect_aperture(img, search_center=(200, 200), search_radius_px=100)
    assert isinstance(det, ApertureDetection)
    assert abs(det.diameter_px - 80) < 80 * 0.1


def test_detect_aperture_handles_smaller_aperture():
    img = _make_hex_with_aperture(
        image_size=400, hex_flat_to_flat_px=200, aperture_diameter_px=40
    )
    det = detect_aperture(img, search_center=(200, 200), search_radius_px=100)
    assert abs(det.diameter_px - 40) < 40 * 0.15


def test_detect_aperture_off_center_hex():
    img = _make_hex_with_aperture(
        image_size=400,
        hex_flat_to_flat_px=160,
        aperture_diameter_px=60,
        center=(250, 150),
    )
    det = detect_aperture(img, search_center=(250, 150), search_radius_px=80)
    assert abs(det.diameter_px - 60) < 60 * 0.12
    # Centroid should be close to the hex center.
    assert abs(det.center[0] - 250) < 8
    assert abs(det.center[1] - 150) < 8


def test_detect_aperture_returns_none_when_no_dark_region():
    img = np.full((300, 300, 3), 240, dtype=np.uint8)
    det = detect_aperture(img, search_center=(150, 150), search_radius_px=80)
    assert det is None
