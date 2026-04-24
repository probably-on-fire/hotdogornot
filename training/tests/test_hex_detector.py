import numpy as np
import pytest
from PIL import Image, ImageDraw

from rfconnectorai.measurement.hex_detector import detect_hex, HexDetection


def _make_hex_image(
    image_size: int = 400,
    flat_to_flat_px: int = 160,
    center: tuple[int, int] | None = None,
    rotation_deg: float = 0.0,
    hex_color: tuple[int, int, int] = (60, 60, 60),
    bg_color: tuple[int, int, int] = (240, 240, 240),
) -> np.ndarray:
    """
    Render a filled hexagon on a light background at a known flat-to-flat size.

    For a regular hexagon, flat-to-flat (apothem × 2) relates to the circumradius
    (distance from center to vertex) by: flat_to_flat = circumradius × sqrt(3).
    """
    import math

    img = Image.new("RGB", (image_size, image_size), bg_color)
    draw = ImageDraw.Draw(img)

    cx, cy = center if center else (image_size // 2, image_size // 2)
    apothem = flat_to_flat_px / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)

    # Hex pointy-top by default; we rotate by rotation_deg (plus 30° so flats
    # line up with the x-axis when rotation_deg == 0).
    verts = []
    for i in range(6):
        a = math.radians(60 * i + 30 + rotation_deg)
        x = cx + circumradius * math.cos(a)
        y = cy + circumradius * math.sin(a)
        verts.append((x, y))

    draw.polygon(verts, fill=hex_color)
    return np.array(img)


def test_detect_hex_recovers_known_flat_to_flat_size():
    img = _make_hex_image(image_size=400, flat_to_flat_px=160)
    det = detect_hex(img)
    assert isinstance(det, HexDetection)
    # Allow ±5% tolerance — contour approximation is not exact.
    assert abs(det.flat_to_flat_px - 160) < 160 * 0.05
    assert 190 < det.center[0] < 210  # centered at 200 ± 10
    assert 190 < det.center[1] < 210


def test_detect_hex_recovers_with_rotation():
    img = _make_hex_image(image_size=400, flat_to_flat_px=120, rotation_deg=17.0)
    det = detect_hex(img)
    assert abs(det.flat_to_flat_px - 120) < 120 * 0.05


def test_detect_hex_recovers_off_center_position():
    img = _make_hex_image(image_size=400, flat_to_flat_px=100, center=(150, 250))
    det = detect_hex(img)
    assert abs(det.flat_to_flat_px - 100) < 100 * 0.07
    assert abs(det.center[0] - 150) < 15
    assert abs(det.center[1] - 250) < 15


def test_detect_hex_returns_none_on_empty_image():
    img = np.full((200, 200, 3), 240, dtype=np.uint8)
    det = detect_hex(img)
    assert det is None


def test_detect_hex_handles_small_flat_to_flat():
    img = _make_hex_image(image_size=200, flat_to_flat_px=40)
    det = detect_hex(img)
    # Very small hex is where detection gets noisy; allow ±15%.
    assert det is not None
    assert abs(det.flat_to_flat_px - 40) < 40 * 0.15
