import math

import numpy as np
from PIL import Image, ImageDraw

from rfconnectorai.measurement.family_detector import detect_family


def _render_face(
    image_size: int = 400,
    hex_ff_px: int = 200,
    aperture_px: int = 80,
    pin_px: int = 20,
    bore_interior_color: tuple[int, int, int] = (20, 20, 20),
    center: tuple[int, int] | None = None,
) -> np.ndarray:
    """Render a mating-face: dark gray hex, aperture with a given interior color, small pin."""
    img = Image.new("RGB", (image_size, image_size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    cx, cy = center if center else (image_size // 2, image_size // 2)

    # Hex
    apothem = hex_ff_px / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)
    verts = [
        (cx + circumradius * math.cos(math.radians(60 * i + 30)),
         cy + circumradius * math.sin(math.radians(60 * i + 30)))
        for i in range(6)
    ]
    draw.polygon(verts, fill=(90, 90, 90))

    # Aperture (interior color varies — the key feature for family detection)
    r = aperture_px / 2.0
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bore_interior_color)

    # Small pin in the center
    pr = pin_px / 2.0
    draw.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=(230, 200, 120))

    return np.array(img)


def test_dark_aperture_interior_classified_as_precision():
    # Air-dielectric precision connector: bore interior is dark
    img = _render_face(bore_interior_color=(20, 20, 20))
    result = detect_family(img, aperture_center=(200, 200), aperture_radius_px=40, pin_radius_px=10)
    assert result.family == "precision", f"expected precision, got {result.family}"


def test_bright_aperture_interior_classified_as_sma():
    # SMA with PTFE dielectric: bore interior is light (whitish plastic)
    img = _render_face(bore_interior_color=(230, 230, 225))
    result = detect_family(img, aperture_center=(200, 200), aperture_radius_px=40, pin_radius_px=10)
    assert result.family == "sma"


def test_brightness_score_higher_for_sma():
    dark = _render_face(bore_interior_color=(20, 20, 20))
    bright = _render_face(bore_interior_color=(230, 230, 225))
    r_dark = detect_family(dark, (200, 200), 40, 10)
    r_bright = detect_family(bright, (200, 200), 40, 10)
    assert r_bright.dielectric_brightness > r_dark.dielectric_brightness
