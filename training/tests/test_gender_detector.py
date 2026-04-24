import math

import numpy as np
from PIL import Image, ImageDraw

from rfconnectorai.measurement.gender_detector import detect_gender


def _render_face_with_pin_protrusion(
    image_size: int = 400,
    aperture_px: int = 80,
    pin_px: int = 20,
    has_pin: bool = True,
) -> np.ndarray:
    """
    Render a mating face with or without a visible pin. For the male case,
    the pin appears as a bright central disc; for female, the aperture center
    is dark (no pin visible through the socket).
    """
    img = Image.new("RGB", (image_size, image_size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    cx = cy = image_size // 2

    # Hex body
    draw.polygon(
        [
            (cx + 100, cy),
            (cx + 50, cy + 87),
            (cx - 50, cy + 87),
            (cx - 100, cy),
            (cx - 50, cy - 87),
            (cx + 50, cy - 87),
        ],
        fill=(90, 90, 90),
    )

    # Aperture (dark bore)
    r = aperture_px / 2.0
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(25, 25, 25))

    if has_pin:
        # Male: bright pin in the center
        pr = pin_px / 2.0
        draw.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=(240, 210, 140))

    return np.array(img)


def test_detects_male_from_bright_central_pin():
    img = _render_face_with_pin_protrusion(has_pin=True)
    result = detect_gender(img, aperture_center=(200, 200), aperture_radius_px=40)
    assert result.gender == "male", f"expected male, got {result.gender}"


def test_detects_female_from_empty_aperture_center():
    img = _render_face_with_pin_protrusion(has_pin=False)
    result = detect_gender(img, aperture_center=(200, 200), aperture_radius_px=40)
    assert result.gender == "female"


def test_center_brightness_higher_for_male():
    male = _render_face_with_pin_protrusion(has_pin=True)
    female = _render_face_with_pin_protrusion(has_pin=False)
    rm = detect_gender(male, (200, 200), 40)
    rf = detect_gender(female, (200, 200), 40)
    assert rm.center_brightness > rf.center_brightness
