import math
import numpy as np
import pytest
from PIL import Image, ImageDraw

from rfconnectorai.measurement.class_predictor import predict_class, Prediction


def _make_connector_face(
    image_size: int,
    hex_flat_to_flat_mm: float,
    aperture_diameter_mm: float,
    pixels_per_mm: float,
) -> np.ndarray:
    """
    Render a connector face with precisely known physical dimensions at a
    chosen rendering scale. Lets us verify end-to-end that the pipeline
    recovers correct *millimeter* measurements.
    """
    img = Image.new("RGB", (image_size, image_size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    cx = cy = image_size // 2

    hex_ff_px = hex_flat_to_flat_mm * pixels_per_mm
    apothem = hex_ff_px / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)
    verts = []
    for i in range(6):
        a = math.radians(60 * i + 30)
        x = cx + circumradius * math.cos(a)
        y = cy + circumradius * math.sin(a)
        verts.append((x, y))
    draw.polygon(verts, fill=(90, 90, 90))

    ap_d_px = aperture_diameter_mm * pixels_per_mm
    r = ap_d_px / 2.0
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 15, 15))
    return np.array(img)


def test_predicts_2_4mm_from_6_35mm_hex_and_2_4mm_aperture():
    # Synthetic: 2.4mm female, 1/4" (6.35mm) hex, 2.4mm aperture
    img = _make_connector_face(
        image_size=500,
        hex_flat_to_flat_mm=6.35,
        aperture_diameter_mm=2.4,
        pixels_per_mm=30.0,
    )
    pred = predict_class(img)
    assert isinstance(pred, Prediction)
    assert pred.class_name == "2.4mm-F"
    assert abs(pred.aperture_mm - 2.4) < 0.3


def test_predicts_2_92mm_from_7_94mm_hex_and_2_92mm_aperture():
    img = _make_connector_face(
        image_size=500,
        hex_flat_to_flat_mm=7.94,
        aperture_diameter_mm=2.92,
        pixels_per_mm=25.0,
    )
    pred = predict_class(img)
    assert pred.class_name == "2.92mm-F"
    assert abs(pred.aperture_mm - 2.92) < 0.35


def test_predicts_3_5mm_from_7_94mm_hex_and_3_5mm_aperture():
    img = _make_connector_face(
        image_size=500,
        hex_flat_to_flat_mm=7.94,
        aperture_diameter_mm=3.5,
        pixels_per_mm=25.0,
    )
    pred = predict_class(img)
    assert pred.class_name == "3.5mm-F"
    assert abs(pred.aperture_mm - 3.5) < 0.35


def test_predicts_unknown_when_aperture_mismatches():
    # Unrealistic: 7.94mm hex but 5mm aperture — no real class matches.
    img = _make_connector_face(
        image_size=500,
        hex_flat_to_flat_mm=7.94,
        aperture_diameter_mm=5.0,
        pixels_per_mm=25.0,
    )
    pred = predict_class(img)
    assert pred.class_name == "Unknown"


def test_returns_unknown_when_no_hex():
    img = np.full((400, 400, 3), 240, dtype=np.uint8)
    pred = predict_class(img)
    assert pred.class_name == "Unknown"
    assert pred.reason != ""
