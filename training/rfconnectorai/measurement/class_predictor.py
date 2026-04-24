"""
Geometry-grounded class predictor for RF connector images.

Combines hex detection and aperture detection to produce a class label. The
pipeline:

  1. Find the hex.  Its flat-to-flat size in pixels + known-in-mm gives
     pixels-per-mm calibration.
  2. Figure out which hex standard we're looking at — 6.35 mm (1/4 in)
     indicates 2.4 mm family; 7.94 mm (5/16 in) indicates 3.5/2.92 mm family.
     The hex alone tells us which family.
  3. Measure the aperture in pixels at the hex center, convert to mm.
  4. Threshold the aperture in mm to pick the specific class.

All distances are physical (mm), which means the predictor is robust across
scale, rotation, and moderate perspective — as long as the hex is visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rfconnectorai.measurement.aperture_detector import detect_aperture
from rfconnectorai.measurement.hex_detector import detect_hex


# Known hex flat-to-flat sizes for the precision connector families.
HEX_SIZE_MM = {
    "sma_precision_large": 7.94,  # 5/16 inch — 3.5 mm and 2.92 mm coupling nuts
    "sma_precision_small": 6.35,  # 1/4 inch — 2.4 mm coupling nut
}

# Tolerance for matching a measured hex size to a known hex standard (mm).
HEX_TOLERANCE_MM = 0.8

# Nominal aperture diameters per class (female).
CLASS_APERTURE_MM = {
    "3.5mm-F": 3.5,
    "2.92mm-F": 2.92,
    "2.4mm-F": 2.4,
}

# How far an aperture measurement can be from the nominal before we stop
# accepting the class as matching.
APERTURE_TOLERANCE_MM = 0.45


@dataclass
class Prediction:
    class_name: str
    hex_flat_to_flat_mm: float | None = None
    aperture_mm: float | None = None
    pixels_per_mm: float | None = None
    reason: str = ""


# Map class → required hex size (the hex family each class uses).
CLASS_TO_HEX_MM = {
    "3.5mm-F":  HEX_SIZE_MM["sma_precision_large"],
    "2.92mm-F": HEX_SIZE_MM["sma_precision_large"],
    "2.4mm-F":  HEX_SIZE_MM["sma_precision_small"],
}


def predict_class(
    image: np.ndarray,
    assumed_pixels_per_mm: float | None = None,
) -> Prediction:
    """
    Predict the connector class from a single image.

    Strategy: try each plausible hex hypothesis (7.94 mm or 6.35 mm). For
    each hypothesis, compute the implied pixels-per-mm, convert the aperture
    pixel size to mm, and see which connector class best fits. Pick the
    hypothesis + class whose aperture mismatch is smallest and whose hex
    family is consistent with the matched class.

    `assumed_pixels_per_mm`, when provided, overrides the hex-hypothesis
    search — useful when the capture pipeline has independent scale info.
    """
    hex_det = detect_hex(image)
    if hex_det is None:
        return Prediction(class_name="Unknown", reason="no hex detected")

    search_radius = hex_det.flat_to_flat_px * 0.5
    aperture = detect_aperture(
        image,
        search_center=hex_det.center,
        search_radius_px=search_radius,
    )
    if aperture is None:
        return Prediction(
            class_name="Unknown",
            reason="hex detected but aperture not detected",
        )

    # Hypotheses to consider: either the user-supplied scale, or each of the
    # two standard hex sizes.
    if assumed_pixels_per_mm is not None:
        hypotheses = [(hex_det.flat_to_flat_px / assumed_pixels_per_mm, assumed_pixels_per_mm)]
    else:
        hypotheses = [
            (hex_mm, hex_det.flat_to_flat_px / hex_mm)
            for hex_mm in HEX_SIZE_MM.values()
        ]

    best: Prediction | None = None
    best_delta = float("inf")

    for hex_ff_mm, ppm in hypotheses:
        aperture_mm = aperture.diameter_px / ppm
        for class_name, nominal_ap_mm in CLASS_APERTURE_MM.items():
            # Hex family must match the class.
            expected_hex_mm = CLASS_TO_HEX_MM[class_name]
            if abs(hex_ff_mm - expected_hex_mm) > HEX_TOLERANCE_MM:
                continue
            delta = abs(aperture_mm - nominal_ap_mm)
            if delta < best_delta:
                best_delta = delta
                best = Prediction(
                    class_name=class_name,
                    hex_flat_to_flat_mm=hex_ff_mm,
                    aperture_mm=aperture_mm,
                    pixels_per_mm=ppm,
                )

    if best is None or best_delta > APERTURE_TOLERANCE_MM:
        return Prediction(
            class_name="Unknown",
            aperture_mm=(best.aperture_mm if best else None),
            hex_flat_to_flat_mm=(best.hex_flat_to_flat_mm if best else None),
            pixels_per_mm=(best.pixels_per_mm if best else None),
            reason=f"no class within {APERTURE_TOLERANCE_MM} mm aperture tolerance",
        )

    return best


def _closest_hex_standard(hex_flat_to_flat_px: float) -> tuple[float, float]:
    """
    Given a hex measurement in pixels, try both standard hex sizes and pick
    the one that gives a plausible pixels-per-mm for a typical phone-captured
    connector photo (roughly 20–80 px/mm).

    Returns (hex_flat_to_flat_mm, pixels_per_mm).
    """
    candidates = []
    for name, hex_mm in HEX_SIZE_MM.items():
        ppm = hex_flat_to_flat_px / hex_mm
        # Prefer candidates whose implied pixels-per-mm falls in a plausible
        # range. If both plausible, we'll disambiguate at the aperture step
        # by returning both and letting the class-fit decide — but here we
        # just pick one to produce a deterministic first pass.
        candidates.append((hex_mm, ppm, abs(ppm - 50)))  # 50 px/mm midpoint heuristic

    candidates.sort(key=lambda t: t[2])
    best_hex_mm, best_ppm, _ = candidates[0]
    return best_hex_mm, best_ppm
