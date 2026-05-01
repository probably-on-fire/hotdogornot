"""
Geometry-grounded class predictor for RF connector images.

Full 8-class pipeline that combines four independent, training-free detectors:

  1. hex_detector          → coupling-nut flat-to-flat in pixels
  2. aperture_detector     → inner bore diameter in pixels
  3. family_detector       → SMA (PTFE dielectric visible) vs precision (air)
  4. gender_detector       → male (pin visible in center) vs female (socket recessed)

From those four measurements we uniquely identify the connector class. Every
step is interpretable; failures are labeled with reasons.

Pipeline for a frontal mating-face image:

    hex → pixels-per-mm           (scale calibration, no ambiguity once hex found)
    aperture → mm                 (outer-conductor inner diameter at the face)
    family (sma | precision)      (from bore annular brightness)
    gender (male | female)        (from aperture central brightness)
    class = build from (family, size-bucket(aperture_mm), gender)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rfconnectorai.measurement.aperture_detector import detect_aperture
from rfconnectorai.measurement.aruco_detector import detect_aruco_marker
from rfconnectorai.measurement.family_detector import detect_family
from rfconnectorai.measurement.gender_detector import detect_gender
from rfconnectorai.measurement.hex_detector import detect_hex


# Single hex flat-to-flat (mm) used as the absolute scale reference.
# Per the actual hardware in use, all four connector families take the
# same wrench — the hex grip area is constant across SMA / 3.5mm /
# 2.92mm / 2.4mm. So we don't need to enumerate hex hypotheses; one
# measurement → one pixels-per-mm.
HEX_SIZE_MM = 7.94   # 5/16 inch flats
HEX_TOLERANCE_MM = 0.8

# Precision-connector size buckets (mm). The 2.92 vs 2.4 boundary is
# geometrically ambiguous in monocular images without absolute scale (the
# aperture/hex ratios are nearly identical). Loose tolerances let borderline
# correct cases through; the predictor emits Unknown when no bucket matches.
# The fundamental ambiguity will need an ArUco marker or LiDAR depth to fully
# resolve in the field.
PRECISION_SIZE_BUCKETS = {
    "3.5mm":  (3.50, 0.45),
    "2.92mm": (2.92, 0.40),
    "2.4mm":  (2.40, 0.35),
}

# SMA aperture is larger because PTFE fills the bore; the "aperture" we
# measure is the outer-conductor bore, nominal ~4.2 mm on SMA.
SMA_APERTURE_MM = 4.20
SMA_APERTURE_TOLERANCE_MM = 0.7


@dataclass
class Prediction:
    class_name: str
    hex_flat_to_flat_mm: float | None = None
    aperture_mm: float | None = None
    pixels_per_mm: float | None = None
    family: str | None = None
    gender: str | None = None
    dielectric_brightness: float | None = None
    center_brightness: float | None = None
    reason: str = ""


def predict_class(
    image: np.ndarray,
    assumed_pixels_per_mm: float | None = None,
    aruco_marker_size_mm: float | None = 25.0,
    require_aruco: bool = False,
) -> Prediction:
    """
    Full 8-class geometry-grounded prediction. Returns Prediction with
    class_name ∈ {SMA-M, SMA-F, 3.5mm-M, 3.5mm-F, 2.92mm-M, 2.92mm-F,
    2.4mm-M, 2.4mm-F, Unknown}.

    Scale-resolution priority:
      1. `assumed_pixels_per_mm` if explicitly provided
      2. ArUco marker in frame (uses `aruco_marker_size_mm` to convert)
      3. Hex-hypothesis enumeration (only when require_aruco=False)

    Path 2 resolves the 2.92/2.4 ambiguity that path 3 cannot. Setting
    require_aruco=True turns Path 3 off and returns Unknown when no marker
    is in frame — this is the high-accuracy mode used by the AR app and
    the video-frame averager.
    """
    # Scale prior: try ArUco first; falls back to hex-hypothesis enumeration.
    if assumed_pixels_per_mm is None and aruco_marker_size_mm is not None:
        aruco = detect_aruco_marker(image, marker_size_mm=aruco_marker_size_mm)
        if aruco is not None:
            assumed_pixels_per_mm = aruco.pixels_per_mm

    if require_aruco and assumed_pixels_per_mm is None:
        return Prediction(
            class_name="Unknown",
            reason="ArUco marker required but not detected",
        )

    hex_det = detect_hex(image)
    if hex_det is None:
        return Prediction(class_name="Unknown", reason="no hex detected")

    aperture_det = detect_aperture(
        image,
        search_center=hex_det.center,
        search_radius_px=hex_det.flat_to_flat_px * 0.5,
    )
    if aperture_det is None:
        return Prediction(
            class_name="Unknown",
            reason="hex detected but aperture not detected",
        )

    # Scale calibration: hex grip is constant across all 4 connector
    # families (same wrench fits all) so one hex measurement → one
    # pixels-per-mm. Only the ArUco override changes the scale.
    if assumed_pixels_per_mm is not None:
        hypotheses = [(hex_det.flat_to_flat_px / assumed_pixels_per_mm, assumed_pixels_per_mm)]
    else:
        hypotheses = [(HEX_SIZE_MM, hex_det.flat_to_flat_px / HEX_SIZE_MM)]

    # The pin (or socket) occupies roughly 30-45% of the aperture radius
    # depending on the family. Excluding 50% from the family-brightness
    # measurement makes the metric robust to bright male pins bleeding into
    # the annulus.
    family_det = detect_family(
        image,
        aperture_center=aperture_det.center,
        aperture_radius_px=aperture_det.diameter_px / 2.0,
        pin_radius_px=aperture_det.diameter_px / 2.0 * 0.50,
    )

    gender_det = detect_gender(
        image,
        aperture_center=aperture_det.center,
        aperture_radius_px=aperture_det.diameter_px / 2.0,
    )

    best: Prediction | None = None
    best_delta = float("inf")

    for hex_ff_mm, ppm in hypotheses:
        aperture_mm = aperture_det.diameter_px / ppm

        # Hex is constant; we don't gate on hex size at all. The
        # aperture in mm directly maps to a class (with family from
        # dielectric brightness as a tiebreaker for the SMA / 3.5mm
        # case where face/aperture sizes are nearly identical).
        if family_det.family == "sma":
            delta = abs(aperture_mm - SMA_APERTURE_MM)
            if delta > SMA_APERTURE_TOLERANCE_MM:
                continue
            candidate = f"SMA-{'M' if gender_det.gender == 'male' else 'F'}"
            if delta < best_delta:
                best_delta = delta
                best = _build_prediction(
                    candidate, hex_ff_mm, aperture_mm, ppm, family_det, gender_det
                )
        else:
            for size_label, (nominal_mm, tolerance_mm) in PRECISION_SIZE_BUCKETS.items():
                delta = abs(aperture_mm - nominal_mm)
                if delta > tolerance_mm:
                    continue
                candidate = f"{size_label}-{'M' if gender_det.gender == 'male' else 'F'}"
                if delta < best_delta:
                    best_delta = delta
                    best = _build_prediction(
                        candidate, hex_ff_mm, aperture_mm, ppm, family_det, gender_det
                    )

    if best is None:
        return Prediction(
            class_name="Unknown",
            family=family_det.family,
            gender=gender_det.gender,
            dielectric_brightness=family_det.dielectric_brightness,
            center_brightness=gender_det.center_brightness,
            reason="no (hex, aperture, family) hypothesis fits a known class",
        )

    return best


def _build_prediction(
    class_name: str,
    hex_ff_mm: float,
    aperture_mm: float,
    ppm: float,
    family_det,
    gender_det,
) -> Prediction:
    return Prediction(
        class_name=class_name,
        hex_flat_to_flat_mm=hex_ff_mm,
        aperture_mm=aperture_mm,
        pixels_per_mm=ppm,
        family=family_det.family,
        gender=gender_det.gender,
        dielectric_brightness=family_det.dielectric_brightness,
        center_brightness=gender_det.center_brightness,
    )
