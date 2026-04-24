"""
ArUco marker detection for absolute scale calibration.

When the hex-anchored measurement is geometrically ambiguous (notably for
2.92mm vs 2.4mm precision connectors, whose aperture/hex ratios are nearly
identical), an operator can place a printed ArUco marker of known physical
size next to the connector. The detector returns the marker's edge length
in pixels, which combined with the printed size in mm gives an unambiguous
pixels-per-mm scale that resolves the hex hypothesis.

We use the 4×4_50 dictionary (4×4 internal grid, 50 unique IDs) — small
enough to print at 25mm without losing detectability, large enough to be
robust at typical phone capture distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ArucoDetection:
    edge_px: float          # mean side-length of the detected marker quadrilateral
    pixels_per_mm: float    # = edge_px / marker_size_mm
    marker_id: int          # which ID was detected
    corners: np.ndarray     # (4, 2) corner pixel positions, clockwise from top-left


def detect_aruco_marker(
    image: np.ndarray,
    marker_size_mm: float,
    dictionary_id: int = cv2.aruco.DICT_4X4_50,
) -> ArucoDetection | None:
    """
    Find the first ArUco marker of `dictionary_id` in `image`. Returns scale
    info or None if no marker is found.

    Picks the largest detected marker if multiple are present (largest = most
    reliable scale measurement).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    # Pick the largest marker (sum of side lengths).
    best_idx = 0
    best_perimeter = 0.0
    for i, corners in enumerate(corners_list):
        pts = corners.reshape(-1, 2)
        perimeter = sum(
            float(np.linalg.norm(pts[j] - pts[(j + 1) % 4])) for j in range(4)
        )
        if perimeter > best_perimeter:
            best_perimeter = perimeter
            best_idx = i

    pts = corners_list[best_idx].reshape(-1, 2)
    edge_lengths = [float(np.linalg.norm(pts[j] - pts[(j + 1) % 4])) for j in range(4)]
    edge_px = float(np.mean(edge_lengths))
    pixels_per_mm = edge_px / marker_size_mm

    return ArucoDetection(
        edge_px=edge_px,
        pixels_per_mm=pixels_per_mm,
        marker_id=int(ids[best_idx][0]),
        corners=pts.astype(np.float32),
    )
