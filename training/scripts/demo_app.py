"""
Streamlit demo for the RF connector measurement pipeline.

Run:
    cd training
    .venv/Scripts/python.exe -m streamlit run scripts/demo_app.py

A browser opens on http://localhost:8501. Upload a connector photo (or
take one with a webcam), see the measurement pipeline output: detected hex,
detected aperture, detected ArUco marker (if any), family + gender brightness
signals, and the predicted connector class.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from rfconnectorai.measurement.aperture_detector import detect_aperture
from rfconnectorai.measurement.aruco_detector import detect_aruco_marker
from rfconnectorai.measurement.class_predictor import predict_class
from rfconnectorai.measurement.family_detector import detect_family
from rfconnectorai.measurement.gender_detector import detect_gender
from rfconnectorai.measurement.hex_detector import detect_hex


st.set_page_config(page_title="AIRED — RF Connector Identifier", layout="wide")
st.title("AIRED — RF Connector Identifier")
st.caption(
    "Upload a frontal mating-face photo of an RF connector. The app detects the hex, "
    "aperture, dielectric, and pin/socket, and predicts the connector class. "
    "Add a 25 mm ArUco scale marker in the frame for higher precision-size accuracy."
)

with st.sidebar:
    st.header("Capture options")
    aruco_size = st.number_input(
        "ArUco marker physical size (mm)", min_value=5.0, max_value=100.0, value=25.0, step=0.5
    )
    show_overlays = st.checkbox("Show detection overlays", value=True)
    st.divider()
    st.markdown(
        "**Tips for a good photo**\n"
        "- Hold the camera roughly perpendicular to the mating face\n"
        "- Make sure the hex coupling nut is fully visible\n"
        "- Use even lighting, plain background\n"
        "- Place the printed ArUco marker on the same surface for scale"
    )

uploaded = st.file_uploader(
    "Upload a connector photo (JPG/PNG)", type=["jpg", "jpeg", "png", "webp"]
)
camera = st.camera_input("...or take one with your webcam")

source = uploaded or camera
if source is None:
    st.info("Upload an image or take a photo to get started.")
    st.stop()

# Load image
img_bytes = source.getvalue()
nparr = np.frombuffer(img_bytes, np.uint8)
img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
if img_bgr is None:
    st.error("Couldn't decode the image. Try a different file.")
    st.stop()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Run individual detectors so we can show intermediate state
hex_det = detect_hex(img_rgb)
aruco = detect_aruco_marker(img_rgb, marker_size_mm=float(aruco_size))
aperture = None
family = None
gender = None
if hex_det is not None:
    aperture = detect_aperture(
        img_rgb,
        search_center=hex_det.center,
        search_radius_px=hex_det.flat_to_flat_px * 0.5,
    )
    if aperture is not None:
        family = detect_family(
            img_rgb,
            aperture_center=aperture.center,
            aperture_radius_px=aperture.diameter_px / 2.0,
            pin_radius_px=aperture.diameter_px / 2.0 * 0.50,
        )
        gender = detect_gender(
            img_rgb,
            aperture_center=aperture.center,
            aperture_radius_px=aperture.diameter_px / 2.0,
        )

# Full prediction
prediction = predict_class(img_rgb, aruco_marker_size_mm=float(aruco_size))

# Compose overlay image
overlay = img_rgb.copy()
if show_overlays:
    if hex_det is not None:
        # Draw hex bbox
        verts = hex_det.vertices_px.astype(np.int32)
        cv2.polylines(overlay, [verts], isClosed=True, color=(0, 200, 255), thickness=3)
        cx, cy = int(hex_det.center[0]), int(hex_det.center[1])
        cv2.circle(overlay, (cx, cy), 6, (0, 200, 255), -1)
    if aperture is not None:
        cx, cy = int(aperture.center[0]), int(aperture.center[1])
        r = int(aperture.diameter_px / 2)
        cv2.circle(overlay, (cx, cy), r, (255, 80, 80), 3)
    if aruco is not None:
        corners = aruco.corners.astype(np.int32)
        cv2.polylines(overlay, [corners], isClosed=True, color=(120, 255, 120), thickness=3)

col1, col2 = st.columns([2, 1])
with col1:
    st.image(overlay, caption="Detections", use_container_width=True)

with col2:
    # Prediction headline
    if prediction.class_name == "Unknown":
        st.error(f"**Predicted: Unknown**")
        if prediction.reason:
            st.write(f"_{prediction.reason}_")
    else:
        st.success(f"**Predicted: {prediction.class_name}**")

    st.markdown("### Measurements")
    rows = []
    if prediction.hex_flat_to_flat_mm is not None:
        rows.append(("Hex flat-to-flat", f"{prediction.hex_flat_to_flat_mm:.2f} mm"))
    if prediction.aperture_mm is not None:
        rows.append(("Aperture diameter", f"{prediction.aperture_mm:.2f} mm"))
    if prediction.pixels_per_mm is not None:
        rows.append(("Scale (px / mm)", f"{prediction.pixels_per_mm:.1f}"))
    if prediction.family is not None:
        rows.append(("Family", prediction.family))
    if prediction.gender is not None:
        rows.append(("Gender", prediction.gender))
    if prediction.dielectric_brightness is not None:
        rows.append(("Dielectric brightness", f"{prediction.dielectric_brightness:.0f} / 255"))
    if prediction.center_brightness is not None:
        rows.append(("Center brightness", f"{prediction.center_brightness:.0f} / 255"))
    if aruco is not None:
        rows.append(("ArUco marker ID", str(aruco.marker_id)))
        rows.append(("ArUco edge", f"{aruco.edge_px:.0f} px"))

    if not rows:
        st.write("_(no measurements — pipeline failed at the first stage)_")
    else:
        for k, v in rows:
            st.write(f"**{k}:** {v}")

st.divider()

with st.expander("Detector outputs (raw)"):
    st.json({
        "hex_detected": hex_det is not None,
        "hex_flat_to_flat_px": hex_det.flat_to_flat_px if hex_det else None,
        "hex_center": [hex_det.center[0], hex_det.center[1]] if hex_det else None,
        "aperture_detected": aperture is not None,
        "aperture_diameter_px": aperture.diameter_px if aperture else None,
        "family": family.family if family else None,
        "dielectric_brightness": family.dielectric_brightness if family else None,
        "gender": gender.gender if gender else None,
        "center_brightness": gender.center_brightness if gender else None,
        "aruco_detected": aruco is not None,
        "aruco_pixels_per_mm": aruco.pixels_per_mm if aruco else None,
        "predicted_class": prediction.class_name,
        "predicted_aperture_mm": prediction.aperture_mm,
    })
