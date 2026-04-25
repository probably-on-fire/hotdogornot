"""
Streamlit demo for the RF connector identifier.

Run:
    cd training
    .venv/Scripts/python.exe -m streamlit run scripts/demo_app.py

A browser opens on http://localhost:8501. Upload a connector photo (or
take one with a webcam), see the ensemble prediction: measurement pipeline
(hex / aperture / ArUco / family / gender / class) + ResNet-18 classifier
prediction + agreement signal.

If no classifier is trained yet (no weights at models/connector_classifier/),
the demo falls back to measurement-only mode.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from rfconnectorai.ensemble import EnsemblePredictor
from rfconnectorai.measurement.aperture_detector import detect_aperture
from rfconnectorai.measurement.aruco_detector import detect_aruco_marker
from rfconnectorai.measurement.class_predictor import predict_class
from rfconnectorai.measurement.family_detector import detect_family
from rfconnectorai.measurement.gender_detector import detect_gender
from rfconnectorai.measurement.hex_detector import detect_hex


REPO_TRAINING = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_TRAINING / "models" / "connector_classifier"


@st.cache_resource(show_spinner=False)
def _load_predictor(model_dir_str: str | None) -> EnsemblePredictor:
    """Cache the loaded classifier across reruns so we don't reload weights
    on every interaction."""
    if model_dir_str is None:
        return EnsemblePredictor(classifier=None)
    return EnsemblePredictor.load(Path(model_dir_str))


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
    require_aruco = st.checkbox(
        "Require ArUco for class prediction",
        value=False,
        help="Strict mode — eliminates 2.92mm vs 2.4mm ambiguity, refuses prediction if no marker.",
    )

    st.divider()
    st.markdown("### Classifier")
    use_classifier = st.checkbox(
        "Use trained classifier (ensemble mode)",
        value=DEFAULT_MODEL_DIR.exists() and (DEFAULT_MODEL_DIR / "weights.pt").exists(),
        help="Cross-checks the measurement pipeline against the ResNet-18 classifier.",
    )
    if use_classifier:
        if not (DEFAULT_MODEL_DIR / "weights.pt").exists():
            st.warning(
                f"No trained model at `{DEFAULT_MODEL_DIR}`. Use the Train Classifier "
                "page to train one first."
            )
            use_classifier = False

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

# Full prediction — ensemble (measurement + classifier) when enabled.
predictor = _load_predictor(str(DEFAULT_MODEL_DIR) if use_classifier else None)
ensemble_result = predictor.predict(
    img_rgb,
    require_aruco=require_aruco,
    aruco_marker_size_mm=float(aruco_size),
)
prediction = ensemble_result.measurement   # keep variable name for downstream code

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
    # Headline reflects the ensemble result, not just the measurement output.
    headline_class = ensemble_result.class_name
    headline_conf = ensemble_result.confidence

    agreement_msg = {
        "agree":              "measurement + classifier agree",
        "disagree":           "measurement and classifier DISAGREE — recapture recommended",
        "measurement_only":   "measurement only (classifier off or failed)",
        "classifier_only":    "classifier only (measurement could not fire)",
        "neither":            "neither pipeline could fire",
    }.get(ensemble_result.agreement, ensemble_result.agreement)

    if headline_class == "Unknown":
        st.error(f"**Predicted: Unknown** — {ensemble_result.reason or agreement_msg}")
    elif ensemble_result.agreement == "disagree":
        st.warning(
            f"**Predicted: {headline_class}** "
            f"(confidence {headline_conf:.0%} — {agreement_msg})"
        )
        st.caption(ensemble_result.reason)
    else:
        st.success(
            f"**Predicted: {headline_class}** "
            f"(confidence {headline_conf:.0%} — {agreement_msg})"
        )

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

# Classifier breakdown — only meaningful if classifier was loaded.
if ensemble_result.classifier is not None:
    clf = ensemble_result.classifier
    st.markdown("### Classifier (ResNet-18) prediction")
    st.write(
        f"**{clf.class_name}** (top class, confidence {clf.confidence:.0%})"
    )
    with st.expander("Per-class probabilities"):
        for cls_name, prob in sorted(clf.probabilities.items(), key=lambda kv: -kv[1]):
            st.write(f"- {cls_name}: {prob:.3f}")

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
        "ensemble_class": ensemble_result.class_name,
        "ensemble_confidence": ensemble_result.confidence,
        "ensemble_agreement": ensemble_result.agreement,
        "classifier_class": ensemble_result.classifier.class_name if ensemble_result.classifier else None,
        "classifier_confidence": ensemble_result.classifier.confidence if ensemble_result.classifier else None,
    })
