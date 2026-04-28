"""
Phone-friendly capture demo for the pitch.

Workflow on a phone browser:
  1. Tap "Take photo" → mobile camera opens
  2. Snap a connector → ensemble prediction shown immediately
  3. Either tap "✓ correct" (auto-submits with predicted class)
     or pick the correct class manually (submits with corrected class)
  4. Image goes to the relay's /rfcai/uploads endpoint, kicks off the
     continuous-learning loop just like the AR app would.

This page is the closest thing to the eventual AR app inline-correction
UX without writing any Unity. It exercises the full backend loop end to
end — relay → sync → daemon → labeled folder → eventual retrain.

Auth note: the device token is read from RFCAI_DEVICE_TOKEN env var. The
demo URL itself should be behind nginx basic auth so random visitors
can't poison the training set.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st

from rfconnectorai.ensemble import EnsemblePredictor


REPO_TRAINING = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_TRAINING / "models" / "connector_classifier"

CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]

RELAY_URL = os.environ.get("RFCAI_RELAY_URL", "https://aired.com/rfcai")
DEVICE_TOKEN = os.environ.get("RFCAI_DEVICE_TOKEN", "")


@st.cache_resource(show_spinner=False)
def _load_predictor() -> EnsemblePredictor:
    classifier_dir = DEFAULT_MODEL_DIR if (DEFAULT_MODEL_DIR / "weights.pt").exists() else None
    return EnsemblePredictor.load(classifier_dir) if classifier_dir else EnsemblePredictor(classifier=None)


def _submit_to_relay(image_bytes: bytes, claimed_class: str, capture_reason: str) -> tuple[bool, str]:
    """POST the image to the relay. Returns (ok, message_for_user)."""
    if not DEVICE_TOKEN:
        return False, "RFCAI_DEVICE_TOKEN not configured on the demo server"
    try:
        resp = requests.post(
            f"{RELAY_URL}/uploads",
            headers={"X-Device-Token": DEVICE_TOKEN},
            data={
                "claimed_class": claimed_class,
                "device_id": "demo-streamlit",
                "capture_reason": capture_reason,
            },
            files=[("frames", ("capture.jpg", image_bytes, "image/jpeg"))],
            timeout=30,
        )
        if resp.status_code == 200:
            blob = resp.json()
            return True, f"Sent for training as `{claimed_class}` (upload_id={blob['upload_id'][:8]}…)"
        return False, f"Relay returned {resp.status_code}: {resp.text[:200]}"
    except requests.RequestException as e:
        return False, f"Network error: {e}"


# ---------------------------------------------------------------------------

st.set_page_config(page_title="RF Connector Demo", layout="centered")

# Mobile-friendly tweaks: bigger buttons, less chrome.
st.markdown(
    """
    <style>
      .stButton button { font-size: 1.05rem; padding: 0.75rem 1rem; }
      .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RF Connector Demo")
st.caption(
    "Take a photo of a connector mating face. The system identifies it and "
    "your confirmed answer feeds back into training so the model gets better."
)

# Persist the captured image + prediction across reruns so the user can confirm
# without re-snapping.
if "capture_bytes" not in st.session_state:
    st.session_state.capture_bytes = None
    st.session_state.prediction = None

img_input = st.camera_input(
    "Take photo (or upload)",
    label_visibility="collapsed",
    key="cam",
)
if img_input is None:
    img_input = st.file_uploader(
        "…or upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

if img_input is not None:
    raw = img_input.getvalue()
    # Only re-run prediction when the bytes actually change.
    if st.session_state.capture_bytes != raw:
        st.session_state.capture_bytes = raw
        nparr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Couldn't decode that image.")
            st.session_state.prediction = None
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with st.spinner("Predicting…"):
                predictor = _load_predictor()
                st.session_state.prediction = predictor.predict(rgb)

if st.session_state.capture_bytes is not None and st.session_state.prediction is not None:
    pred = st.session_state.prediction

    if pred.class_name == "Unknown":
        st.warning(
            "Couldn't identify the connector. Try a clearer photo of the "
            "mating face, perpendicular to the camera."
        )
    else:
        agree = pred.agreement
        confidence_label = f"{pred.confidence:.0%}"
        if agree == "agree":
            st.success(f"**{pred.class_name}**  ({confidence_label} — measurement + classifier agree)")
        elif agree == "disagree":
            st.warning(f"**{pred.class_name}**  ({confidence_label} — uncertain, please confirm)")
        elif agree == "measurement_only":
            st.info(f"**{pred.class_name}**  ({confidence_label} — measurement only)")
        elif agree == "classifier_only":
            st.info(f"**{pred.class_name}**  ({confidence_label} — classifier only)")
        else:
            st.info(f"**{pred.class_name}**  ({confidence_label})")

    st.divider()
    st.markdown("### Help us train")
    st.caption(
        "Tap ✓ if the prediction is right, or pick the correct class. "
        "Either way, your photo gets labeled and used to improve the model."
    )

    correction = st.selectbox(
        "Connector class",
        options=["(use prediction)"] + CANONICAL_CLASSES,
        index=0,
    )

    col1, col2 = st.columns(2)
    confirm = col1.button(
        "✓ Submit",
        type="primary",
        use_container_width=True,
        disabled=correction == "(use prediction)" and pred.class_name == "Unknown",
    )
    col2.button("Reset", on_click=lambda: st.session_state.update(
        capture_bytes=None, prediction=None,
    ), use_container_width=True)

    if confirm:
        if correction == "(use prediction)":
            chosen_class = pred.class_name
            reason = "auto_confirmed"
        else:
            chosen_class = correction
            reason = "user_corrected"

        if chosen_class == "Unknown":
            st.error("Pick a class from the dropdown first.")
        else:
            with st.spinner("Sending…"):
                ok, msg = _submit_to_relay(
                    image_bytes=st.session_state.capture_bytes,
                    claimed_class=chosen_class,
                    capture_reason=reason,
                )
            if ok:
                st.success(msg)
                st.balloons()
                # Reset so user can take another photo
                st.session_state.capture_bytes = None
                st.session_state.prediction = None
            else:
                st.error(msg)

# Footer / debug expander
with st.expander("Debug info"):
    st.write(f"Relay: `{RELAY_URL}`")
    st.write(f"Token configured: {'yes' if DEVICE_TOKEN else 'NO — uploads will fail'}")
    st.write(f"Classifier: `{DEFAULT_MODEL_DIR}` ({'loaded' if (DEFAULT_MODEL_DIR / 'weights.pt').exists() else 'measurement-only'})")
