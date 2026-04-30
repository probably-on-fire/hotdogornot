"""
Phone-friendly capture demo for the pitch.

Workflow on a phone browser:
  1. Tap "Take photo" → mobile camera opens
  2. Snap a connector → image POSTs to /rfcai/predict, gets back per-detection
     class + confidence + bbox; rendered with bounding boxes overlaid.
  3. Either tap "✓ correct" (submits to /rfcai/uploads with the predicted
     class) or pick the correct class manually (submits with correction)
  4. Either way, the image enters the continuous-learning loop.

Two endpoints are used:
  - POST /rfcai/predict — detect+crop+classify per frame (live inference)
  - POST /rfcai/uploads — store labeled training data (active learning)

This page is the closest thing to the eventual AR app inline-correction
UX without writing any Unity.

Auth note: the device token is read from RFCAI_DEVICE_TOKEN env var. The
demo URL itself should be behind nginx basic auth.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st


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


def _predict_via_relay(image_bytes: bytes) -> tuple[dict | None, str]:
    """POST the image to /rfcai/predict. Returns (response_dict, error_msg)."""
    if not DEVICE_TOKEN:
        return None, "RFCAI_DEVICE_TOKEN not configured on the demo server"
    try:
        resp = requests.post(
            f"{RELAY_URL}/predict",
            headers={"X-Device-Token": DEVICE_TOKEN},
            files=[("image", ("capture.jpg", image_bytes, "image/jpeg"))],
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Relay returned {resp.status_code}: {resp.text[:200]}"
    except requests.RequestException as e:
        return None, f"Network error: {e}"


def _submit_to_relay(image_bytes: bytes, claimed_class: str, capture_reason: str) -> tuple[bool, str]:
    """POST the image to /rfcai/uploads. Returns (ok, message_for_user)."""
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


def _draw_bboxes(image_bytes: bytes, predict_response: dict) -> bytes:
    """Render bounding boxes + class labels on the original image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None: return image_bytes
    for p in predict_response.get("predictions", []):
        b = p["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        # Pad the bbox visualization a bit since the detector returns the
        # tight contour box; the actual classifier saw a padded crop.
        pad = int(max(w, h) * 0.4)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(bgr.shape[1], x + w + pad)
        y1 = min(bgr.shape[0], y + h + pad)
        color = (0, 200, 50)  # green BGR
        cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 4)
        label = f"{p['class_name']} {p['confidence']:.0%}"
        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(bgr, (x0, y0 - th - 12), (x0 + tw + 8, y0), color, -1)
        cv2.putText(bgr, label, (x0 + 4, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else image_bytes


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
    st.session_state.predict_response = None
    st.session_state.predict_error = None

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
    if st.session_state.capture_bytes != raw:
        st.session_state.capture_bytes = raw
        with st.spinner("Sending to /rfcai/predict…"):
            resp, err = _predict_via_relay(raw)
        st.session_state.predict_response = resp
        st.session_state.predict_error = err

if st.session_state.capture_bytes is not None:
    if st.session_state.predict_error:
        st.error(f"Prediction failed: {st.session_state.predict_error}")
        st.image(st.session_state.capture_bytes, caption="captured", use_container_width=True)
    elif st.session_state.predict_response is not None:
        resp = st.session_state.predict_response
        n = len(resp.get("predictions", []))
        if n == 0:
            st.warning("No connector detected in the image. Try a clearer mating-face photo.")
            st.image(st.session_state.capture_bytes, caption="captured", use_container_width=True)
        else:
            # Render bounding boxes + class labels on the image
            annotated = _draw_bboxes(st.session_state.capture_bytes, resp)
            st.image(annotated, caption=f"{n} detection(s) over {resp['image_width']}x{resp['image_height']}", use_container_width=True)
            st.markdown("### Detections")
            for i, p in enumerate(resp["predictions"]):
                cn = p["class_name"]
                cf = p["confidence"]
                bar = "█" * int(cf * 20)
                st.write(f"**{i+1}.** `{cn}` — {cf:.0%}  {bar}")

    if (st.session_state.predict_response is not None
            and len(st.session_state.predict_response.get("predictions", [])) > 0):
        pred_class = st.session_state.predict_response["predictions"][0]["class_name"]
    else:
        pred_class = "Unknown"

    st.divider()
    st.markdown("### Help us train")
    st.caption(
        "Tap ✓ if the top prediction is right, or pick the correct class. "
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
        disabled=correction == "(use prediction)" and pred_class == "Unknown",
    )
    col2.button("Reset", on_click=lambda: st.session_state.update(
        capture_bytes=None, predict_response=None, predict_error=None,
    ), use_container_width=True)

    if confirm:
        if correction == "(use prediction)":
            chosen_class = pred_class
            reason = "auto_confirmed"
        else:
            chosen_class = correction
            reason = "user_corrected"

        if chosen_class == "Unknown":
            st.error("Pick a class from the dropdown first.")
        else:
            with st.spinner("Sending to /rfcai/uploads…"):
                ok, msg = _submit_to_relay(
                    image_bytes=st.session_state.capture_bytes,
                    claimed_class=chosen_class,
                    capture_reason=reason,
                )
            if ok:
                st.success(msg)
                st.balloons()
                st.session_state.capture_bytes = None
                st.session_state.predict_response = None
                st.session_state.predict_error = None
            else:
                st.error(msg)

# Footer / debug expander
with st.expander("Debug info"):
    st.write(f"Relay: `{RELAY_URL}`")
    st.write(f"Token configured: {'yes' if DEVICE_TOKEN else 'NO — predictions + uploads will fail'}")
    st.write("Predict endpoint: `POST /predict`")
    st.write("Upload endpoint: `POST /uploads`")
    if st.session_state.get("predict_response"):
        st.write("**Last raw /predict response:**")
        st.json(st.session_state.predict_response)
