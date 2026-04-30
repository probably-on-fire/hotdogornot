"""
RF Connector AI demo — main page.

Workflow on a phone or laptop browser:
  1. Tap "Take photo" (or upload an image)
  2. Image POSTs to /rfcai/predict; per-detection class + confidence + bbox
     comes back, rendered as green bounding boxes on the captured image.
  3. Either tap "✓ correct" (submits to /uploads with the predicted class)
     or pick the correct class manually (submits with the user-corrected
     class as a training-data correction).
  4. Either way the image enters the continuous-learning loop.

Uses the same /predict endpoint the Unity AR app uses, so this page is a
direct test bed for the live model.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st


CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]

RELAY_URL = os.environ.get("RFCAI_RELAY_URL", "https://aired.com/rfcai")
DEVICE_TOKEN = os.environ.get("RFCAI_DEVICE_TOKEN", "")


def _predict_via_relay(image_bytes: bytes) -> tuple[dict | None, str | None]:
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
        return None, f"relay returned {resp.status_code}: {resp.text[:200]}"
    except requests.RequestException as e:
        return None, f"network error: {e}"


def _submit_to_relay(image_bytes: bytes, claimed_class: str, capture_reason: str) -> tuple[bool, str]:
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
        return False, f"relay returned {resp.status_code}: {resp.text[:200]}"
    except requests.RequestException as e:
        return False, f"network error: {e}"


def _draw_bboxes(image_bytes: bytes, predict_response: dict) -> bytes:
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None: return image_bytes
    for p in predict_response.get("predictions", []):
        b = p["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        # Pad the visualization box; the classifier saw a padded crop anyway.
        pad = int(max(w, h) * 0.4)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(bgr.shape[1], x + w + pad)
        y1 = min(bgr.shape[0], y + h + pad)
        # Color by confidence: green > 0.75, yellow > 0.5, red below.
        c = p["confidence"]
        color = (50, 200, 50) if c >= 0.75 else (10, 180, 230) if c >= 0.5 else (60, 60, 220)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 4)
        label = f"{p['class_name']} {c:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(bgr, (x0, y0 - th - 12), (x0 + tw + 8, y0), color, -1)
        cv2.putText(bgr, label, (x0 + 4, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else image_bytes


# ---------------------------------------------------------------------------

st.set_page_config(page_title="RF Connector Demo", layout="centered")
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
    "your confirmed answer feeds back into training so the model improves."
)

if "capture_bytes" not in st.session_state:
    st.session_state.capture_bytes = None
    st.session_state.predict_response = None
    st.session_state.predict_error = None

img_input = st.camera_input("Take photo (or upload)", label_visibility="collapsed", key="cam")
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
        with st.spinner("Identifying…"):
            resp, err = _predict_via_relay(raw)
        st.session_state.predict_response = resp
        st.session_state.predict_error = err

if st.session_state.capture_bytes is not None:
    if st.session_state.predict_error:
        st.error(f"Prediction failed: {st.session_state.predict_error}")
        st.image(st.session_state.capture_bytes, use_container_width=True)
        pred_class = "Unknown"
    elif st.session_state.predict_response is not None:
        resp = st.session_state.predict_response
        n = len(resp.get("predictions", []))
        if n == 0:
            st.warning("No connector detected. Try a clearer mating-face photo.")
            st.image(st.session_state.capture_bytes, use_container_width=True)
            pred_class = "Unknown"
        else:
            annotated = _draw_bboxes(st.session_state.capture_bytes, resp)
            st.image(annotated, caption=f"{n} detection(s)", use_container_width=True)
            st.markdown("### Detections")
            for i, p in enumerate(resp["predictions"]):
                bar = "█" * int(p["confidence"] * 20)
                st.write(f"**{i+1}.** `{p['class_name']}` — {p['confidence']:.0%}  {bar}")
            pred_class = resp["predictions"][0]["class_name"]
    else:
        pred_class = "Unknown"

    st.divider()
    st.markdown("### Help us train")
    st.caption("Tap ✓ if the top prediction is right, or pick the correct class.")

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
    col2.button(
        "Reset",
        on_click=lambda: st.session_state.update(
            capture_bytes=None, predict_response=None, predict_error=None,
        ),
        use_container_width=True,
    )

    if confirm:
        chosen = pred_class if correction == "(use prediction)" else correction
        reason = "auto_confirmed" if correction == "(use prediction)" else "user_corrected"
        if chosen == "Unknown":
            st.error("Pick a class from the dropdown first.")
        else:
            with st.spinner("Sending…"):
                ok, msg = _submit_to_relay(
                    image_bytes=st.session_state.capture_bytes,
                    claimed_class=chosen,
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

with st.expander("Debug"):
    st.write(f"Relay: `{RELAY_URL}`")
    st.write(f"Token configured: {'yes' if DEVICE_TOKEN else 'NO — predictions + uploads will fail'}")
    if st.session_state.get("predict_response"):
        st.json(st.session_state.predict_response)
