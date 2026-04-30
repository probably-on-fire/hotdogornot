"""
Detect + crop + classify FastAPI service.

Runs on the training box (where torch + classifier weights live), exposed
to the public via a reverse SSH tunnel through aired.com — see
deploy/systemd/rfcai-predict.service + rfcai-predict-tunnel.service.

Single endpoint:

    POST /predict   (X-Device-Token required)
         multipart: image=<JPEG/PNG>
         response:
           {
             "image_width": 1920,
             "image_height": 1080,
             "predictions": [
               {
                 "class_name": "2.4mm-M",
                 "confidence": 0.83,
                 "probabilities": {"SMA-M": 0.01, ...},
                 "bbox": {"x": 612, "y": 415, "w": 240, "h": 240}
               }
             ]
           }

The phone or web client sends a full camera frame; we run the same
connector_crops detector the labeler uses, classify each detected blob
against the trained ResNet-18, and return per-detection results.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from rfconnectorai.classifier.predict import ConnectorClassifier
from rfconnectorai.data_fetch.connector_crops import detect_connector_crops


DEFAULT_MODEL_DIR = Path("./models/connector_classifier")
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # 25 MB; phone frames usually 1-3 MB


def _config_from_env() -> dict:
    return {
        "model_dir": Path(os.environ.get("RFCAI_MODEL_DIR", DEFAULT_MODEL_DIR)),
        "device_token": os.environ.get("RFCAI_DEVICE_TOKEN", ""),
        "max_upload_bytes": int(os.environ.get("RFCAI_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)),
        "max_detections": int(os.environ.get("RFCAI_MAX_DETECTIONS", 4)),
    }


def create_app(config: dict | None = None) -> FastAPI:
    cfg = config or _config_from_env()
    model_dir: Path = cfg["model_dir"]
    device_token: str = cfg["device_token"]
    max_upload_bytes: int = cfg["max_upload_bytes"]
    max_detections: int = cfg["max_detections"]

    app = FastAPI(title="RF Connector AI prediction service", version="1.0.0")

    # Load classifier once at boot so each request is fast.
    classifier: ConnectorClassifier | None = None
    if (model_dir / "weights.pt").exists() and (model_dir / "labels.json").exists():
        try:
            classifier = ConnectorClassifier.load(model_dir)
        except Exception as e:
            # Don't fail to start — endpoint surfaces the error per-request.
            print(f"[predict_service] classifier load failed: {e}")

    def require_token(x_device_token: str = Header(None)) -> str:
        if not device_token:
            raise HTTPException(status_code=503, detail="server token not configured")
        if x_device_token != device_token:
            raise HTTPException(status_code=401, detail="invalid device token")
        return x_device_token

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "classifier_loaded": classifier is not None,
            "max_detections": max_detections,
        }

    @app.post("/predict")
    async def predict(
        image: UploadFile = File(...),
        _: str = Depends(require_token),
    ):
        if classifier is None:
            raise HTTPException(status_code=503, detail="classifier not loaded yet")

        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty image")
        if len(data) > max_upload_bytes:
            raise HTTPException(status_code=413, detail="image too large")

        # Decode straight to BGR; opencv handles JPEG / PNG / WebP.
        nparr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="couldn't decode image")
        h, w = bgr.shape[:2]

        # Stage 1: detect connector blobs and get tight padded crops.
        crops = detect_connector_crops(bgr, max_crops=max_detections)

        # Stage 2: classify each crop. Classifier expects RGB.
        predictions = []
        for c in crops:
            rgb_crop = cv2.cvtColor(c.crop, cv2.COLOR_BGR2RGB)
            pred = classifier.predict(rgb_crop)
            x, y, bw, bh = c.bbox
            predictions.append({
                "class_name": pred.class_name,
                "confidence": float(pred.confidence),
                "probabilities": {k: float(v) for k, v in pred.probabilities.items()},
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
            })

        return JSONResponse({
            "image_width": w,
            "image_height": h,
            "predictions": predictions,
        })

    return app


# Module-level for `uvicorn rfconnectorai.server.predict_service:app`
app = create_app()
