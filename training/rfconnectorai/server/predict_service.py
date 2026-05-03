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
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rfconnectorai.classifier.predict import ConnectorClassifier
from rfconnectorai.data_fetch.connector_crops import detect_connector_crops
from rfconnectorai.server.labeler import create_router as create_labeler_router


DEFAULT_MODEL_DIR = Path("./models/connector_classifier")
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # 25 MB; phone frames usually 1-3 MB
DEFAULT_MAX_VIDEO_BYTES = 200 * 1024 * 1024   # 200 MB; phone clips can be big at 4K
DEFAULT_VIDEO_FPS = 1.0                       # sample 1 frame/sec — plenty for ID
DEFAULT_VIDEO_MAX_FRAMES = 30                 # cap at ~30s of video


def _config_from_env() -> dict:
    return {
        "model_dir": Path(os.environ.get("RFCAI_MODEL_DIR", DEFAULT_MODEL_DIR)),
        "device_token": os.environ.get("RFCAI_DEVICE_TOKEN", ""),
        "max_upload_bytes": int(os.environ.get("RFCAI_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)),
        "max_video_bytes": int(os.environ.get("RFCAI_MAX_VIDEO_BYTES", DEFAULT_MAX_VIDEO_BYTES)),
        "video_fps": float(os.environ.get("RFCAI_VIDEO_FPS", DEFAULT_VIDEO_FPS)),
        "video_max_frames": int(os.environ.get("RFCAI_VIDEO_MAX_FRAMES", DEFAULT_VIDEO_MAX_FRAMES)),
        "max_detections": int(os.environ.get("RFCAI_MAX_DETECTIONS", 4)),
    }


def create_app(config: dict | None = None) -> FastAPI:
    cfg = config or _config_from_env()
    model_dir: Path = cfg["model_dir"]
    device_token: str = cfg["device_token"]
    max_upload_bytes: int = cfg["max_upload_bytes"]
    max_video_bytes: int = cfg["max_video_bytes"]
    video_fps: float = cfg["video_fps"]
    video_max_frames: int = cfg["video_max_frames"]
    max_detections: int = cfg["max_detections"]

    app = FastAPI(title="RF Connector AI prediction service", version="1.0.0")

    # CORS: allow the Flutter web build (and any future browser-side
    # client) to call /predict and /labeler/* from a different origin.
    # Auth is via header (X-Device-Token / Basic), so the browser sends
    # a preflight OPTIONS — without this it gets blocked.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # internal R&D tool; tighten if exposed publicly
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

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

    def _classify_frame(bgr: np.ndarray) -> list[dict]:
        """Detect + classify on one BGR frame, return prediction dicts."""
        crops = detect_connector_crops(bgr, max_crops=max_detections)
        out = []
        for c in crops:
            rgb_crop = cv2.cvtColor(c.crop, cv2.COLOR_BGR2RGB)
            pred = classifier.predict(rgb_crop)
            x, y, bw, bh = c.bbox
            out.append({
                "class_name": pred.class_name,
                "confidence": float(pred.confidence),
                "probabilities": {k: float(v) for k, v in pred.probabilities.items()},
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
            })
        return out

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

        return JSONResponse({
            "image_width": w,
            "image_height": h,
            "predictions": _classify_frame(bgr),
        })

    @app.post("/predict-video")
    async def predict_video(
        video: UploadFile = File(...),
        _: str = Depends(require_token),
    ):
        """Sample a video at video_fps, classify each frame, return the
        single highest-confidence prediction across all frames + bbox /
        frame index. Frame extraction uses ffmpeg via imageio_ffmpeg."""
        if classifier is None:
            raise HTTPException(status_code=503, detail="classifier not loaded yet")

        ext = Path(video.filename or "").suffix.lower()
        if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            raise HTTPException(status_code=400, detail=f"unsupported video extension {ext!r}")
        data = await video.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty video")
        if len(data) > max_video_bytes:
            raise HTTPException(status_code=413, detail="video too large")

        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ffmpeg not available: {e}")

        best: dict | None = None
        best_frame_idx: int = -1
        frames_scanned = 0
        img_w = img_h = 0
        with tempfile.TemporaryDirectory(prefix="predict_video_") as tmp:
            tmpp = Path(tmp)
            src = tmpp / f"src{ext}"
            src.write_bytes(data)
            # Hard frame cap via -frames:v keeps long uploads bounded.
            subprocess.run(
                [ff, "-y", "-i", str(src),
                 "-vf", f"fps={video_fps}", "-frames:v", str(video_max_frames),
                 "-q:v", "4", str(tmpp / "f_%04d.jpg")],
                capture_output=True, check=False,
            )
            frames = sorted(tmpp.glob("f_*.jpg"))
            for i, fp in enumerate(frames):
                bgr = cv2.imread(str(fp))
                if bgr is None:
                    continue
                if img_w == 0:
                    img_h, img_w = bgr.shape[:2]
                frames_scanned += 1
                for p in _classify_frame(bgr):
                    if best is None or p["confidence"] > best["confidence"]:
                        best = p
                        best_frame_idx = i

        return JSONResponse({
            "image_width": img_w,
            "image_height": img_h,
            "predictions": [best] if best else [],
            "frames_scanned": frames_scanned,
            "best_frame_index": best_frame_idx,
        })

    # Mount the HTMX-driven training-data labeler at /labeler/.
    # Reachable at https://aired.com/rfcai/labeler/ via the existing
    # /rfcai/* nginx wildcard. Auth is HTTP Basic via LABELER_USER /
    # LABELER_PASS env vars (separate from the device-token auth on
    # /predict and /uploads).
    app.include_router(create_labeler_router())

    return app


# Module-level for `uvicorn rfconnectorai.server.predict_service:app`
app = create_app()
