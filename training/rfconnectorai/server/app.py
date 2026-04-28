"""
Relay-server FastAPI app.

Sits between the AR app and the training machine. Three responsibilities:

  1. Accept frame uploads from the app (POST /uploads). Writes them into
     `incoming/<upload_id>/` where the ingestion daemon picks them up.
  2. Advertise the current model version (GET /model/version). The app
     polls this on launch to decide whether to download a fresh model.
  3. Serve the current model weights + labels as static files (GET
     /model/weights, GET /model/labels), so the app can pull them when
     a new version is published.

The server itself is stateless. It reads the model manifest off disk
each time, and writes uploads to a directory. All persistence happens
on the filesystem the relay shares with the training machine.

Auth: a single shared device token in the X-Device-Token header. Trivial
on purpose — for an internal R&D pitch, this is enough. Swap for proper
per-device tokens later.

Configuration (env vars, see deploy notes):

    RFCAI_INCOMING_DIR        directory uploads land in (default ./incoming)
    RFCAI_MODEL_DIR           directory the trained model lives in
                              (default ./models/connector_classifier)
    RFCAI_DEVICE_TOKEN        shared secret the app must send
    RFCAI_MAX_UPLOAD_BYTES    per-upload size cap (default 100MB)

Run:

    uvicorn rfconnectorai.server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse


DEFAULT_INCOMING = Path("./incoming")
DEFAULT_MODEL_DIR = Path("./models/connector_classifier")
DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024   # 100 MB
READY_SENTINEL = ".ready"
MANIFEST_FILENAME = "manifest.json"
LABELS_FILENAME = "labels.json"


def _config_from_env() -> dict:
    return {
        "incoming_dir": Path(os.environ.get("RFCAI_INCOMING_DIR", DEFAULT_INCOMING)),
        "model_dir": Path(os.environ.get("RFCAI_MODEL_DIR", DEFAULT_MODEL_DIR)),
        "device_token": os.environ.get("RFCAI_DEVICE_TOKEN", ""),
        "max_upload_bytes": int(os.environ.get("RFCAI_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)),
    }


def create_app(config: dict | None = None) -> FastAPI:
    """
    Build the FastAPI app. `config` lets tests inject a temp-dir setup
    without setting env vars; production reads env via _config_from_env.
    """
    cfg = config or _config_from_env()
    incoming_dir: Path = cfg["incoming_dir"]
    model_dir: Path = cfg["model_dir"]
    device_token: str = cfg["device_token"]
    max_upload_bytes: int = cfg["max_upload_bytes"]

    incoming_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="RF Connector AI relay",
        version="1.0.0",
    )

    # --- auth dependency ---------------------------------------------------

    def require_token(x_device_token: str = Header(None)) -> str:
        # If no token is configured server-side, fail closed — never serve
        # an unauthenticated server in production.
        if not device_token:
            raise HTTPException(status_code=503, detail="server token not configured")
        if x_device_token != device_token:
            raise HTTPException(status_code=401, detail="invalid device token")
        return x_device_token

    # --- model version endpoints ------------------------------------------

    def _read_manifest() -> dict | None:
        p = model_dir / MANIFEST_FILENAME
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    @app.get("/model/version")
    def get_model_version():
        manifest = _read_manifest()
        if manifest is None:
            return JSONResponse({"version": 0}, status_code=200)
        return {"version": int(manifest.get("version", 0))}

    @app.get("/model/latest")
    def get_model_latest(_: str = Depends(require_token)):
        """Return enough info for the app to fetch the current model."""
        manifest = _read_manifest()
        if manifest is None:
            raise HTTPException(status_code=404, detail="no model published yet")
        return {
            "version": int(manifest.get("version", 0)),
            "weights_filename": manifest.get("weights_filename"),
            "labels_filename": manifest.get("labels_filename", LABELS_FILENAME),
            "weights_sha256": manifest.get("weights_sha256"),
            "labels_sha256": manifest.get("labels_sha256"),
            "trained_at": manifest.get("trained_at"),
            "weights_url": "/model/weights",
            "labels_url": "/model/labels",
        }

    @app.get("/model/weights")
    def get_model_weights(_: str = Depends(require_token)):
        manifest = _read_manifest()
        if manifest is None:
            raise HTTPException(status_code=404, detail="no model published yet")
        weights = model_dir / manifest["weights_filename"]
        if not weights.exists():
            raise HTTPException(status_code=500,
                                detail=f"manifest references missing file {weights.name}")
        return FileResponse(
            weights,
            media_type="application/octet-stream",
            filename=weights.name,
        )

    @app.get("/model/labels")
    def get_model_labels(_: str = Depends(require_token)):
        labels = model_dir / LABELS_FILENAME
        if not labels.exists():
            raise HTTPException(status_code=404, detail="labels.json not yet written")
        return FileResponse(labels, media_type="application/json")

    # --- upload endpoint --------------------------------------------------

    @app.post("/uploads")
    async def post_upload(
        request: Request,
        claimed_class: str = Form(...),
        device_id: str = Form(...),
        capture_reason: str = Form("low_confidence"),    # "manual" | "low_confidence"
        frames: list[UploadFile] = File(...),
        _: str = Depends(require_token),
    ):
        if not frames:
            raise HTTPException(status_code=400, detail="no frames in upload")
        if len(frames) > 200:
            raise HTTPException(status_code=400,
                                detail=f"too many frames ({len(frames)}; max 200)")

        upload_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_") + os.urandom(3).hex()

        # Stage to a temp dir first so a partial write never leaves a
        # half-formed upload visible to the daemon. We only move it into
        # incoming/ once everything is on disk + the .ready sentinel exists.
        with tempfile.TemporaryDirectory(prefix="rfcai_upload_") as tmp:
            staging = Path(tmp) / upload_id
            staging.mkdir()
            total_bytes = 0
            for i, fh in enumerate(frames):
                if fh.filename is None:
                    continue
                ext = Path(fh.filename).suffix.lower() or ".jpg"
                if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                data = await fh.read()
                total_bytes += len(data)
                if total_bytes > max_upload_bytes:
                    raise HTTPException(status_code=413,
                                        detail="upload exceeds size limit")
                (staging / f"frame_{i:03d}{ext}").write_bytes(data)

            manifest = {
                "upload_id": upload_id,
                "claimed_class": claimed_class,
                "device_id": device_id,
                "capture_reason": capture_reason,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "n_frames": sum(1 for _ in staging.iterdir()),
            }
            (staging / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
            # Ready sentinel touched LAST. Daemon waits on it.
            (staging / READY_SENTINEL).write_text("")

            # Move atomically into incoming/ so the daemon doesn't see partial state.
            target = incoming_dir / upload_id
            shutil.move(str(staging), str(target))

        return {
            "upload_id": upload_id,
            "n_frames_received": manifest["n_frames"],
            "claimed_class": claimed_class,
        }

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "model_version": (_read_manifest() or {}).get("version", 0)}

    return app


# Module-level app instance for `uvicorn rfconnectorai.server.app:app`.
app = create_app()
