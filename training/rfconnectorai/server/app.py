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
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, Response


# Aired.com-side archive of uploaded training videos. The relay TEEs
# /labeler/upload-video bodies here BEFORE forwarding the original
# multipart to the training box for extraction. After this, every
# uploaded clip has two copies: one durable on aired.com (this dir,
# class-prefix named) and one on the box where extraction runs. The
# aired.com copy is the long-term archive; the box copy is a
# processing scratch space.
DEFAULT_VIDEOS_DIR = Path("/srv/rfcai/videos")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
CANONICAL_FAMILIES = ("SMA", "1.85mm", "2.4mm", "2.92mm", "3.5mm")
CANONICAL_GENDERS = ("M", "F")


def _videos_root() -> Path:
    return Path(os.environ.get("RFCAI_VIDEOS_DIR", DEFAULT_VIDEOS_DIR)).resolve()


def _class_video_filename(family: str, gender: str, ext: str,
                          when: float | None = None) -> str:
    """Mirror of labeler.py's _class_video_filename so the aired.com
    copy gets the same `<family>-<gender>_<ISO timestamp>.<ext>` name.
    `:` is replaced with `-` in the time portion since `:` isn't a
    legal filename character on every filesystem we'd ever care about."""
    ts = time.gmtime(when) if when is not None else time.gmtime()
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S", ts)
    return f"{family}-{gender}_{stamp}{ext.lower()}"


def _video_sidecar_path(video_path: Path) -> Path:
    return video_path.with_suffix(video_path.suffix + ".crops.json")


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
            "weights_onnx_filename": manifest.get("weights_onnx_filename"),
            "labels_filename": manifest.get("labels_filename", LABELS_FILENAME),
            "weights_sha256": manifest.get("weights_sha256"),
            "weights_onnx_sha256": manifest.get("weights_onnx_sha256"),
            "labels_sha256": manifest.get("labels_sha256"),
            "trained_at": manifest.get("trained_at"),
            "weights_url": "/model/weights",
            "weights_onnx_url": "/model/weights.onnx",
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

    @app.get("/model/weights.onnx")
    def get_model_weights_onnx(_: str = Depends(require_token)):
        """ONNX-format weights for the AR app's Sentis runtime."""
        manifest = _read_manifest()
        if manifest is None or not manifest.get("weights_onnx_filename"):
            raise HTTPException(status_code=404, detail="no ONNX model published yet")
        weights = model_dir / manifest["weights_onnx_filename"]
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

    # --- labeler proxy ----------------------------------------------------
    #
    # The training-data labeler runs on the training box (where the data
    # lives) and is reverse-tunneled to 127.0.0.1:8504 here. nginx already
    # forwards /rfcai/* to this relay app, so we proxy /labeler/* through
    # to the tunnel. All headers and bodies pass through unchanged so HTTP
    # Basic auth, HTMX requests, and image responses all just work.

    _LABELER_BACKEND = os.environ.get(
        "RFCAI_LABELER_BACKEND", "http://127.0.0.1:8504"
    )
    # Strip hop-by-hop headers per RFC 7230 §6.1.
    _HOP_BY_HOP = {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade",
        "content-length", "content-encoding",
    }

    async def _proxy_labeler(request: Request, suffix: str) -> Response:
        target = f"{_LABELER_BACKEND}/rfcai/labeler/{suffix}"
        if request.url.query:
            target += f"?{request.url.query}"
        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
        }
        body = await request.body()
        # Generous timeout: a cold-start grid load runs Hough+blur+dHash
        # over every labeled crop (~500 files at ~50ms each = ~30-60s on
        # the training-box CPU). Subsequent loads are instant from cache.
        # Bumped to 10 min so big-video /upload-video uploads (ffmpeg
        # extract + per-frame Hough + per-crop classify on a 100MB+ clip)
        # don't 500 here while the upstream is still working.
        async with httpx.AsyncClient(timeout=600.0) as client:
            upstream = await client.request(
                method=request.method,
                url=target,
                headers=fwd_headers,
                content=body,
                follow_redirects=False,
            )
        resp_headers = {
            k: v for k, v in upstream.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=resp_headers,
        )

    # --- aired.com-side video archive --------------------------------
    # The relay TEEs every /labeler/upload-video into /srv/rfcai/videos/
    # with a class-prefix filename, THEN forwards the original multipart
    # to the box so the existing extraction path keeps working. Result:
    # aired.com always has a durable archive copy regardless of what
    # happens to the box.

    async def _validate_admin_via_box(authz_header: str) -> bool:
        """Forward the caller's Authorization header to the box's cheap
        /labeler/auth/whoami endpoint. Returns True if the box says the
        caller is admin-authenticated."""
        if not authz_header:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_LABELER_BACKEND}/rfcai/labeler/auth/whoami",
                    headers={"Authorization": authz_header},
                )
            return resp.status_code == 200
        except Exception:
            return False

    @app.post("/labeler/upload-video")
    async def labeler_upload_video_tee(request: Request):
        # Bearer-validate before we save anything. Don't want random
        # bodies hitting /srv/rfcai/videos/.
        if not await _validate_admin_via_box(
            request.headers.get("authorization", "")
        ):
            raise HTTPException(401, "admin auth required")

        # Parse the multipart once. We consume the body to extract the
        # file + form fields, then re-construct it when forwarding to
        # the box — extra in-memory work, but it lets us name the
        # aired.com archive copy by class without guessing.
        form = await request.form()
        family = str(form.get("family") or "")
        gender = str(form.get("gender") or "")
        upload = form.get("file")
        if family not in CANONICAL_FAMILIES:
            raise HTTPException(400, f"unknown family {family!r}")
        if gender not in CANONICAL_GENDERS:
            raise HTTPException(400, f"unknown gender {gender!r}")
        if upload is None or not getattr(upload, "filename", None):
            raise HTTPException(400, "missing file")
        target_cls = f"{family}-{gender}"
        original_filename = Path(upload.filename).name
        ext = Path(upload.filename).suffix.lower()
        if ext not in VIDEO_EXTS:
            raise HTTPException(400, f"unsupported video extension {ext!r}")

        data = await upload.read()

        # Save the archive copy on aired.com.
        videos_dir = _videos_root()
        videos_dir.mkdir(parents=True, exist_ok=True)
        assigned_name = _class_video_filename(family, gender, ext)
        archive_path = videos_dir / assigned_name
        n = 1
        while archive_path.exists():
            stem = Path(assigned_name).stem
            archive_path = videos_dir / f"{stem}_dup{n}{ext}"
            n += 1
        archive_path.write_bytes(data)
        # Minimal sidecar — the box's response will fill in the
        # processing-state fields, but this guarantees the aired.com
        # copy always has the class info even if the box never
        # finishes (e.g. the box is offline).
        sidecar = _video_sidecar_path(archive_path)
        sidecar.write_text(json.dumps({
            "video_filename": archive_path.name,
            "original_filename": original_filename,
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "family": family,
            "gender": gender,
            "target_class": target_cls,
            "size_bytes": len(data),
            "archive_location": "aired.com",
            # Filled when (if) the box reports back via the existing
            # /labeler/upload-video extraction path.
            "extraction_state": "pending",
            "n_frames_extracted": None,
            "extracted_crops": [],
            "processed_at": None,
        }, indent=2))

        # Re-build the multipart for the upstream forward. The box runs
        # its own extraction job on its own copy; we don't dedupe storage
        # tonight, just guarantee aired.com has the durable archive.
        forward_files = {
            "file": (
                original_filename,
                data,
                getattr(upload, "content_type", "application/octet-stream"),
            ),
        }
        forward_data = {
            k: str(v) for k, v in form.items()
            if k != "file" and not hasattr(v, "filename")
        }
        fwd_headers = {
            "Authorization": request.headers.get("authorization", ""),
        }
        archive_url = (
            f"/rfcai/labeler/videos/download/{archive_path.name}"
        )
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                upstream = await client.post(
                    f"{_LABELER_BACKEND}/rfcai/labeler/upload-video",
                    data=forward_data,
                    files=forward_files,
                    headers=fwd_headers,
                )
        except Exception as e:
            # Box unreachable. The aired.com archive copy is on disk,
            # so the upload isn't lost; just signal the failure.
            return JSONResponse(
                status_code=502,
                content={
                    "archived_on_relay": True,
                    "archive_url": archive_url,
                    "archive_filename": archive_path.name,
                    "size_bytes": len(data),
                    "target_class": target_cls,
                    "forwarded_to_box": False,
                    "box_error": str(e),
                },
            )

        # Preserve the box's response shape so the Flutter app's
        # VideoTrainingUploadResult parser keeps working — just patch
        # in the archive fields when the response is JSON.
        ct = upstream.headers.get("content-type", "")
        if ct.startswith("application/json"):
            try:
                merged = upstream.json()
            except Exception:
                merged = {}
            merged["archived_on_relay"] = True
            merged["archive_url"] = archive_url
            merged["archive_filename"] = archive_path.name
            return JSONResponse(
                status_code=upstream.status_code, content=merged,
            )
        # HTML / opaque body: pass through unchanged, but record the
        # archive event in a custom header so a curious client can see
        # the relay also stored a copy.
        resp_headers = dict(upstream.headers)
        for hop in (
            "content-length", "content-encoding",
            "transfer-encoding", "connection",
        ):
            resp_headers.pop(hop, None)
        resp_headers["X-Relay-Archive"] = archive_path.name
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=resp_headers,
        )

    _VIDEO_MIME = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
    }

    @app.get("/labeler/videos/download/{name}")
    async def labeler_videos_download(name: str, request: Request):
        """Local-first download — aired.com's /srv/rfcai/videos/ archive
        wins over the box-served copy. Falls back to the existing
        labeler proxy when the file isn't in the archive, so share
        links to box-only files (the historical .MOVs etc.) keep
        working unchanged."""
        if "/" in name or ".." in name or name.startswith("."):
            raise HTTPException(400, "invalid name")
        local = _videos_root() / name
        try:
            local_resolved = local.resolve()
            local_resolved.relative_to(_videos_root())
        except (OSError, ValueError):
            local_resolved = None
        if (local_resolved
                and local_resolved.is_file()
                and local_resolved.suffix.lower() in VIDEO_EXTS):
            return FileResponse(
                local_resolved,
                media_type=_VIDEO_MIME.get(
                    local_resolved.suffix.lower(),
                    "application/octet-stream",
                ),
                filename=name,
            )
        # Fall through to the box.
        return await _proxy_labeler(request, f"videos/download/{name}")

    @app.api_route("/labeler", methods=["GET", "POST"])
    async def labeler_root(request: Request):
        return await _proxy_labeler(request, "")

    @app.api_route("/labeler/{suffix:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def labeler_path(request: Request, suffix: str):
        return await _proxy_labeler(request, suffix)

    return app


# Module-level app instance for `uvicorn rfconnectorai.server.app:app`.
app = create_app()
