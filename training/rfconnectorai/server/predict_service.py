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
from starlette.middleware.sessions import SessionMiddleware

from rfconnectorai.classifier.predict import (
    ConnectorClassifier, EnsembleClassifier,
)
from rfconnectorai.data_fetch.connector_crops import (
    detect_connector_crops, detect_connector_crops_yolo,
)
from rfconnectorai.server.labeler import create_router as create_labeler_router


DEFAULT_MODEL_DIR = Path("./models/connector_classifier")
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # 25 MB; phone frames usually 1-3 MB
DEFAULT_MAX_VIDEO_BYTES = 200 * 1024 * 1024   # 200 MB; phone clips can be big at 4K
DEFAULT_VIDEO_FPS = 1.0                       # sample 1 frame/sec — plenty for ID
DEFAULT_VIDEO_MAX_FRAMES = 30                 # cap at ~30s of video
# Pre-filter: each detected crop runs through rembg; if the central
# foreground silhouette covers less than this fraction, we treat the
# crop as "not actually a connector" (just background texture that
# Hough latched onto). 0.20 keeps real connector crops easily,
# rejects ~all random-noise/junk crops, catches most wood-texture
# false positives in our local benchmark.
DEFAULT_CLASSIFY_ON_CLEANED = False
# When True, the rembg-cleaned crop (alpha composited on white) is fed
# to the classifier instead of the raw crop. Training data includes
# both raw and rembg-cleaned variants so the model has seen both, and
# inferring on cleaned crops removes background patterns that may bias
# the classifier. Toggle via RFCAI_CLASSIFY_ON_CLEANED env var.
DEFAULT_MIN_FG_FRACTION = 0.05      # below this rembg saw essentially nothing
# Bimodal center-density rule. detect_connector_crops returns TIGHT
# crops, so a real connector silhouette appears in one of two ways:
#   1. Fills the crop edge-to-edge → near-uniform foreground (ratio ~1)
#   2. Sits as a small object in the middle of a slightly oversized
#      crop → high inner-vs-outer ratio (ratio ≫ 5)
# Wood-pattern false positives that fool rembg consistently land in
# the *middle* of this range (ratio 2-4): rembg sees a moderately
# centered blob that doesn't look like either pattern. We KEEP crops
# with ratio <= LOW_RATIO OR ratio >= HIGH_RATIO, and reject the
# in-between zone.
DEFAULT_LOW_CENTER_RATIO = 2.0     # tuned to recover 2.4mm-F crops at 1.74/2.14
DEFAULT_HIGH_CENTER_RATIO = 5.0
DEFAULT_MIN_UNIFORM_FG = 0.20       # for the "fills the crop" pattern

# YOLO fallback: when Hough returns no crops, try the trained YOLO
# detector (committed at models/detector/best.pt). Hough handles
# perpendicular face-on shots; YOLO catches off-axis cases Hough
# misses. The rembg fg filter still gates whatever YOLO returns, so
# the 0% false-positive property on background frames is preserved.
DEFAULT_YOLO_FALLBACK = False
DEFAULT_YOLO_WEIGHTS = "models/detector/best.pt"   # relative to repo root
DEFAULT_YOLO_CONF = 0.20

# Spec lookup. After classification, each prediction is enriched with
# a `spec` block from connectors.yaml (frequency range, impedance,
# vendor variants, etc.) so the phone app can show a richer result
# card without the classifier needing to know any of that.
DEFAULT_SPECS_PATH = "rfconnectorai/specs/connectors.yaml"


def _config_from_env() -> dict:
    return {
        "model_dir": Path(os.environ.get("RFCAI_MODEL_DIR", DEFAULT_MODEL_DIR)),
        "device_token": os.environ.get("RFCAI_DEVICE_TOKEN", ""),
        "max_upload_bytes": int(os.environ.get("RFCAI_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)),
        "max_video_bytes": int(os.environ.get("RFCAI_MAX_VIDEO_BYTES", DEFAULT_MAX_VIDEO_BYTES)),
        "video_fps": float(os.environ.get("RFCAI_VIDEO_FPS", DEFAULT_VIDEO_FPS)),
        "video_max_frames": int(os.environ.get("RFCAI_VIDEO_MAX_FRAMES", DEFAULT_VIDEO_MAX_FRAMES)),
        "max_detections": int(os.environ.get("RFCAI_MAX_DETECTIONS", 4)),
        "fg_filter_enabled": os.environ.get("RFCAI_FG_FILTER", "1") not in ("0", "false", "False"),
        "min_fg_fraction": float(os.environ.get("RFCAI_MIN_FG_FRACTION", DEFAULT_MIN_FG_FRACTION)),
        "min_uniform_fg": float(os.environ.get("RFCAI_MIN_UNIFORM_FG", DEFAULT_MIN_UNIFORM_FG)),
        "low_center_ratio": float(os.environ.get("RFCAI_LOW_CENTER_RATIO", DEFAULT_LOW_CENTER_RATIO)),
        "high_center_ratio": float(os.environ.get("RFCAI_HIGH_CENTER_RATIO", DEFAULT_HIGH_CENTER_RATIO)),
        "classify_on_cleaned": os.environ.get("RFCAI_CLASSIFY_ON_CLEANED", "0") in ("1", "true", "True"),
        "yolo_fallback": os.environ.get("RFCAI_YOLO_FALLBACK", "0") in ("1", "true", "True"),
        "yolo_weights": Path(os.environ.get("RFCAI_YOLO_WEIGHTS", DEFAULT_YOLO_WEIGHTS)),
        "yolo_conf": float(os.environ.get("RFCAI_YOLO_CONF", DEFAULT_YOLO_CONF)),
        "specs_path": Path(os.environ.get("RFCAI_SPECS_PATH", DEFAULT_SPECS_PATH)),
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
    fg_filter_enabled: bool = cfg["fg_filter_enabled"]
    min_fg_fraction: float = cfg["min_fg_fraction"]
    min_uniform_fg: float = cfg["min_uniform_fg"]
    low_center_ratio: float = cfg["low_center_ratio"]
    high_center_ratio: float = cfg["high_center_ratio"]
    classify_on_cleaned: bool = cfg["classify_on_cleaned"]
    yolo_fallback_enabled: bool = cfg.get("yolo_fallback", False)
    yolo_weights_path: Path = cfg.get("yolo_weights", Path(DEFAULT_YOLO_WEIGHTS))
    yolo_conf_threshold: float = cfg.get("yolo_conf", DEFAULT_YOLO_CONF)
    specs_path: Path = cfg.get("specs_path", Path(DEFAULT_SPECS_PATH))

    # Optional swap-in: YOLO11n + EfficientNetV2-S pipeline (sourced from
    # trextrader/hotdogornot). When RFCAI_USE_JERRY_PIPELINE=1 is set and
    # RFCAI_JERRY_MODEL_DIR points at a folder with {detector,classifier}.onnx
    # + classifier_labels.json (+ optional thresholds.json), all /predict
    # calls route through it instead of the Hough+ResNet18 path below.
    # Unset → existing path runs unchanged. Validated against our 35-image
    # holdout at 97.1% Full / 100.0% Gender vs the legacy path's 68.6%.
    jerry_pipeline = None
    if os.environ.get("RFCAI_USE_JERRY_PIPELINE") == "1":
        jerry_dir = os.environ.get("RFCAI_JERRY_MODEL_DIR")
        if not jerry_dir:
            raise RuntimeError(
                "RFCAI_USE_JERRY_PIPELINE=1 but RFCAI_JERRY_MODEL_DIR is unset"
            )
        from rfconnectorai.pipeline.jerry_pipeline import JerryPipeline
        jerry_pipeline = JerryPipeline(Path(jerry_dir))
        print(
            f"[predict] jerry pipeline enabled: dir={jerry_dir} "
            f"classes={jerry_pipeline.class_names} box_min={jerry_pipeline.box_min}",
            flush=True,
        )

    app = FastAPI(title="RF Connector AI prediction service", version="1.0.0")

    app.add_middleware(
        SessionMiddleware,
        secret_key=os.environ.get(
            "RFCAI_SESSION_SECRET",
            # Dev default — overridden by env var in production via
            # /etc/default/rfcai-predict. Generated via secrets.token_urlsafe(32).
            "dev-session-secret-CHANGE-IN-PROD-via-RFCAI_SESSION_SECRET",
        ),
        https_only=False,  # box runs HTTP behind nginx HTTPS proxy
        same_site="lax",
    )

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
    # If RFCAI_ENSEMBLE_WEIGHTS=path1,path2,... is set, also load those
    # weights and use an EnsembleClassifier — averages softmax across
    # all models for variance reduction. All weights must share the
    # same labels.json (same class set + input_size).
    classifier = None
    if (model_dir / "weights.pt").exists() and (model_dir / "labels.json").exists():
        try:
            ensemble_paths_raw = os.environ.get("RFCAI_ENSEMBLE_WEIGHTS", "").strip()
            if ensemble_paths_raw:
                extra = [Path(p.strip()) for p in ensemble_paths_raw.split(",") if p.strip()]
                weights = [model_dir / "weights.pt"] + extra
                missing = [p for p in weights if not p.exists()]
                if missing:
                    raise FileNotFoundError(
                        f"ensemble weight(s) missing: {missing}")
                classifier = ConnectorClassifier.load_ensemble(
                    weights, model_dir / "labels.json",
                )
                print(f"[predict_service] loaded ENSEMBLE of {len(weights)} "
                      f"models: {[p.name for p in weights]}")
            else:
                classifier = ConnectorClassifier.load(model_dir)
        except Exception as e:
            print(f"[predict_service] classifier load failed: {e}")

    # YOLO fallback detector — only fires when Hough returns 0 crops.
    # Lazy import so the service still starts if ultralytics is not
    # installed (fallback simply disables).
    yolo_model = None
    if yolo_fallback_enabled:
        try:
            from ultralytics import YOLO   # type: ignore
            if not yolo_weights_path.exists():
                raise FileNotFoundError(yolo_weights_path)
            yolo_model = YOLO(str(yolo_weights_path))
            print(f"[predict_service] YOLO fallback enabled "
                  f"({yolo_weights_path}, conf={yolo_conf_threshold})")
        except Exception as e:
            print(f"[predict_service] YOLO fallback unavailable: {e}")
            yolo_model = None

    # Spec lookup. Maps family name → spec dict from connectors.yaml.
    # Built once at startup; lookup per-prediction is O(1).
    spec_table: dict[str, dict] = {}
    if specs_path.exists():
        try:
            import yaml   # type: ignore
            raw = yaml.safe_load(specs_path.read_text())
            # connectors.yaml structure (schema v1): top-level
            # `connectors:` list, each entry has `id` and
            # `display_name`. We index by display_name lowercased so a
            # class_name like "2.4mm-M" looks up the "2.4mm" entry
            # with a single dict access.
            if isinstance(raw, dict) and "connectors" in raw:
                for entry in raw["connectors"]:
                    if not isinstance(entry, dict):
                        continue
                    display = entry.get("display_name") or entry.get("name")
                    family_id = entry.get("id") or entry.get("family")
                    if display:
                        spec_table[str(display).lower()] = entry
                    if family_id:
                        spec_table[str(family_id).lower()] = entry
            elif isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, dict):
                        spec_table[str(k).lower()] = v
            print(f"[predict_service] spec lookup loaded "
                  f"({len(spec_table)} keys from {specs_path})")
        except Exception as e:
            print(f"[predict_service] spec lookup unavailable: {e}")

    def _lookup_spec(class_name: str) -> dict | None:
        """`class_name` like '2.4mm-M' → spec dict for '2.4mm' family,
        or None if no entry."""
        if "-" not in class_name:
            return None
        family = class_name.rsplit("-", 1)[0].lower()
        # Normalise the family key the same way connectors.yaml uses it
        # ('2.4mm', 'sma', etc.).
        return spec_table.get(family) or spec_table.get(family.upper())

    def _decompose_probabilities(
        probabilities: dict[str, float],
    ) -> tuple[str, str, float, float]:
        """From a 6-class softmax keyed by 'FAMILY-GENDER', marginalize
        into family-confidence and gender-confidence. Returns the most
        likely family + gender + their marginal probabilities."""
        fam_p: dict[str, float] = {}
        gen_p: dict[str, float] = {}
        for cls, p in probabilities.items():
            if "-" not in cls:
                continue
            f, g = cls.rsplit("-", 1)
            fam_p[f] = fam_p.get(f, 0.0) + float(p)
            gen_p[g] = gen_p.get(g, 0.0) + float(p)
        top_fam = max(fam_p.items(), key=lambda kv: kv[1]) if fam_p else ("", 0.0)
        top_gen = max(gen_p.items(), key=lambda kv: kv[1]) if gen_p else ("", 0.0)
        return top_fam[0], top_gen[0], top_fam[1], top_gen[1]

    # rembg session for the foreground pre-filter. Lazy import so the
    # service still starts on a box without rembg (filter just disables
    # itself in that case).
    rembg_session = None
    if fg_filter_enabled:
        try:
            from rembg import new_session   # type: ignore
            # u2netp is the smaller/faster variant of U^2-Net. ~10x
            # faster on CPU at our crop sizes vs the default u2net,
            # silhouettes are nearly identical for connector-sized
            # subjects. The fg-filter heuristics (min_fg, center
            # density ratios) were tuned against u2net but read across
            # cleanly because we threshold alpha at 32, well above any
            # model-edge fuzz difference between the two variants.
            rembg_model = os.environ.get("RFCAI_REMBG_MODEL", "u2netp")
            rembg_session = new_session(rembg_model)
            print(f"[predict_service] rembg model: {rembg_model}")
            print("[predict_service] foreground filter enabled "
                  f"(min_fg={min_fg_fraction}, "
                  f"keep_low<={low_center_ratio} OR "
                  f"keep_high>={high_center_ratio} "
                  f"with min_uniform_fg={min_uniform_fg})")
        except Exception as e:
            print(f"[predict_service] rembg unavailable, fg filter disabled: {e}")
            rembg_session = None

    def _crop_passes_fg_filter(
        crop_bgr: np.ndarray,
    ) -> tuple[bool, float, float, np.ndarray | None]:
        """Returns (keep, fg_fraction, center_density_ratio, rgba).
        rgba is the rembg output (H,W,4) on success — useful for the
        optional 'classify on cleaned crop' path."""
        if rembg_session is None:
            return True, 1.0, 1.0, None
        try:
            from rembg import remove   # type: ignore
            # rembg treats raw ndarrays as RGB; we have BGR from cv2.
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            rgba = remove(crop_rgb, session=rembg_session)
        except Exception:
            return True, 1.0, 1.0, None   # fail open on rembg errors
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            print(f"[predict_service] WARNING: rembg returned non-RGBA "
                  f"({rgba.shape}); fg filter inactive on this crop")
            return True, 1.0, 1.0, None
        alpha = rgba[:, :, 3]
        h, w = alpha.shape
        if h == 0 or w == 0:
            return False, 0.0, 0.0, rgba
        fg_total = (alpha > 32)
        fg_frac = float(fg_total.sum()) / float(alpha.size)
        cy0, cy1 = h // 4, h - h // 4
        cx0, cx1 = w // 4, w - w // 4
        inner = fg_total[cy0:cy1, cx0:cx1]
        inner_area = max(1, inner.size)
        outer_area = max(1, fg_total.size - inner_area)
        inner_density = float(inner.sum()) / inner_area
        outer_density = (float(fg_total.sum() - inner.sum())) / outer_area
        center_ratio = (inner_density / outer_density) if outer_density > 1e-6 \
                       else (10.0 if inner_density > 0 else 0.0)
        if fg_frac < min_fg_fraction:
            return False, fg_frac, center_ratio, rgba
        keep_uniform = (fg_frac >= min_uniform_fg
                        and center_ratio <= low_center_ratio)
        keep_centered = center_ratio >= high_center_ratio
        return (keep_uniform or keep_centered), fg_frac, center_ratio, rgba

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
            "ensemble_size": (len(classifier.models)
                              if isinstance(classifier, EnsembleClassifier)
                              else 1 if classifier else 0),
            "max_detections": max_detections,
            "classify_on_cleaned": classify_on_cleaned,
            "fg_filter": {
                "enabled": fg_filter_enabled,
                "available": rembg_session is not None,
                "min_fg": min_fg_fraction,
                "min_uniform_fg": min_uniform_fg,
                "low_center_ratio": low_center_ratio,
                "high_center_ratio": high_center_ratio,
            },
            "yolo_fallback": {
                "enabled": yolo_fallback_enabled,
                "available": yolo_model is not None,
                "conf_threshold": yolo_conf_threshold,
            },
            "spec_lookup": {
                "available": bool(spec_table),
                "families_indexed": len(spec_table),
            },
        }

    def _classify_frame(bgr: np.ndarray) -> list[dict]:
        """Detect + classify on one BGR frame, return prediction dicts.
        Each crop is screened by the rembg foreground filter; rejected
        crops are dropped before classification (the bias-locked
        ResNet-18 will otherwise emit a confident wrong answer for
        background patches)."""
        # Short-circuit: when the Jerry pipeline is active, skip the
        # entire Hough+rembg+ResNet18 path. The YOLO detector is robust
        # enough that the foreground filter is redundant.
        if jerry_pipeline is not None:
            preds = jerry_pipeline.run(bgr)
            for p in preds:
                p["spec"] = _lookup_spec(p["class_name"])
            return preds
        import time as _time
        _t0 = _time.perf_counter()
        crops = detect_connector_crops(bgr, max_crops=max_detections)
        _t_hough_ms = (_time.perf_counter() - _t0) * 1000
        _per_crop_ms: list[dict] = []
        crop_source = "hough"
        if not crops and yolo_model is not None:
            # Hough found nothing — try YOLO as a fallback. The rembg
            # foreground filter still runs per-crop below, so an empty
            # frame (wall, desk) that YOLO mis-detects still gets
            # rejected before classification.
            crops = detect_connector_crops_yolo(
                bgr,
                yolo_model,
                max_crops=max_detections,
                conf=yolo_conf_threshold,
            )
            if crops:
                crop_source = "yolo"
        out = []
        for c in crops:
            _t_c = _time.perf_counter()
            keep, fg_frac, center_ratio, rgba = _crop_passes_fg_filter(c.crop)
            _t_fg_ms = (_time.perf_counter() - _t_c) * 1000
            _t_c = _time.perf_counter()
            if not keep:
                _per_crop_ms.append({"fg": _t_fg_ms, "kept": False})
                continue
            if classify_on_cleaned and rgba is not None:
                # Composite the rembg silhouette over white, drop alpha.
                # Training data includes _clean variants so the model
                # has seen this distribution; should reduce background-
                # pattern bias in the prediction.
                rgb_pixels = rgba[:, :, :3].astype(np.float32)
                alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
                white = np.full_like(rgb_pixels, 255.0)
                composited = (rgb_pixels * alpha + white * (1.0 - alpha)).astype(np.uint8)
                # rembg returned RGB ordering already.
                pred = classifier.predict(composited)
            else:
                rgb_crop = cv2.cvtColor(c.crop, cv2.COLOR_BGR2RGB)
                pred = classifier.predict(rgb_crop)
            _t_cls_ms = (_time.perf_counter() - _t_c) * 1000
            _per_crop_ms.append({
                "fg": round(_t_fg_ms, 1),
                "cls": round(_t_cls_ms, 1),
                "crop_shape": list(c.crop.shape),
            })
            x, y, bw, bh = c.bbox
            probs = {k: float(v) for k, v in pred.probabilities.items()}
            family, gender, fam_conf, gen_conf = _decompose_probabilities(probs)
            spec = _lookup_spec(pred.class_name)
            print(f"[predict_timing] hough={_t_hough_ms:.0f}ms "
                  f"per_crop={_per_crop_ms}  total_so_far={(_time.perf_counter()-_t0)*1000:.0f}ms",
                  flush=True)
            out.append({
                # Original fields — kept verbatim for backwards compat
                # with the Flutter app.
                "class_name": pred.class_name,
                "confidence": float(pred.confidence),
                "probabilities": probs,
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                # New structured fields — additive, safe for older
                # clients to ignore.
                "family": family,
                "gender": gender,
                "family_confidence": fam_conf,
                "gender_confidence": gen_conf,
                "spec": spec,
                "_diag": {
                    "fg_fraction": fg_frac,
                    "center_density_ratio": center_ratio,
                    "crop_source": crop_source,
                },
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
