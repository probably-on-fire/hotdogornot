"""
RF Connector AI labeler — FastAPI + HTMX UI for cleaning training data.

Mounts at `/labeler/*` inside the predict service, exposed via the same
nginx + reverse-tunnel chain at `aired.com/rfcai/labeler/`.

The Streamlit Review tab is functional but slow: every click triggers a
script rerun, scroll position is lost, layout reflows are awkward. This
is a real web app — server-rendered HTML with HTMX for per-tile DOM
swaps, so a Delete click vanishes the tile from the grid in place
without reloading anything else.

Routes:
  GET  /labeler/                         main page (filters + grid + JS hotkeys)
  GET  /labeler/grid                     grid partial (HTMX-targeted on filter change)
  GET  /labeler/img                      serve a labeled image (path traversal guarded)
  POST /labeler/delete                   unlink one file → empty 200 (HTMX swaps tile out)
  POST /labeler/flip                     move file to opposite-gender folder → empty 200
  POST /labeler/bulk-delete              unlink many files → updated stats fragment

Auth: HTTP Basic via LABELER_USER / LABELER_PASS env vars. If unset the
service refuses to serve the labeler routes (predict still works).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from rfconnectorai.data_fetch.connector_crops import detect_connector_crops_hough
from rfconnectorai.server import auth as _auth


CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
    # 1.85mm — newest addition (~67 GHz, smallest precision connector).
    # Currently zero training data; auto_retrain drops the class from
    # the head until it has ≥ MIN_SAMPLES_PER_CLASS images. Same status
    # SMA had/has.
    "1.85mm-M", "1.85mm-F",
]


def _data_root() -> Path:
    """The labeled-data folder this service writes into. Resolved fresh
    every call so a service restart picks up env changes."""
    return Path(os.environ.get(
        "RFCAI_LABELED_DIR",
        "/opt/rfcai/repo/training/data/labeled/embedder",
    )).resolve()


def _test_holdout_root() -> Path:
    return Path(os.environ.get(
        "RFCAI_TEST_HOLDOUT_DIR",
        "/opt/rfcai/repo/training/data/test_holdout",
    )).resolve()


def _videos_root() -> Path:
    return Path(os.environ.get(
        "RFCAI_VIDEOS_DIR",
        "/opt/rfcai/repo/training/data/videos",
    )).resolve()


def _snapshot_root() -> Path:
    """Directory holding pre-built tarball snapshots of training data.
    Public-readable downloads for forks/collaborators (see /snapshots route)."""
    return Path(os.environ.get(
        "RFCAI_SNAPSHOT_DIR",
        "/opt/rfcai/repo/training/data/snapshots",
    )).resolve()


def _users_db_path() -> Path:
    """SQLite path for the labeler users table. Lives under data/
    so it's covered by the existing systemd ReadWritePaths."""
    return Path(os.environ.get(
        "RFCAI_USERS_DB",
        "/opt/rfcai/repo/training/data/labeler_users.db",
    )).resolve()


def _source_backup_root() -> Path:
    """Immutable backup of source images uploaded via /upload-train and
    /upload-test. Every saved file is hardlinked here; deleting the
    working copy in _data_root() / _test_holdout_root() leaves the
    backup intact. Hardlinks add zero disk cost (same inode).

    Not exposed via any read route — invisible to /grid, /img,
    /snapshots. Operators can tar it for offsite backup.
    """
    return Path(os.environ.get(
        "RFCAI_SOURCE_BACKUP_DIR",
        "/opt/rfcai/repo/training/data/source_backup",
    )).resolve()


CANONICAL_FAMILIES = ["SMA", "3.5mm", "2.92mm", "2.4mm", "1.85mm"]


def _drive_backup(local_path: Path, target_class: str) -> dict | None:
    """Upload `local_path` to the configured Drive folder. Returns
    {"id": "...", "webViewLink": "..."} on success, None if Drive
    backup isn't configured or the call fails.

    Reads:
      RFCAI_DRIVE_FOLDER_ID         — destination folder ID (required)
      RFCAI_DRIVE_CREDENTIALS_PATH  — path to service-account JSON (required)
    Both unset → no-op (returns None). Failures are logged, never raised.
    """
    folder_id = os.environ.get("RFCAI_DRIVE_FOLDER_ID", "").strip()
    creds_path = os.environ.get("RFCAI_DRIVE_CREDENTIALS_PATH", "").strip()
    if not folder_id or not creds_path:
        return None
    try:
        # Lazy import so the service doesn't fail to boot when the
        # Drive client libs aren't installed.
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        creds = service_account.Credentials.from_service_account_file(
            creds_path, scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        media = MediaFileUpload(
            str(local_path),
            mimetype="video/mp4",
            resumable=False,
        )
        # Tag the file with the target class so the Drive view groups
        # naturally; the on-disk filename already encodes class+timestamp.
        meta = {
            "name": local_path.name,
            "parents": [folder_id],
            "description": f"target_class={target_class}",
        }
        f = svc.files().create(
            body=meta, media_body=media,
            fields="id, webViewLink",
            supportsAllDrives=True,
        ).execute()
        return {"id": f.get("id"), "webViewLink": f.get("webViewLink")}
    except Exception as e:
        print(f"[upload-video] Drive backup failed: {e}", flush=True)
        return None


_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _resolve_session(raw: str | None) -> str:
    """Validate a client-supplied session token or default to today's UTC date.

    Format constraint keeps filenames safe (no path traversal, no spaces, no
    dots that would confuse extension parsing). Empty/missing -> server stamp."""
    if not raw:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not _SESSION_RE.fullmatch(raw):
        raise HTTPException(400, f"invalid session token {raw!r}")
    return raw


# ---------------------------------------------------------------------------
# Per-crop signal computation (cached in process memory by path+mtime)
# ---------------------------------------------------------------------------

_signals_cache: dict[tuple[str, float], dict] = {}


def _compute_signals(path_str: str) -> dict:
    """Hough-circle count + Laplacian-variance sharpness + dHash."""
    import imagehash
    from PIL import Image

    bgr = cv2.imread(path_str)
    if bgr is None:
        return {"n_circles": 0, "blur_var": 0.0, "dhash_hex": ""}

    h, w = bgr.shape[:2]
    short = min(h, w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    min_r = max(15, int(short * 0.10))
    max_r = max(min_r + 1, int(short * 0.45))
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r,
        param1=80, param2=30, minRadius=min_r, maxRadius=max_r,
    )
    n_circles = 0 if circles is None else len(circles[0])
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dhash_hex = str(imagehash.dhash(Image.fromarray(rgb), hash_size=8))
    return {"n_circles": n_circles, "blur_var": blur_var, "dhash_hex": dhash_hex}


def _signals_for(path: Path) -> dict:
    try:
        key = (str(path), path.stat().st_mtime)
    except FileNotFoundError:
        return {"n_circles": 0, "blur_var": 0.0, "dhash_hex": ""}
    if key not in _signals_cache:
        _signals_cache[key] = _compute_signals(str(path))
    return _signals_cache[key]


# ---------------------------------------------------------------------------
# Records & filtering
# ---------------------------------------------------------------------------


@dataclass
class CropRecord:
    path: Path
    cls: str
    name: str
    n_circles: int
    blur_var: float
    dhash_hex: str
    is_duplicate: bool = False

    @property
    def path_str(self) -> str:
        return str(self.path)

    @property
    def flip_target(self) -> str | None:
        family, gender = self.cls.rsplit("-", 1)
        new_cls = f"{family}-{'F' if gender == 'M' else 'M'}"
        return new_cls if new_cls in CANONICAL_CLASSES else None


def _list_records(classes: list[str]) -> list[CropRecord]:
    root = _data_root()
    records: list[CropRecord] = []
    for cls in classes:
        if cls not in CANONICAL_CLASSES:
            continue
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(p for p in cls_dir.iterdir() if p.is_file()):
            sig = _signals_for(img_path)
            records.append(CropRecord(
                path=img_path,
                cls=cls,
                name=img_path.name,
                n_circles=sig["n_circles"],
                blur_var=sig["blur_var"],
                dhash_hex=sig["dhash_hex"],
            ))
    _mark_duplicates(records)
    return records


def _mark_duplicates(records: list[CropRecord], max_distance: int = 6) -> None:
    """In each near-duplicate group, the sharpest crop is the keeper,
    every other one gets is_duplicate=True. Pairwise hamming distance —
    O(n²) which is fine up to a few thousand crops."""
    import imagehash

    n = len(records)
    if n == 0: return
    hashes = [
        imagehash.hex_to_hash(r.dhash_hex) if r.dhash_hex else None
        for r in records
    ]
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        if hashes[i] is None: continue
        for j in range(i + 1, n):
            if hashes[j] is None: continue
            if (hashes[i] - hashes[j]) <= max_distance:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for root_idx, idxs in groups.items():
        if len(idxs) <= 1:
            continue
        sharpest = max(idxs, key=lambda i: records[i].blur_var)
        for i in idxs:
            if i != sharpest:
                records[i].is_duplicate = True


def _safe_path(raw: str) -> Path:
    """Validate that `raw` points inside the labeled-data root, the
    test-holdout root, or the videos root. Prevents path-traversal
    attacks via a crafted form value. All three roots are accepted so
    /delete can undo uploads made via /upload-test, /upload-train,
    AND /upload-video (the latter is used by the /videos management
    page)."""
    try:
        candidate = Path(raw).resolve()
    except Exception:
        raise HTTPException(400, "bad path")
    for root in (_data_root(), _test_holdout_root(), _videos_root()):
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise HTTPException(400, "path outside data roots")


def _video_sidecar_path(video_path: Path) -> Path:
    """Where to write the JSON manifest for a video — sibling file with
    a .crops.json suffix. Lets the /videos management page show how many
    agg_*.jpg crops a video produced + delete them as a bundle."""
    return video_path.with_suffix(video_path.suffix + ".crops.json")


def _class_video_filename(family: str, gender: str, ext: str,
                          when: float | None = None) -> str:
    """Build a self-describing video filename so a directory listing
    sorts naturally by connector type: `2.4mm-M_2026-05-21T18-23-15.mp4`.
    Uses ISO 8601 timestamps with `-` separating time parts (`:` is not
    a valid character in filenames on common filesystems).
    """
    target_cls = f"{family}-{gender}"
    ts = time.gmtime(when) if when is not None else time.gmtime()
    stamp = time.strftime("%Y-%m-%dT%H-%M-%S", ts)
    return f"{target_cls}_{stamp}{ext.lower()}"


def _filename_already_class_prefixed(name: str, family: str, gender: str) -> bool:
    """True if `name` already leads with the canonical class prefix —
    no need to rename on annotate when the upload path got it right."""
    return name.startswith(f"{family}-{gender}_")


def _class_counts() -> dict[str, int]:
    root = _data_root()
    out = {}
    for cls in CANONICAL_CLASSES:
        d = root / cls
        out[cls] = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
    return out


def _real_capture_counts(root: Path) -> dict[str, int]:
    """
    Count real human captures per class — excludes rembg-derived
    variants (_clean / _bg* / _z* / _central) and pure synth files
    (synth_*). Used by the stats endpoint to show the user the number
    they're actually moving with each shutter tap, not the augmented
    training-set size.
    """
    out: dict[str, int] = {}
    for cls in CANONICAL_CLASSES:
        d = root / cls
        if not d.is_dir():
            out[cls] = 0
            continue
        n = 0
        for p in d.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            stem = p.stem
            if stem.startswith("synth_"):
                continue
            if stem.endswith(("_clean", "_mask", "_central", "_centralv2")):
                continue
            # _bg<digits>, _z<digits> — derived augmentations.
            if "_bg" in stem and stem.rsplit("_bg", 1)[1].isdigit():
                continue
            if "_z" in stem and stem.rsplit("_z", 1)[1].isdigit():
                continue
            n += 1
        out[cls] = n
    return out


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_security = HTTPBasic()


def _require_basic_auth(creds: HTTPBasicCredentials = Depends(_security)) -> str:
    user = os.environ.get("LABELER_USER", "")
    pwd = os.environ.get("LABELER_PASS", "")
    if not user or not pwd:
        raise HTTPException(503, "labeler auth not configured (set LABELER_USER and LABELER_PASS)")
    if not (secrets.compare_digest(creds.username, user)
            and secrets.compare_digest(creds.password, pwd)):
        raise HTTPException(
            401, "invalid credentials",
            headers={"WWW-Authenticate": "Basic realm=labeler"},
        )
    return creds.username


def require_admin(request: Request) -> _auth.User:
    """Auth dependency for admin-only routes.

    Accepts either:
      1. A valid session cookie (set by /labeler/login)
      2. HTTP Basic Authorization header (for the Flutter app and
         curl users — credentials validated against the users DB,
         not the LABELER_USER/PASS env vars)

    Returns the authenticated User on success.
    Raises HTTPException(401) on failure.
    """
    db_path = _users_db_path()

    # Try session first.
    sess_user_id = request.session.get("user_id")
    if sess_user_id is not None:
        # Confirm the user still exists (admin may have deleted them
        # since they logged in).
        for u in _auth.list_users(db_path):
            if u.id == sess_user_id:
                return u
        # Stale session — clear it.
        request.session.clear()

    # Try Bearer token next.
    auth_header_lower = (request.headers.get("authorization") or "").lower()
    if auth_header_lower.startswith("bearer "):
        token = request.headers["authorization"].split(None, 1)[1].strip()
        user = _auth.authenticate_token(db_path, token)
        if user is not None:
            return user

    # Fall back to HTTP Basic.
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("basic "):
        import base64
        try:
            decoded = base64.b64decode(auth_header.split(None, 1)[1]).decode("utf-8")
            username, _, password = decoded.partition(":")
        except Exception:
            decoded = None
            username = password = ""
        if username:
            user = _auth.authenticate(db_path, username, password)
            if user is not None:
                return user

    raise HTTPException(
        status_code=401,
        detail="login required",
        headers={"WWW-Authenticate": 'Basic realm="labeler"'},
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _backup_hardlink(src: Path, cls: str, backup_root: Path) -> None:
    """Create an immutable hardlink of `src` under `backup_root/<cls>/`.

    Idempotent — re-running on an existing backup path is a no-op (the
    FileExistsError is swallowed). Failures are not raised: we'd rather
    keep the upload's 200 OK than fail the request because the backup
    filesystem hiccupped. (Hardlink-across-filesystems is the most
    common failure mode and we log it for surface visibility.)
    """
    import logging
    try:
        target_dir = backup_root / cls
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name
        try:
            os.link(src, target)
        except FileExistsError:
            pass  # idempotent
    except OSError as e:
        logging.getLogger("rfcai.labeler").warning(
            "source backup hardlink failed for %s -> %s/%s: %s",
            src, backup_root, cls, e,
        )


def create_router(classifier=None) -> APIRouter:
    """`classifier`: optional ConnectorClassifier instance. When passed,
    `/upload-video` callers can set `with_predict=true` to also receive
    a JSON prediction over the extracted crops (alongside saving them as
    training data). Left as None when the server starts without weights."""
    # Prefix matches the public URL path. nginx on aired.com forwards
    # /rfcai/* to the predict service backend with the /rfcai prefix
    # preserved (for paths nginx doesn't have an explicit rewrite for),
    # so backend routes must include /rfcai/ to be reachable externally.
    r = APIRouter(prefix="/rfcai/labeler", tags=["labeler"])

    # Read-only routes (stats / index / grid / img) are public so the
    # labeler grid can be shared as a viewer. Write routes (delete,
    # flip, bulk-delete, upload-*) stay basic-auth gated below.
    @r.get("/stats")
    def stats():
        """
        Per-class real-capture counts for the training set and the
        held-out test set. Used by the Flutter Contribute screen to
        show the user which classes are starved before they capture.
        """
        return {
            "train": _real_capture_counts(_data_root()),
            "holdout": _real_capture_counts(_test_holdout_root()),
        }

    @r.get("/", response_class=HTMLResponse)
    def index(request: Request):
        counts = _class_counts()
        return templates.TemplateResponse(
            request,
            "labeler/index.html",
            {
                "classes": CANONICAL_CLASSES,
                "counts": counts,
                "total": sum(counts.values()),
            },
        )

    @r.get("/grid", response_class=HTMLResponse)
    def grid(
        request: Request,
        cls: list[str] = Query(default=CANONICAL_CLASSES),
        only_no_circle: bool = False,
        only_multi: bool = False,
        hide_dups: bool = False,
        blur_threshold: int = 0,
        sort: str = "class",
        page: int = 1,
        per_page: int = 24,
        partial: bool = False,
    ):
        all_records = _list_records(cls)
        records = list(all_records)
        if only_no_circle:
            records = [r for r in records if r.n_circles == 0]
        if only_multi:
            records = [r for r in records if r.n_circles >= 2]
        if hide_dups:
            records = [r for r in records if not r.is_duplicate]
        if blur_threshold > 0:
            records = [r for r in records if r.blur_var >= blur_threshold]
        if sort == "blur_asc":
            records.sort(key=lambda r: r.blur_var)
        elif sort == "circles_asc":
            records.sort(key=lambda r: r.n_circles)
        else:
            records.sort(key=lambda r: (r.cls, r.name))

        n_total = len(all_records)
        n_visible = len(records)
        n_no_circle = sum(1 for r in all_records if r.n_circles == 0)
        n_multi = sum(1 for r in all_records if r.n_circles >= 2)
        n_dups = sum(1 for r in all_records if r.is_duplicate)

        per_page = max(8, min(5000, per_page))
        page = max(1, page)
        start = (page - 1) * per_page
        end = min(start + per_page, n_visible)
        batch = records[start:end]
        has_more = end < n_visible

        # Echo filter params for the next-page sentinel and the bulk-delete form.
        filter_query = _filter_qs(
            cls, only_no_circle, only_multi, hide_dups, blur_threshold, sort, per_page,
        )

        # Subsequent infinite-scroll loads ask for partial=true so we skip
        # the surrounding chrome (stats line, bulk-delete bar) and just
        # return the tiles + a new sentinel that the previous one replaces.
        template = "labeler/grid_batch.html" if partial else "labeler/grid.html"

        return templates.TemplateResponse(
            request,
            template,
            {
                "records": batch,
                "n_total": n_total,
                "n_visible": n_visible,
                "n_no_circle": n_no_circle,
                "n_multi": n_multi,
                "n_dups": n_dups,
                "page": page,
                "next_page": page + 1,
                "has_more": has_more,
                "start": start,
                "end": end,
                "per_page": per_page,
                "filter_query": filter_query,
            },
        )

    @r.get("/img")
    def serve_img(path: str):
        p = _safe_path(path)
        if not p.exists():
            raise HTTPException(404, "not found")
        return FileResponse(str(p))

    @r.get("/snapshots")
    def list_snapshots():
        """List downloadable dataset snapshots. Public — same posture as
        the rest of the read API."""
        root = _snapshot_root()
        if not root.is_dir():
            return {"snapshots": []}
        out: list[dict] = []
        for p in sorted(root.iterdir()):
            if not p.is_file():
                continue
            if p.suffix not in (".gz", ".zip", ".tar"):
                continue
            out.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "url": f"/rfcai/labeler/snapshots/{p.name}",
            })
        return {"snapshots": out}

    @r.get("/snapshots/{name}")
    def serve_snapshot(name: str):
        """Stream a dataset snapshot to the caller. Public download —
        anyone with the URL can grab it. Filenames are constrained to
        the snapshot root; no path traversal possible."""
        if "/" in name or ".." in name or name.startswith("."):
            raise HTTPException(400, "invalid snapshot name")
        root = _snapshot_root()
        path = root / name
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise HTTPException(400, "invalid path")
        if not path.is_file():
            raise HTTPException(404, "snapshot not found")
        media = "application/gzip" if name.endswith(".gz") else "application/octet-stream"
        return FileResponse(path, media_type=media, filename=name)

    @r.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, error: str = "", next: str = "/rfcai/labeler/"):
        return templates.TemplateResponse(
            request,
            "labeler/login.html",
            {"error": error, "next": next},
        )

    @r.post("/login")
    def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        next: str = Form("/rfcai/labeler/"),
    ):
        from fastapi.responses import RedirectResponse
        user = _auth.authenticate(_users_db_path(), username, password)
        if user is None:
            # Re-render the login page with an error (303 to avoid
            # form re-submission on refresh).
            qs = urllib.parse.urlencode({"error": "invalid credentials", "next": next})
            return RedirectResponse(f"/rfcai/labeler/login?{qs}", status_code=303)
        request.session["user_id"] = user.id
        request.session["username"] = user.username
        return RedirectResponse(next, status_code=303)

    @r.post("/logout")
    def logout(request: Request):
        from fastapi.responses import RedirectResponse
        request.session.clear()
        return RedirectResponse("/rfcai/labeler/", status_code=303)

    @r.get("/admin/users", response_class=HTMLResponse)
    def admin_users_page(
        request: Request,
        user=Depends(require_admin),
        created: str = "",
        deleted: str = "",
        error: str = "",
        password: str = "",
    ):
        users = _auth.list_users(_users_db_path())
        return templates.TemplateResponse(
            request,
            "labeler/admin_users.html",
            {
                "users": users,
                "current_user": user,
                "created": created,
                "deleted": deleted,
                "error": error,
                "new_password": password,
            },
        )

    @r.post("/admin/users")
    def admin_users_create(
        request: Request,
        user=Depends(require_admin),
        username: str = Form(...),
        password: str = Form(""),
    ):
        from fastapi.responses import RedirectResponse
        username = username.strip()
        if not username or not username.replace("_", "").replace("-", "").isalnum():
            qs = urllib.parse.urlencode({
                "error": f"invalid username {username!r} (use letters, digits, _ or -)",
            })
            return RedirectResponse(f"/rfcai/labeler/admin/users?{qs}", status_code=303)

        # Empty password -> generate one and surface in the redirect.
        gen_password = ""
        if not password:
            password = secrets.token_urlsafe(18)
            gen_password = password

        try:
            new_user = _auth.create_user(
                _users_db_path(), username, password, role="admin",
            )
        except _auth.UserExists:
            qs = urllib.parse.urlencode({
                "error": f"username {username!r} already exists",
            })
            return RedirectResponse(f"/rfcai/labeler/admin/users?{qs}", status_code=303)

        qs = urllib.parse.urlencode({
            "created": new_user.username,
            "password": gen_password,  # empty if admin supplied one
        })
        return RedirectResponse(f"/rfcai/labeler/admin/users?{qs}", status_code=303)

    @r.post("/admin/users/{user_id}/delete")
    def admin_users_delete(
        user_id: int,
        user=Depends(require_admin),
    ):
        from fastapi.responses import RedirectResponse
        if user_id == user.id:
            qs = urllib.parse.urlencode({
                "error": "you cannot delete your own account",
            })
            return RedirectResponse(f"/rfcai/labeler/admin/users?{qs}", status_code=303)
        target = None
        for u in _auth.list_users(_users_db_path()):
            if u.id == user_id:
                target = u
                break
        _auth.delete_user(_users_db_path(), user_id)
        deleted_name = target.username if target else f"id-{user_id}"
        qs = urllib.parse.urlencode({"deleted": deleted_name})
        return RedirectResponse(f"/rfcai/labeler/admin/users?{qs}", status_code=303)

    @r.post("/delete", response_class=HTMLResponse)
    def delete(path: str = Form(...), user=Depends(require_admin)):
        p = _safe_path(path)
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                raise HTTPException(500, f"delete failed: {e}")
        # Empty body — HTMX swaps the tile out.
        return Response(content="", media_type="text/html")

    @r.post("/flip", response_class=HTMLResponse)
    def flip(path: str = Form(...), user=Depends(require_admin)):
        p = _safe_path(path)
        if not p.exists():
            return Response(content="", media_type="text/html")
        cls = p.parent.name
        if cls not in CANONICAL_CLASSES:
            raise HTTPException(400, f"source class {cls!r} not canonical")
        family, gender = cls.rsplit("-", 1)
        new_cls = f"{family}-{'F' if gender == 'M' else 'M'}"
        if new_cls not in CANONICAL_CLASSES:
            raise HTTPException(400, f"flip target {new_cls!r} not canonical")
        tgt_dir = _data_root() / new_cls
        tgt_dir.mkdir(parents=True, exist_ok=True)
        stem, ext = p.stem, p.suffix
        dst = tgt_dir / p.name
        n = 1
        while dst.exists():
            dst = tgt_dir / f"{stem}_dup{n}{ext}"
            n += 1
        try:
            shutil.move(str(p), str(dst))
        except Exception as e:
            raise HTTPException(500, f"flip failed: {e}")
        return Response(content="", media_type="text/html")

    @r.post("/bulk-delete", response_class=HTMLResponse)
    def bulk_delete(
        request: Request,
        paths: list[str] = Form(...),
        user=Depends(require_admin),
    ):
        deleted = 0
        for raw in paths:
            try:
                p = _safe_path(raw)
            except HTTPException:
                continue
            if p.exists():
                try:
                    p.unlink()
                    deleted += 1
                except Exception:
                    pass
        return HTMLResponse(f"<div class='success'>Deleted {deleted} image(s).</div>")

    @r.post("/upload-train")
    async def upload_train(
        cls: str = Form(...),
        session: str = Form(""),
        images: list[UploadFile] = File(...),
        user=Depends(require_admin),
    ):
        """Drop phone photos directly into the training set for a class.

        Saved to data/labeled/embedder/<class>/photo_<session>_<stem><ext>.
        Returns JSON {saved: [{cls, path}], errors: [str]} so the Flutter
        Undo flow can stash the server-authoritative path."""
        if cls not in CANONICAL_CLASSES:
            raise HTTPException(400, f"unknown class {cls!r}")
        session_tag = _resolve_session(session)
        out_dir = _data_root() / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[dict] = []
        errors: list[str] = []
        for image in images:
            if not image.filename:
                errors.append("empty filename")
                continue
            ext = Path(image.filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                errors.append(f"bad ext {image.filename!r}")
                continue
            stem = Path(image.filename).stem or f"upload_{int(time.time())}"
            stem = Path(stem).name
            dst = out_dir / f"photo_{session_tag}_{stem}{ext}"
            n = 1
            while dst.exists():
                dst = out_dir / f"photo_{session_tag}_{stem}_dup{n}{ext}"
                n += 1
            data = await image.read()
            dst.write_bytes(data)
            saved.append({"cls": cls, "path": str(dst)})
            _backup_hardlink(dst, cls, _source_backup_root())
        global _signals_cache
        _signals_cache = {}
        return {"saved": saved, "errors": errors}

    @r.post("/upload-test")
    async def upload_test(
        cls: str = Form(...),
        session: str = Form(""),
        images: list[UploadFile] = File(...),
        user=Depends(require_admin),
    ):
        if cls not in CANONICAL_CLASSES:
            raise HTTPException(400, f"unknown class {cls!r}")
        session_tag = _resolve_session(session)
        out_dir = _test_holdout_root() / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[dict] = []
        errors: list[str] = []
        for image in images:
            if not image.filename:
                errors.append("empty filename")
                continue
            ext = Path(image.filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                errors.append(f"bad ext {image.filename!r}")
                continue
            stem = Path(image.filename).stem or f"upload_{int(time.time())}"
            stem = Path(stem).name
            dst = out_dir / f"{session_tag}_{stem}{ext}"
            n = 1
            while dst.exists():
                dst = out_dir / f"{session_tag}_{stem}_dup{n}{ext}"
                n += 1
            data = await image.read()
            dst.write_bytes(data)
            saved.append({"cls": cls, "path": str(dst)})
            _backup_hardlink(dst, cls, _source_backup_root())
        return {"saved": saved, "errors": errors}

    @r.post("/upload-video")
    async def upload_video(
        family: str = Form(...),
        gender: str = Form(...),
        fps: float = Form(5.0),
        sensitivity: float = Form(2.0),
        max_crops: int = Form(5),
        # When true (Flutter Contribute uploader), classify each extracted
        # crop and return JSON with the best prediction alongside the
        # saved-crop summary. Default false preserves the HTML response
        # the web labeler UI expects.
        with_predict: bool = Form(False),
        file: UploadFile = File(...),
        user=Depends(require_admin),
    ):
        if family not in CANONICAL_FAMILIES:
            raise HTTPException(400, f"unknown family {family!r}")
        target_cls = f"{family}-{gender}"
        if target_cls not in CANONICAL_CLASSES:
            raise HTTPException(400, f"unknown class {target_cls!r}")
        if not file.filename:
            raise HTTPException(400, "no file")
        ext = Path(file.filename).suffix.lower()
        if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            raise HTTPException(400, f"unsupported video extension {ext!r}")
        # Stash a copy of the source video for re-extraction later.
        # Assigned name leads with the class so a sorted /videos listing
        # groups by type — much friendlier for the team (Jerry) to scan
        # than `REC_<uuid>.mp4` straight from iOS. The original client
        # filename is preserved in the sidecar for provenance.
        videos_dir = _videos_root()
        videos_dir.mkdir(parents=True, exist_ok=True)
        original_filename = Path(file.filename).name
        assigned_name = _class_video_filename(family, gender, ext)
        saved_video = videos_dir / assigned_name
        n = 1
        while saved_video.exists():
            stem = Path(assigned_name).stem
            saved_video = videos_dir / f"{stem}_dup{n}{ext}"
            n += 1
        data = await file.read()

        # Everything below is CPU/IO-bound (200MB disk write, ffmpeg
        # extract, per-frame Hough, per-crop classify). Doing it on the
        # event-loop thread wedges every other labeler route — sign-in,
        # grid, stats — until the upload finishes. Move it to a worker.
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            raise HTTPException(500, f"ffmpeg not available: {e}")
        out_dir = _data_root() / target_cls
        out_dir.mkdir(parents=True, exist_ok=True)
        do_predict = with_predict and classifier is not None

        def _process_sync() -> tuple[list[str], dict | None, int]:
            saved_video.write_bytes(data)
            # Pick a fresh starting index for "agg_NNNN.jpg" output names.
            existing = []
            for p in out_dir.glob("agg_*.jpg"):
                tail = p.stem[len("agg_"):]
                if tail.isdigit():
                    existing.append(int(tail))
            idx = max(existing) + 1 if existing else 0

            saved_crops_inner: list[str] = []
            best_pred_inner: dict | None = None
            with tempfile.TemporaryDirectory(prefix="upload_video_") as tmp:
                tmpp = Path(tmp)
                subprocess.run(
                    [ff, "-y", "-i", str(saved_video),
                     "-vf", f"fps={fps}", "-q:v", "4",
                     str(tmpp / "f_%04d.jpg")],
                    capture_output=True, check=False,
                )
                frame_paths = sorted(tmpp.glob("f_*.jpg"))
                for fp in frame_paths:
                    bgr = cv2.imread(str(fp))
                    if bgr is None:
                        continue
                    # Reuse the Hough detector with the same defaults the
                    # bulk-extract script tuned (pad_frac=0.35, param2=22).
                    results = detect_connector_crops_hough(
                        bgr,
                        max_crops=int(max_crops),
                        pad_frac=0.35,
                        accumulator_threshold=22,
                    )
                    for r2 in results:
                        out_path = out_dir / f"agg_{idx:04d}.jpg"
                        cv2.imwrite(
                            str(out_path),
                            r2.crop, [cv2.IMWRITE_JPEG_QUALITY, 90],
                        )
                        idx += 1
                        saved_crops_inner.append(str(out_path))
                        if do_predict:
                            rgb = cv2.cvtColor(r2.crop, cv2.COLOR_BGR2RGB)
                            try:
                                cp = classifier.predict(rgb)
                                # ClassifierPrediction → dict so it
                                # round-trips through JSONResponse and
                                # matches the /predict-video schema the
                                # Flutter app already parses.
                                pred = {
                                    "class_name": cp.class_name,
                                    "confidence": float(cp.confidence),
                                    "probabilities": {
                                        k: float(v) for k, v in cp.probabilities.items()
                                    },
                                    "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
                                }
                            except Exception:
                                pred = None
                            if pred is not None and (
                                best_pred_inner is None
                                or pred["confidence"] > best_pred_inner["confidence"]
                            ):
                                best_pred_inner = pred
            return saved_crops_inner, best_pred_inner, len(frame_paths)

        saved_crops, best_pred, n_frames = await run_in_threadpool(_process_sync)

        # Backup the source video to Drive — no-op when env isn't set.
        # Runs in the threadpool too so the HTTP call doesn't block.
        drive_ref = await run_in_threadpool(_drive_backup, saved_video, target_cls)

        # Write the sidecar so the /videos page can later show this video
        # alongside the crops it produced and offer a one-click "delete
        # video + its crops" action. Lives next to the .mp4.
        try:
            sidecar = _video_sidecar_path(saved_video)
            sidecar.write_text(json.dumps({
                "video_filename": saved_video.name,
                "original_filename": original_filename,
                "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "uploaded_by": getattr(user, "username", None),
                "family": family,
                "gender": gender,
                "target_class": target_cls,
                "fps": float(fps),
                "max_crops_per_frame": int(max_crops),
                "n_frames_extracted": n_frames,
                "extracted_crops": saved_crops,
            }, indent=2))
        except Exception as e:
            # Non-fatal — the video + crops are already on disk; the
            # sidecar just makes management easier. Log via the response.
            print(f"[upload-video] sidecar write failed: {e}", flush=True)

        saved_crops_count = len(saved_crops)
        # Bust the signal cache so the new crops show up scored on next grid load.
        global _signals_cache
        _signals_cache = {}

        if with_predict:
            return JSONResponse({
                "saved_crops": saved_crops_count,
                "frames_scanned": n_frames,
                "target_class": target_cls,
                "predictions": [best_pred] if best_pred else [],
                "classifier_loaded": classifier is not None,
                "drive_backup": drive_ref,
            })
        return HTMLResponse(
            f"<div style='color:#4ade80'>Extracted <strong>{saved_crops_count}</strong> crops "
            f"from {saved_video.name} into <code>{target_cls}/</code> "
            f"({n_frames} frames at {fps} fps). Refresh the grid to see them.</div>"
        )

    # --- Video management ---------------------------------------------
    # Lists the .mp4/.mov/etc files in _videos_root() along with the
    # crop count derived from each sidecar (if present). Provides a
    # delete action that removes the video and optionally the agg_*.jpg
    # crops it produced. Pre-existing historical videos that lack a
    # sidecar (the original 3 .MOV training clips) show "no record" and
    # can still be deleted as raw files.

    _VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

    def _infer_family_from_name(stem: str) -> str | None:
        """Best-effort family extraction from a filename stem. Covers the
        historical training clips (2_4mm.MOV, 2_92mm.MOV, 3_5mm.MOV) and
        any future ad-hoc uploads that follow the convention. Returns one
        of CANONICAL_FAMILIES or None if no match."""
        normalized = stem.lower().replace("-", "_")
        for fam in CANONICAL_FAMILIES:
            # Match "2_4mm", "2.4mm", "24mm" etc. against canonical "2.4mm"
            fam_loose = fam.lower().replace(".", "_")
            if fam_loose in normalized or fam.lower() in normalized:
                return fam
        # SMA appears as a token, sometimes lowercase.
        if "sma" in normalized:
            return "SMA" if "SMA" in CANONICAL_FAMILIES else None
        return None

    def _video_record(p: Path) -> dict:
        mtime = p.stat().st_mtime if p.exists() else 0
        # Friendly "YYYY-MM-DD HH:MM" rendering for the file's mtime.
        # When the sidecar is present we prefer its uploaded_at field,
        # but fall back to mtime for the historical .MOV files that
        # predate the sidecar mechanism.
        mtime_display = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(mtime)
        ) if mtime else ""
        info: dict = {
            "name": p.name,
            "path": str(p),
            "size_bytes": p.stat().st_size if p.exists() else 0,
            "mtime": mtime,
            "mtime_display": mtime_display,
            "has_sidecar": False,
            "sidecar_path": None,
            "uploaded_at": None,
            "uploaded_at_display": mtime_display,
            "uploaded_at_source": "mtime",
            "uploaded_by": None,
            "target_class": None,
            "family": None,
            "gender": None,
            "family_source": None,    # "sidecar" | "filename" | None
            "gender_source": None,    # "sidecar" | None
            "fps": None,
            "n_frames_extracted": None,
            "n_crops": 0,
            "n_crops_still_on_disk": 0,
        }
        sidecar = _video_sidecar_path(p)
        if sidecar.exists():
            try:
                manifest = json.loads(sidecar.read_text())
                info["has_sidecar"] = True
                info["sidecar_path"] = str(sidecar)
                info["uploaded_at"] = manifest.get("uploaded_at")
                # Sidecar uploaded_at is a UTC ISO8601 string. Render in
                # local time for display when possible — falls back to
                # the raw string on parse failure.
                if info["uploaded_at"]:
                    try:
                        from datetime import datetime, timezone
                        ts = datetime.strptime(
                            info["uploaded_at"], "%Y-%m-%dT%H:%M:%SZ",
                        ).replace(tzinfo=timezone.utc).astimezone()
                        info["uploaded_at_display"] = ts.strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        info["uploaded_at_source"] = "sidecar"
                    except Exception:
                        info["uploaded_at_display"] = info["uploaded_at"]
                        info["uploaded_at_source"] = "sidecar-raw"
                info["uploaded_by"] = manifest.get("uploaded_by")
                info["target_class"] = manifest.get("target_class")
                info["family"] = manifest.get("family")
                info["gender"] = manifest.get("gender")
                if info["family"]:
                    info["family_source"] = "sidecar"
                if info["gender"]:
                    info["gender_source"] = "sidecar"
                info["fps"] = manifest.get("fps")
                info["n_frames_extracted"] = manifest.get("n_frames_extracted")
                crops = manifest.get("extracted_crops") or []
                info["n_crops"] = len(crops)
                info["n_crops_still_on_disk"] = sum(
                    1 for c in crops if Path(c).exists()
                )
            except Exception as e:
                # Corrupted sidecar — treat as "no record" so the user
                # can still delete the .mp4. Surface the error in the UI.
                info["sidecar_error"] = str(e)

        # If the sidecar is missing OR doesn't include family, fall back
        # to a filename heuristic so the historical .MOV files don't
        # display as Unknown. Gender can't be inferred from filename in
        # the existing corpus, so it stays None until the user annotates.
        if not info["family"]:
            inferred = _infer_family_from_name(p.stem)
            if inferred:
                info["family"] = inferred
                info["family_source"] = "filename"
                if info["gender"] and (
                    f"{inferred}-{info['gender']}" in CANONICAL_CLASSES
                ):
                    info["target_class"] = f"{inferred}-{info['gender']}"
        return info

    @r.get("/videos", response_class=HTMLResponse)
    def videos_list(request: Request):
        # Public read — anyone can see what's been uploaded. Writes
        # (delete) still require an admin session.
        vroot = _videos_root()
        vroot.mkdir(parents=True, exist_ok=True)
        files = sorted(
            (p for p in vroot.iterdir()
             if p.is_file() and p.suffix.lower() in _VIDEO_EXTS),
            key=lambda p: -p.stat().st_mtime,   # newest first
        )
        records = [_video_record(p) for p in files]
        return templates.TemplateResponse(
            request,
            "labeler/videos.html",
            {
                "videos": records,
                "n_total": len(records),
                "n_with_crops": sum(1 for r in records if r["n_crops"] > 0),
                "videos_root": str(vroot),
            },
        )

    @r.post("/videos/annotate", response_class=HTMLResponse)
    def videos_annotate(
        path: str = Form(...),
        family: str = Form(...),
        gender: str = Form(...),
        user=Depends(require_admin),
    ):
        """Retroactively label a video that lacks a sidecar (legacy
        .MOVs, 504-orphaned uploads, etc.). Writes a minimal manifest so
        future visits to /videos show the connector type; does NOT touch
        any extracted crops — those are recorded separately when the
        video flows through /upload-video."""
        if family not in CANONICAL_FAMILIES:
            raise HTTPException(400, f"unknown family {family!r}")
        target_cls = f"{family}-{gender}"
        if target_cls not in CANONICAL_CLASSES:
            raise HTTPException(400, f"unknown class {target_cls!r}")
        target = _safe_path(path)
        if not target.exists() or target.suffix.lower() not in _VIDEO_EXTS:
            raise HTTPException(404, "video not found")
        sidecar = _video_sidecar_path(target)
        existing: dict = {}
        if sidecar.exists():
            try:
                existing = json.loads(sidecar.read_text())
            except Exception:
                existing = {}

        # If the file doesn't already lead with the class, rename it +
        # its sidecar so the directory listing stays self-documenting.
        # The original filename is preserved in the sidecar as
        # original_filename for provenance — useful when comparing
        # against client logs / nginx access logs.
        renamed = False
        if not _filename_already_class_prefixed(target.name, family, gender):
            new_name = _class_video_filename(
                family, gender, target.suffix, when=target.stat().st_mtime,
            )
            new_target = target.parent / new_name
            n = 1
            while new_target.exists() and new_target != target:
                stem = Path(new_name).stem
                new_target = target.parent / f"{stem}_dup{n}{target.suffix.lower()}"
                n += 1
            existing.setdefault("original_filename", target.name)
            try:
                target.rename(new_target)
                if sidecar.exists():
                    new_sidecar = _video_sidecar_path(new_target)
                    sidecar.rename(new_sidecar)
                    sidecar = new_sidecar
                else:
                    sidecar = _video_sidecar_path(new_target)
                target = new_target
                renamed = True
            except Exception as e:
                print(f"[videos-annotate] rename failed: {e}", flush=True)

        # Preserve any existing manifest fields (uploaded_at, crops,
        # uploaded_by) and just patch the family/gender. For brand-new
        # annotations the file gets seeded with sane defaults.
        existing.update({
            "video_filename": target.name,
            "family": family,
            "gender": gender,
            "target_class": target_cls,
        })
        existing.setdefault("uploaded_at",
            time.strftime("%Y-%m-%dT%H:%M:%SZ",
                          time.gmtime(target.stat().st_mtime)))
        existing.setdefault("uploaded_by", getattr(user, "username", None))
        existing.setdefault("extracted_crops", [])
        existing.setdefault("n_frames_extracted", None)
        existing.setdefault("fps", None)
        existing["annotated_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
        )
        existing["annotated_by"] = getattr(user, "username", None)
        if renamed:
            existing["renamed_at"] = existing["annotated_at"]
        sidecar.write_text(json.dumps(existing, indent=2))
        # Return a tiny HTMX-swappable confirmation that the row script
        # can use to reload itself.
        return HTMLResponse(
            f'<div hx-trigger="load delay:300ms" '
            f'hx-get="/rfcai/labeler/videos" hx-target="body" '
            f'hx-swap="outerHTML" '
            f'style="color: var(--green); font-size: 12px;">'
            f'Saved as <strong>{target_cls}</strong>. Refreshing…</div>'
        )

    @r.get("/videos/download/{name}")
    def videos_download(name: str):
        """Serve a training video as a download. Public read so the
        team (Jerry, etc.) can grab the cleaned-up source clips via a
        shareable URL — same posture as /snapshots. Filenames are
        constrained to the videos root; no path traversal possible."""
        if "/" in name or ".." in name or name.startswith("."):
            raise HTTPException(400, "invalid video name")
        root = _videos_root()
        path = root / name
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise HTTPException(400, "invalid path")
        if not path.is_file():
            raise HTTPException(404, "video not found")
        if path.suffix.lower() not in _VIDEO_EXTS:
            raise HTTPException(400, "not a video file")
        ext = path.suffix.lower()
        media = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
            ".mkv": "video/x-matroska",
            ".avi": "video/x-msvideo",
        }.get(ext, "application/octet-stream")
        return FileResponse(path, media_type=media, filename=name)

    @r.post("/videos/delete", response_class=HTMLResponse)
    def videos_delete(
        path: str = Form(...),
        also_crops: bool = Form(False),
        user=Depends(require_admin),
    ):
        target = _safe_path(path)
        if not target.exists():
            # Already gone — return empty (HTMX will swap the row out
            # either way).
            return Response(content="", media_type="text/html")
        if target.suffix.lower() not in _VIDEO_EXTS:
            raise HTTPException(400, "not a video file")

        deleted_crops = 0
        if also_crops:
            sidecar = _video_sidecar_path(target)
            if sidecar.exists():
                try:
                    manifest = json.loads(sidecar.read_text())
                    for crop_path in manifest.get("extracted_crops", []):
                        cp = Path(crop_path)
                        if cp.exists():
                            try:
                                cp.unlink()
                                deleted_crops += 1
                            except Exception as e:
                                print(
                                    f"[videos-delete] crop unlink failed "
                                    f"{cp}: {e}",
                                    flush=True,
                                )
                except Exception as e:
                    print(f"[videos-delete] sidecar parse failed: {e}",
                          flush=True)

        # Sidecar + video itself
        sidecar = _video_sidecar_path(target)
        if sidecar.exists():
            try:
                sidecar.unlink()
            except Exception:
                pass
        try:
            target.unlink()
        except Exception as e:
            raise HTTPException(500, f"video delete failed: {e}")

        # Bust the signal cache so the grid stops showing crops that
        # might have just been removed.
        global _signals_cache
        _signals_cache = {}

        return Response(content="", media_type="text/html")

    @r.post("/api-tokens/exchange")
    def api_tokens_exchange(
        username: str = Form(...),
        password: str = Form(...),
        name: str = Form("API client"),
    ):
        """Exchange username+password for a long-lived API token.

        Returns the plaintext token exactly once — the caller stores it
        and sends it as `Authorization: Bearer <token>` on subsequent
        requests. The server stores only the scrypt hash.

        Public route. Rate limiting is delegated to nginx; if abuse
        becomes a concern we can add per-IP throttling here.
        """
        db_path = _users_db_path()
        user = _auth.authenticate(db_path, username, password)
        if user is None:
            raise HTTPException(401, "invalid credentials")
        # Cap name length so admins listing tokens see something sensible.
        token_name = (name or "API client")[:64].strip() or "API client"
        _record, plaintext = _auth.create_token(
            db_path, user.id, name=token_name,
        )
        return {
            "token": plaintext,
            "name": token_name,
            "user": {
                "id": user.id,
                "username": user.username,
                "role": user.role,
            },
        }

    return r


def _filter_qs(cls: list[str], only_no_circle: bool, only_multi: bool,
               hide_dups: bool, blur_threshold: int, sort: str,
               per_page: int) -> str:
    parts = [("cls", c) for c in cls]
    if only_no_circle: parts.append(("only_no_circle", "true"))
    if only_multi: parts.append(("only_multi", "true"))
    if hide_dups: parts.append(("hide_dups", "true"))
    if blur_threshold > 0: parts.append(("blur_threshold", str(blur_threshold)))
    if sort != "class": parts.append(("sort", sort))
    if per_page != 32: parts.append(("per_page", str(per_page)))
    return urllib.parse.urlencode(parts)
