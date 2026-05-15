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
from fastapi.responses import HTMLResponse, FileResponse, Response
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
    """Validate that `raw` points inside the labeled-data root or the
    test-holdout root. Prevents path-traversal attacks via a crafted
    form value. Both roots are accepted so that /delete can undo uploads
    made via /upload-test as well as /upload-train."""
    try:
        candidate = Path(raw).resolve()
    except Exception:
        raise HTTPException(400, "bad path")
    for root in (_data_root(), _test_holdout_root()):
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise HTTPException(400, "path outside data roots")


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


def create_router() -> APIRouter:
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

    @r.post("/upload-video", response_class=HTMLResponse)
    async def upload_video(
        family: str = Form(...),
        gender: str = Form(...),
        fps: float = Form(5.0),
        sensitivity: float = Form(2.0),
        max_crops: int = Form(5),
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
        videos_dir = _videos_root()
        videos_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file.filename).stem or f"upload_{int(time.time())}"
        stem = Path(stem).name
        saved_video = videos_dir / f"{stem}{ext}"
        n = 1
        while saved_video.exists():
            saved_video = videos_dir / f"{stem}_dup{n}{ext}"
            n += 1
        data = await file.read()
        saved_video.write_bytes(data)

        # Extract frames at requested fps, run Hough, dump crops to
        # the target class directory (user can still Flip mistakes
        # via the labeler grid).
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            raise HTTPException(500, f"ffmpeg not available: {e}")
        # target_cls already validated above
        out_dir = _data_root() / target_cls
        out_dir.mkdir(parents=True, exist_ok=True)
        # Pick a fresh starting index for "agg_NNNN.jpg" output names.
        existing = []
        for p in out_dir.glob("agg_*.jpg"):
            tail = p.stem[len("agg_"):]
            if tail.isdigit():
                existing.append(int(tail))
        idx = max(existing) + 1 if existing else 0

        saved_crops = 0
        with tempfile.TemporaryDirectory(prefix="upload_video_") as tmp:
            tmpp = Path(tmp)
            subprocess.run(
                [ff, "-y", "-i", str(saved_video),
                 "-vf", f"fps={fps}", "-q:v", "4",
                 str(tmpp / "f_%04d.jpg")],
                capture_output=True, check=False,
            )
            frames = sorted(tmpp.glob("f_*.jpg"))
            for fp in frames:
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
                    cv2.imwrite(
                        str(out_dir / f"agg_{idx:04d}.jpg"),
                        r2.crop, [cv2.IMWRITE_JPEG_QUALITY, 90],
                    )
                    idx += 1
                    saved_crops += 1

        # Bust the signal cache so the new crops show up scored on next grid load.
        global _signals_cache
        _signals_cache = {}

        return HTMLResponse(
            f"<div style='color:#4ade80'>Extracted <strong>{saved_crops}</strong> crops "
            f"from {saved_video.name} into <code>{target_cls}/</code> "
            f"({len(frames)} frames at {fps} fps). Refresh the grid to see them.</div>"
        )

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
