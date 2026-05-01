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
import secrets
import shutil
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates


CANONICAL_CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]


def _data_root() -> Path:
    """The labeled-data folder this service writes into. Resolved fresh
    every call so a service restart picks up env changes."""
    return Path(os.environ.get(
        "RFCAI_LABELED_DIR",
        "/opt/rfcai/repo/training/data/labeled/embedder",
    )).resolve()


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
    """Validate that `raw` points inside the labeled-data root. Prevents
    path-traversal attacks via a crafted form value."""
    root = _data_root()
    try:
        candidate = Path(raw).resolve()
    except Exception:
        raise HTTPException(400, "bad path")
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(400, "path outside data root")
    return candidate


def _class_counts() -> dict[str, int]:
    root = _data_root()
    out = {}
    for cls in CANONICAL_CLASSES:
        d = root / cls
        out[cls] = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
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


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def create_router() -> APIRouter:
    # Prefix matches the public URL path. nginx on aired.com forwards
    # /rfcai/* to the predict service backend with the /rfcai prefix
    # preserved (for paths nginx doesn't have an explicit rewrite for),
    # so backend routes must include /rfcai/ to be reachable externally.
    r = APIRouter(prefix="/rfcai/labeler", tags=["labeler"])

    @r.get("/", response_class=HTMLResponse)
    def index(request: Request, _: str = Depends(_require_basic_auth)):
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
        _: str = Depends(_require_basic_auth),
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
    def serve_img(path: str, _: str = Depends(_require_basic_auth)):
        p = _safe_path(path)
        if not p.exists():
            raise HTTPException(404, "not found")
        return FileResponse(str(p))

    @r.post("/delete", response_class=HTMLResponse)
    def delete(path: str = Form(...), _: str = Depends(_require_basic_auth)):
        p = _safe_path(path)
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                raise HTTPException(500, f"delete failed: {e}")
        # Empty body — HTMX swaps the tile out.
        return Response(content="", media_type="text/html")

    @r.post("/flip", response_class=HTMLResponse)
    def flip(path: str = Form(...), _: str = Depends(_require_basic_auth)):
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
        _: str = Depends(_require_basic_auth),
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
