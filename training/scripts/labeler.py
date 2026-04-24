#!/usr/bin/env python3
"""
Image labeling UI for the RF connector training set.

Three workflows in one tool:
  1. Auto-fetch: per-class "Fetch N" button runs Bing Image Search via icrawler
     and dumps candidates straight into the labeled folder. You then review the
     grid and click X on anything that doesn't fit.
  2. Drag-and-drop: drag images from a Digikey/Mouser tab into the matching
     class column.
  3. URL paste: paste an image URL into the box at the top of any column.

Run:
    python scripts/labeler.py

Then open http://localhost:5179.
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory
from icrawler.builtin import BingImageCrawler


CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]

# Default Bing search queries per class. The 3.5mm precision RF connector is
# easily confused with audio jacks; the query bias toward "microwave" or
# "precision" helps.
DEFAULT_QUERIES = {
    "SMA-M":   "SMA male connector RF coaxial plug",
    "SMA-F":   "SMA female connector RF coaxial jack",
    "3.5mm-M": "3.5mm male precision microwave RF connector",
    "3.5mm-F": "3.5mm female precision microwave RF connector",
    "2.92mm-M": "2.92mm K connector male RF microwave",
    "2.92mm-F": "2.92mm K connector female RF microwave",
    "2.4mm-M": "2.4mm male microwave RF connector precision",
    "2.4mm-F": "2.4mm female microwave RF connector precision",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "image"


def load_files(out_dir: Path, cls: str) -> list[dict]:
    d = out_dir / cls
    if not d.is_dir():
        return []
    items = []
    for p in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_file() or p.name.startswith("."):
            continue
        items.append({"name": p.name, "size": p.stat().st_size})
    return items


def build_app(out_dir: Path) -> Flask:
    app = Flask(__name__, static_folder=None)
    fetch_locks: dict[str, threading.Lock] = {cls: threading.Lock() for cls in CLASSES}
    fetch_state: dict[str, dict] = {cls: {"running": False, "query": None} for cls in CLASSES}

    @app.route("/")
    def index():
        return INDEX_HTML

    @app.route("/api/state")
    def state():
        result = {}
        for cls in CLASSES:
            files = load_files(out_dir, cls)
            result[cls] = {
                "count": len(files),
                "files": files,
                "fetching": fetch_state[cls]["running"],
                "query": fetch_state[cls]["query"] or DEFAULT_QUERIES.get(cls, cls),
            }
        return jsonify(result)

    @app.route("/api/image/<cls>/<path:filename>")
    def image(cls, filename):
        if cls not in CLASSES:
            return jsonify({"error": "unknown class"}), 404
        return send_from_directory(out_dir / cls, filename)

    @app.route("/api/fetch/<cls>", methods=["POST"])
    def fetch(cls):
        if cls not in CLASSES:
            return jsonify({"error": f"unknown class {cls}"}), 400
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or DEFAULT_QUERIES.get(cls, cls)).strip()
        try:
            n = max(1, min(100, int(body.get("n", 30))))
        except (TypeError, ValueError):
            n = 30

        if not fetch_locks[cls].acquire(blocking=False):
            return jsonify({"error": "fetch already in progress for this class"}), 409

        fetch_state[cls] = {"running": True, "query": query}

        def worker():
            try:
                target_dir = out_dir / cls
                target_dir.mkdir(parents=True, exist_ok=True)
                crawler = BingImageCrawler(
                    storage={"root_dir": str(target_dir)},
                    feeder_threads=1,
                    parser_threads=1,
                    downloader_threads=4,
                    log_level=30,  # WARNING; suppress per-image INFO chatter
                )
                crawler.crawl(
                    keyword=query,
                    max_num=n,
                    file_idx_offset="auto",
                )
            except Exception as e:
                print(f"[labeler] fetch {cls!r} failed: {e}", file=sys.stderr)
            finally:
                fetch_state[cls] = {"running": False, "query": query}
                fetch_locks[cls].release()

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"started": True, "query": query, "n": n})

    @app.route("/api/delete/<cls>/<path:filename>", methods=["DELETE"])
    def delete(cls, filename):
        if cls not in CLASSES:
            return jsonify({"error": "unknown class"}), 404
        # Prevent path traversal.
        target = (out_dir / cls / filename).resolve()
        if (out_dir / cls).resolve() not in target.parents:
            return jsonify({"error": "path traversal"}), 400
        if not target.is_file():
            return jsonify({"error": "not found"}), 404
        target.unlink()
        return jsonify({"deleted": filename, "count": len(load_files(out_dir, cls))})

    @app.route("/api/upload/<cls>", methods=["POST"])
    def upload(cls):
        if cls not in CLASSES:
            return jsonify({"error": f"unknown class {cls}"}), 400

        target_dir = out_dir / cls
        target_dir.mkdir(parents=True, exist_ok=True)

        if "file" in request.files:
            f = request.files["file"]
            raw = f.read()
            if not raw:
                return jsonify({"error": "empty body"}), 400
            ext = guess_ext(f.mimetype, f.filename)
            base = sanitize(Path(f.filename or "image").stem)[:60]
            out_path = unique_path(target_dir, base, ext)
            out_path.write_bytes(raw)
            return jsonify({"saved": out_path.name, "count": len(load_files(out_dir, cls))})

        body = request.get_json(silent=True) or {}
        url = (body.get("url") or "").strip()
        if not url:
            return jsonify({"error": "no file part and no url provided"}), 400

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"},
                timeout=20,
            )
        except requests.RequestException as e:
            return jsonify({"error": f"request failed: {type(e).__name__}: {e}"}), 502
        if resp.status_code != 200:
            return jsonify({"error": f"HTTP {resp.status_code} from upstream"}), 502
        if not resp.headers.get("Content-Type", "").startswith("image/"):
            return jsonify({"error": f"unexpected Content-Type {resp.headers.get('Content-Type')!r}"}), 502

        ext = guess_ext(resp.headers.get("Content-Type"), url)
        base = sanitize(Path(url.split("?", 1)[0]).stem)[:60] or "image"
        out_path = unique_path(target_dir, base, ext)
        out_path.write_bytes(resp.content)
        return jsonify({"saved": out_path.name, "count": len(load_files(out_dir, cls))})

    return app


def guess_ext(content_type: str | None, hint: str | None) -> str:
    ct = (content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "gif" in ct: return ".gif"
    h = (hint or "").lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if h.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def unique_path(target_dir: Path, base: str, ext: str) -> Path:
    candidate = target_dir / f"{base}{ext}"
    if not candidate.exists():
        return candidate
    for i in range(1, 10000):
        candidate = target_dir / f"{base}_{i}{ext}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Too many name collisions")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RF Connector Labeler</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 12px; background: #111; color: #eee; }
  h1 { font-size: 18px; margin: 0 0 4px 0; }
  .help { font-size: 12px; color: #aaa; margin-bottom: 12px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }

  .col {
    border: 1px solid #2a2a2a; border-radius: 8px; padding: 10px;
    background: #1a1a1a; transition: background 120ms, border-color 120ms;
    display: flex; flex-direction: column; min-height: 200px;
  }
  .col.hover { background: #223a2a; border-color: #4caf50; }
  .col h2 { font-size: 14px; margin: 0 0 6px 0; display: flex; justify-content: space-between; align-items: center; }
  .count { background: #333; color: #fff; border-radius: 12px; padding: 1px 9px; font-size: 12px; font-weight: 600; }

  .controls { display: flex; gap: 4px; margin-bottom: 6px; }
  .controls input { flex: 1; padding: 4px 6px; border: 1px solid #333; background: #0c0c0c; color: #ddd; border-radius: 3px; font-size: 11px; min-width: 0; }
  .controls button { padding: 4px 8px; border: 1px solid #2a6e2a; background: #1d4d1d; color: #ddf; border-radius: 3px; font-size: 11px; cursor: pointer; }
  .controls button:disabled { background: #333; color: #888; cursor: not-allowed; border-color: #444; }
  .controls button.fetching { background: #654a1d; border-color: #aa8030; }

  .urlbox { width: 100%; padding: 4px 6px; border: 1px solid #333; background: #0c0c0c; color: #ddd; border-radius: 3px; font-size: 11px; margin-bottom: 6px; }

  .thumbs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; }
  .thumb { position: relative; aspect-ratio: 1; background: #000; border-radius: 3px; overflow: hidden; }
  .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .thumb .x {
    position: absolute; top: 2px; right: 2px; width: 18px; height: 18px;
    background: rgba(220, 50, 50, 0.85); color: #fff; border: none; border-radius: 50%;
    font-size: 12px; line-height: 18px; padding: 0; cursor: pointer; opacity: 0; transition: opacity 80ms;
  }
  .thumb:hover .x { opacity: 1; }
  .thumb .x:hover { background: rgb(255, 60, 60); }

  .empty { color: #555; font-size: 11px; text-align: center; padding: 20px 0; }

  #log {
    font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: #888;
    max-height: 100px; overflow-y: auto; background: #0a0a0a; border: 1px solid #222;
    border-radius: 4px; padding: 6px; margin-top: 12px;
  }
  .ok { color: #6cce6c; }
  .err { color: #ff8a8a; }
  .info { color: #6ca8ce; }
</style>
</head>
<body>
<h1>RF Connector Labeler</h1>
<div class="help">
  Click <b>Fetch</b> on each class to auto-pull ~30 candidate images from Bing.
  Hover any thumbnail and click the red X to remove it. You can also drag images
  from a Digikey tab into a column, or paste an image URL.
</div>

<div class="grid" id="grid"></div>

<div id="log"></div>

<script>
const CLASSES = {{CLASSES_JSON}};
const grid = document.getElementById('grid');
const log = document.getElementById('log');

function logLine(text, cls) {
  const div = document.createElement('div');
  div.className = cls || '';
  div.textContent = '[' + new Date().toLocaleTimeString() + '] ' + text;
  log.prepend(div);
  while (log.children.length > 60) log.removeChild(log.lastChild);
}

function makeColumn(cls) {
  const col = document.createElement('div');
  col.className = 'col';
  col.dataset.cls = cls;
  col.innerHTML = `
    <h2><span>${cls}</span><span class="count" id="count-${cls}">0</span></h2>
    <div class="controls">
      <input type="text" id="query-${cls}" placeholder="search query">
      <button id="fetch-${cls}">Fetch 30</button>
    </div>
    <input type="text" class="urlbox" id="url-${cls}" placeholder="or paste image URL + Enter">
    <div class="thumbs" id="thumbs-${cls}"><div class="empty">No images yet</div></div>
  `;
  const fetchBtn = col.querySelector('#fetch-' + cls);
  const queryInput = col.querySelector('#query-' + cls);
  const urlInput = col.querySelector('#url-' + cls);

  fetchBtn.addEventListener('click', () => fetchClass(cls, queryInput.value.trim()));
  urlInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && urlInput.value.trim()) {
      uploadUrl(cls, urlInput.value.trim());
      urlInput.value = '';
    }
  });

  ['dragenter', 'dragover'].forEach(ev =>
    col.addEventListener(ev, e => { e.preventDefault(); col.classList.add('hover'); }));
  ['dragleave', 'drop'].forEach(ev =>
    col.addEventListener(ev, e => col.classList.remove('hover')));
  col.addEventListener('drop', e => {
    e.preventDefault();
    handleDrop(cls, e);
  });

  return col;
}

CLASSES.forEach(cls => grid.appendChild(makeColumn(cls)));

async function fetchClass(cls, query) {
  const body = { n: 30 };
  if (query) body.query = query;
  try {
    const r = await fetch('/api/fetch/' + encodeURIComponent(cls), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (r.ok) {
      logLine('fetching ' + cls + ' (' + j.query + ')...', 'info');
    } else {
      logLine('fetch ' + cls + ': ' + (j.error || r.statusText), 'err');
    }
  } catch (e) {
    logLine('fetch ' + cls + ' network error: ' + e, 'err');
  }
}

async function uploadFile(cls, file) {
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/upload/' + encodeURIComponent(cls), { method: 'POST', body: fd });
    const j = await r.json();
    if (r.ok) {
      logLine('uploaded → ' + cls + '/' + j.saved, 'ok');
    } else {
      logLine('upload ' + cls + ': ' + (j.error || r.statusText), 'err');
    }
  } catch (e) {
    logLine('upload network error: ' + e, 'err');
  }
}

async function uploadUrl(cls, url) {
  try {
    const r = await fetch('/api/upload/' + encodeURIComponent(cls), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const j = await r.json();
    if (r.ok) {
      logLine('saved → ' + cls + '/' + j.saved, 'ok');
    } else {
      logLine(cls + ' URL: ' + (j.error || r.statusText), 'err');
    }
  } catch (e) {
    logLine('URL network error: ' + e, 'err');
  }
}

async function handleDrop(cls, e) {
  const files = Array.from(e.dataTransfer.files || []).filter(f => f.type.startsWith('image/'));
  for (const f of files) await uploadFile(cls, f);
  if (files.length === 0) {
    const url = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain');
    if (url && /^https?:/.test(url.trim())) {
      await uploadUrl(cls, url.trim());
    } else {
      logLine('drop: no image found in payload', 'err');
    }
  }
}

async function deleteImage(cls, filename) {
  try {
    const r = await fetch('/api/delete/' + encodeURIComponent(cls) + '/' + encodeURIComponent(filename), { method: 'DELETE' });
    const j = await r.json();
    if (r.ok) {
      logLine('deleted ' + cls + '/' + filename, 'info');
    } else {
      logLine('delete ' + cls + ': ' + (j.error || r.statusText), 'err');
    }
  } catch (e) {
    logLine('delete network error: ' + e, 'err');
  }
}

let lastSeenFiles = {};
function renderColumn(cls, info) {
  const countEl = document.getElementById('count-' + cls);
  const thumbsEl = document.getElementById('thumbs-' + cls);
  const fetchBtn = document.getElementById('fetch-' + cls);
  const queryInput = document.getElementById('query-' + cls);

  countEl.textContent = info.count;

  if (info.fetching) {
    fetchBtn.textContent = 'Fetching…';
    fetchBtn.classList.add('fetching');
    fetchBtn.disabled = true;
  } else {
    fetchBtn.textContent = 'Fetch 30';
    fetchBtn.classList.remove('fetching');
    fetchBtn.disabled = false;
  }
  if (queryInput && !queryInput.value && info.query) queryInput.placeholder = info.query;

  // Only redraw thumbnails when the file list changes (cheap polling).
  const sig = info.files.map(f => f.name).join('|');
  if (lastSeenFiles[cls] === sig) return;
  lastSeenFiles[cls] = sig;

  if (info.files.length === 0) {
    thumbsEl.innerHTML = '<div class="empty">No images yet</div>';
    return;
  }
  thumbsEl.innerHTML = '';
  for (const f of info.files) {
    const t = document.createElement('div');
    t.className = 'thumb';
    t.innerHTML = `
      <img src="/api/image/${encodeURIComponent(cls)}/${encodeURIComponent(f.name)}" loading="lazy" alt="">
      <button class="x" title="Remove">×</button>
    `;
    t.querySelector('.x').addEventListener('click', async (e) => {
      e.stopPropagation();
      t.style.opacity = '0.4';
      await deleteImage(cls, f.name);
      // Optimistic remove; next poll syncs truth.
      t.remove();
    });
    thumbsEl.appendChild(t);
  }
}

async function refresh() {
  try {
    const r = await fetch('/api/state');
    const j = await r.json();
    for (const cls of CLASSES) {
      if (j[cls]) renderColumn(cls, j[cls]);
    }
  } catch (e) { /* ignore */ }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
""".replace("{{CLASSES_JSON}}", str(CLASSES).replace("'", '"'))


def main():
    ap = argparse.ArgumentParser()
    default_out = Path(__file__).resolve().parent.parent / "data" / "labeled" / "embedder"
    ap.add_argument("--out-dir", type=Path, default=default_out)
    ap.add_argument("--port", type=int, default=5179)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    app = build_app(out_dir)
    print(f"Saving to: {out_dir}")
    print(f"Open: http://localhost:{args.port}/")
    print("Press Ctrl+C to stop.")
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    sys.exit(main())
