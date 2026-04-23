#!/usr/bin/env python3
"""
Drag-and-drop labeling UI for collecting connector images.

Run:
    python scripts/labeler.py

Then open http://localhost:5179 in your browser. Drag images from any other
tab (Digikey, Mouser, etc.) into the matching class column. Saved files land
in training/data/labeled/embedder/<CLASS>/.

The app accepts:
  - Image files dragged from the OS file system
  - Images dragged from another browser tab (Chrome/Firefox copy the bytes)
  - Pasted image URLs (paste into the URL box at the top of any column)
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory


CLASSES = [
    "SMA-M", "SMA-F",
    "3.5mm-M", "3.5mm-F",
    "2.92mm-M", "2.92mm-F",
    "2.4mm-M", "2.4mm-F",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "image"


def load_count(out_dir: Path, cls: str) -> int:
    d = out_dir / cls
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and not p.name.startswith("."))


def build_app(out_dir: Path) -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.route("/")
    def index():
        return INDEX_HTML

    @app.route("/api/counts")
    def counts():
        return jsonify({cls: load_count(out_dir, cls) for cls in CLASSES})

    @app.route("/api/upload/<cls>", methods=["POST"])
    def upload(cls: str):
        if cls not in CLASSES:
            return jsonify({"error": f"unknown class {cls}"}), 400

        target_dir = out_dir / cls
        target_dir.mkdir(parents=True, exist_ok=True)

        # Two upload modes: a file part (drag-drop) or a JSON URL.
        if "file" in request.files:
            f = request.files["file"]
            raw = f.read()
            if not raw:
                return jsonify({"error": "empty body"}), 400

            content_type = f.mimetype or ""
            ext = guess_ext(content_type, f.filename)
            base = sanitize(Path(f.filename or "image").stem)[:60]
            out_path = unique_path(target_dir, base, ext)
            out_path.write_bytes(raw)
            return jsonify({"saved": str(out_path.relative_to(out_dir.parent.parent.parent if out_path.is_absolute() else Path('.'))), "filename": out_path.name, "count": load_count(out_dir, cls)})

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

        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return jsonify({"error": f"unexpected Content-Type {content_type!r}"}), 502

        ext = guess_ext(content_type, url)
        base = sanitize(Path(url.split("?", 1)[0]).stem)[:60] or "image"
        out_path = unique_path(target_dir, base, ext)
        out_path.write_bytes(resp.content)
        return jsonify({"saved": out_path.name, "count": load_count(out_dir, cls)})

    return app


def guess_ext(content_type: str, hint: str) -> str:
    ct = (content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "gif" in ct: return ".gif"
    # Fall back to URL/file extension
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
<title>RF Connector Image Labeler</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 12px; background: #111; color: #eee; }
  h1 { font-size: 18px; margin: 0 0 8px 0; }
  .help { font-size: 13px; color: #aaa; margin-bottom: 16px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .col {
    border: 2px dashed #444; border-radius: 8px; padding: 12px;
    min-height: 220px; background: #1c1c1c; transition: background 120ms, border-color 120ms;
  }
  .col.hover { background: #223a2a; border-color: #4caf50; }
  .col h2 { font-size: 16px; margin: 0 0 6px 0; display: flex; justify-content: space-between; align-items: center; }
  .count { background: #333; color: #fff; border-radius: 12px; padding: 1px 9px; font-size: 13px; font-weight: 600; }
  input[type=text] {
    width: 100%; padding: 6px 8px; margin-top: 6px; border: 1px solid #333;
    background: #0c0c0c; color: #eee; border-radius: 4px; font-size: 12px; box-sizing: border-box;
  }
  .drophint { color: #777; font-size: 12px; margin-top: 4px; text-align: center; }
  #log { font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: #888; max-height: 120px; overflow-y: auto; background: #0a0a0a; border: 1px solid #222; border-radius: 4px; padding: 6px; margin-top: 12px; }
  .ok { color: #6cce6c; }
  .err { color: #ff8a8a; }
</style>
</head>
<body>
<h1>RF Connector Image Labeler</h1>
<div class="help">
  Drag images from a Digikey/Mouser tab directly into the matching class column.
  You can also paste an image URL into the URL box and press Enter. Files land
  in <code>training/data/labeled/embedder/&lt;CLASS&gt;/</code>.
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
  while (log.children.length > 100) log.removeChild(log.lastChild);
}

function makeColumn(cls) {
  const col = document.createElement('div');
  col.className = 'col';
  col.dataset.cls = cls;
  col.innerHTML = `
    <h2><span>${cls}</span><span class="count" id="count-${cls}">0</span></h2>
    <input type="text" placeholder="Paste image URL, press Enter" />
    <div class="drophint">Drag images here</div>
  `;
  const input = col.querySelector('input');
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && input.value.trim()) {
      uploadUrl(cls, input.value.trim());
      input.value = '';
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

async function uploadFile(cls, file) {
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/upload/' + encodeURIComponent(cls), { method: 'POST', body: fd });
    const j = await r.json();
    if (r.ok) {
      document.getElementById('count-' + cls).textContent = j.count;
      logLine('saved → ' + cls + '/' + j.filename, 'ok');
    } else {
      logLine('error ' + cls + ': ' + (j.error || r.statusText), 'err');
    }
  } catch (e) {
    logLine('network error: ' + e, 'err');
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
      document.getElementById('count-' + cls).textContent = j.count;
      logLine('saved → ' + cls + '/' + j.saved, 'ok');
    } else {
      logLine('error ' + cls + ': ' + (j.error || r.statusText), 'err');
    }
  } catch (e) {
    logLine('network error: ' + e, 'err');
  }
}

async function handleDrop(cls, e) {
  // Files dragged from disk
  const files = Array.from(e.dataTransfer.files || []).filter(f => f.type.startsWith('image/'));
  for (const f of files) {
    await uploadFile(cls, f);
  }

  // Image dragged from another browser tab (Chrome): provides URL via dataTransfer
  if (files.length === 0) {
    const url = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain');
    if (url && /^https?:/.test(url.trim())) {
      await uploadUrl(cls, url.trim());
    } else {
      logLine('drop: no image found in payload', 'err');
    }
  }
}

async function refreshCounts() {
  try {
    const r = await fetch('/api/counts');
    const j = await r.json();
    for (const [cls, n] of Object.entries(j)) {
      const el = document.getElementById('count-' + cls);
      if (el) el.textContent = n;
    }
  } catch (e) { /* ignore */ }
}

refreshCounts();
setInterval(refreshCounts, 5000);
</script>
</body>
</html>
""".replace("{{CLASSES_JSON}}", str(CLASSES).replace("'", '"'))


def main():
    ap = argparse.ArgumentParser()
    # Default lives at <repo>/training/data/labeled/embedder regardless of where
    # the script is invoked from.
    default_out = Path(__file__).resolve().parent.parent / "data" / "labeled" / "embedder"
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
    )
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
