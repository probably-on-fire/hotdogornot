"""Generate the app icon PNGs from the aired.com brand mark.

Source: `assets/icon/source/aired_logo_full.png` — the full aired.com
logo (red circuit-trace brain on white, "AI RED" wordmark below).

Outputs (1024x1024 each):
  flutter/assets/icon/icon.png             — brain mark (no wordmark)
                                             centered on white, with
                                             subtle padding so iOS
                                             corner-rounding doesn't
                                             clip the silhouette.
  flutter/assets/icon/icon_foreground.png  — brain mark only, on
                                             transparent canvas, sized
                                             to fit Android adaptive
                                             icon's ~66% safe zone.

The wordmark is cropped out because it would be unreadable at
home-screen sizes (e.g. 60-120px). The brain alone is the recognizable
aired.com mark.

Re-run after replacing the source logo:
  python flutter/tool/generate_icon.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

OUT_SIZE = 1024
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "icon"
SOURCE = ASSETS / "source" / "aired_logo_full.png"

# White matches the aired.com homepage background — the logo was
# designed to sit on white. Android adaptive icons get the same
# background color via flutter_launcher_icons config.
BG_WHITE = (255, 255, 255, 255)
TRANSPARENT = (0, 0, 0, 0)

# Padding inside the canvas so iOS's corner mask + Android adaptive
# safe zone don't clip the brain. Tuned visually against the source.
FULL_ICON_BRAIN_FRACTION = 0.78          # brain occupies 78% of canvas
FOREGROUND_BRAIN_FRACTION = 0.62         # smaller for adaptive safe zone


def _opaque_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    """Tightest bounding box around non-transparent / non-white pixels.
    The source logo is white background, so we scan for any pixel that
    isn't pure white *or* has alpha < 255 — that's the brain + wordmark."""
    rgba = img.convert("RGBA")
    w, h = rgba.size
    px = rgba.load()
    min_x, min_y, max_x, max_y = w, h, 0, 0
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            # "Logo pixel" = anything appreciably non-white. Threshold
            # generous enough to ignore JPEG/WebP compression specks
            # in the white background.
            if a > 0 and (r < 240 or g < 240 or b < 240):
                if x < min_x: min_x = x
                if y < min_y: min_y = y
                if x > max_x: max_x = x
                if y > max_y: max_y = y
    return (min_x, min_y, max_x + 1, max_y + 1)


def _crop_brain_only(source: Image.Image) -> Image.Image:
    """Crop the full logo to just the brain mark, dropping the AI RED
    wordmark below it.

    Heuristic: the brain and the wordmark are separated by a band of
    pure white. Walk the rows from top to bottom, count "logo pixels"
    per row (anything not white-ish), then find the longest run of
    near-empty rows in the bottom half of the bbox — that's the gap.
    Crop just above the start of that gap so we keep the entire brain
    (including the two prominent nodes at the bottom of the brain
    stem) and lose the wordmark.
    """
    bbox = _opaque_bbox(source)
    x0, y0, x1, y1 = bbox
    rgba = source.convert("RGBA")
    px = rgba.load()

    # Count logo pixels per row inside the horizontal extent of the bbox.
    row_density = []
    for y in range(y0, y1):
        n = 0
        for x in range(x0, x1):
            r, g, b, a = px[x, y]
            if a > 0 and (r < 240 or g < 240 or b < 240):
                n += 1
        row_density.append(n)

    # Search for the gap only in the bottom half of the bbox — the top
    # half is all brain. "Empty" = fewer than 0.5% of the row's pixels
    # are logo (kills JPEG/WebP compression noise without missing real
    # gaps).
    h = y1 - y0
    width_px = x1 - x0
    empty_threshold = max(2, int(width_px * 0.005))
    search_start = h // 2

    best_gap_start = -1
    best_gap_len = 0
    cur_start = -1
    cur_len = 0
    for i in range(search_start, h):
        if row_density[i] <= empty_threshold:
            if cur_start < 0:
                cur_start = i
                cur_len = 1
            else:
                cur_len += 1
            if cur_len > best_gap_len:
                best_gap_len = cur_len
                best_gap_start = cur_start
        else:
            cur_start = -1
            cur_len = 0

    if best_gap_start < 0 or best_gap_len < 4:
        # No clear gap found — assume there's no wordmark to drop and
        # return the full bbox.
        return source.crop((x0, y0, x1, y1))

    # Keep everything above the gap. The gap starts a few px below the
    # last brain pixel; that natural breathing room is fine to keep.
    crop_bottom = y0 + best_gap_start
    return source.crop((x0, y0, x1, crop_bottom))


def _composite_centered(brain: Image.Image, fraction: float,
                        bg: tuple[int, int, int, int]) -> Image.Image:
    """Drop the brain mark onto a 1024-square canvas, scaled so its
    longest dimension is `fraction * OUT_SIZE`."""
    canvas = Image.new("RGBA", (OUT_SIZE, OUT_SIZE), bg)
    bw, bh = brain.size
    target_long = int(OUT_SIZE * fraction)
    scale = target_long / max(bw, bh)
    nw, nh = int(bw * scale), int(bh * scale)
    resized = brain.resize((nw, nh), Image.LANCZOS)
    x = (OUT_SIZE - nw) // 2
    y = (OUT_SIZE - nh) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(
            f"Missing source logo at {SOURCE}. Drop the full aired.com "
            f"logo PNG there (with the AI RED wordmark — the script "
            f"crops it off automatically) and re-run."
        )
    source = Image.open(SOURCE).convert("RGBA")
    brain = _crop_brain_only(source)

    full = _composite_centered(brain, FULL_ICON_BRAIN_FRACTION, BG_WHITE)
    full.save(ASSETS / "icon.png", format="PNG", optimize=True)
    print(f"wrote {ASSETS / 'icon.png'}")

    foreground = _composite_centered(brain, FOREGROUND_BRAIN_FRACTION, TRANSPARENT)
    foreground.save(ASSETS / "icon_foreground.png", format="PNG", optimize=True)
    print(f"wrote {ASSETS / 'icon_foreground.png'}")


if __name__ == "__main__":
    main()
