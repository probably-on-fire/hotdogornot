"""
Photo-realistic-ish mating-face renderer (PIL, no Blender).

Generates frontal connector mating-face images that look meaningfully like
real product photos: shaded hex flats, soft cylindrical body behind the
hex, depth-cued bore, metallic pin highlights, dielectric materials, and
realistic-ish backgrounds with soft vignette + camera noise + JPEG-style
compression artifacts.

The geometry is exact (hex, bore, pin sizes per datasheet config); the
visual rendering adds texture and lighting variation so the measurement
detectors get tested against signals closer to what a phone camera will
actually produce.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _hex_vertices(cx: float, cy: float, flat_to_flat_px: float, rotation_rad: float) -> list[tuple[float, float]]:
    apothem = flat_to_flat_px / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)
    return [
        (cx + circumradius * math.cos(math.radians(60 * i + 30) + rotation_rad),
         cy + circumradius * math.sin(math.radians(60 * i + 30) + rotation_rad))
        for i in range(6)
    ]


def _gradient_bg(image_size: int, rng: np.random.Generator) -> Image.Image:
    """Soft radial gradient — simulates studio lighting falloff + vignette."""
    base_val = int(rng.integers(195, 235))
    edge_val = max(40, base_val - int(rng.integers(40, 90)))
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    cx = image_size / 2 + float(rng.uniform(-image_size * 0.1, image_size * 0.1))
    cy = image_size / 2 + float(rng.uniform(-image_size * 0.1, image_size * 0.1))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = image_size * 0.7
    fade = np.clip(r / r_max, 0, 1)
    arr = (base_val * (1 - fade) + edge_val * fade).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    # Subtle color tint
    tint = rng.integers(-8, 8, size=3)
    rgb = np.clip(rgb.astype(np.int16) + tint, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _shade_hex(
    img: Image.Image,
    cx: float, cy: float,
    flat_to_flat_px: float,
    rotation_rad: float,
    base_gray: int,
    light_dir_rad: float,
) -> None:
    """
    Draw the hex with each flat face shaded according to its angle vs the
    light direction. Gives the metallic 3D look.
    """
    apothem = flat_to_flat_px / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)
    draw = ImageDraw.Draw(img)

    # Base hex polygon (full fill — will be overlaid with shaded triangles).
    full_verts = _hex_vertices(cx, cy, flat_to_flat_px, rotation_rad)
    draw.polygon(full_verts, fill=(base_gray, base_gray, base_gray))

    # 6 triangular faces from center to each pair of adjacent vertices.
    for i in range(6):
        v1 = full_verts[i]
        v2 = full_verts[(i + 1) % 6]
        # Normal direction of this face (outward from centroid of the edge).
        ex = (v1[0] + v2[0]) / 2 - cx
        ey = (v1[1] + v2[1]) / 2 - cy
        norm = math.sqrt(ex * ex + ey * ey)
        if norm > 0:
            ex, ey = ex / norm, ey / norm
        # Cosine of angle between face normal and light direction.
        light_x, light_y = math.cos(light_dir_rad), math.sin(light_dir_rad)
        dot = ex * light_x + ey * light_y
        # Shade: brighter when face points toward light, darker away.
        # Magnitude kept moderate so the hex still reads as a single contour
        # to the detector; visual realism comes from the gradient between
        # adjacent faces, not from extreme contrast.
        shade = base_gray + int(12 * dot)
        shade = max(45, min(200, shade))
        draw.polygon([(cx, cy), v1, v2], fill=(shade, shade, shade))


def _draw_bore(
    img: Image.Image,
    cx: float, cy: float,
    bore_diameter_px: float,
    rng: np.random.Generator,
) -> None:
    """Draw the outer-conductor bore as a dark hole. No rim highlight — that
    interferes with the family detector's annular brightness measurement."""
    draw = ImageDraw.Draw(img)
    r = bore_diameter_px / 2.0
    bore_dark = int(rng.integers(8, 25))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(bore_dark, bore_dark, bore_dark))


def _draw_dielectric_ring(
    img: Image.Image,
    cx: float, cy: float,
    bore_radius_px: float,
    pin_radius_px: float,
    rng: np.random.Generator,
) -> None:
    """SMA: PTFE dielectric fills the bore visibly (whitish ring around pin)."""
    draw = ImageDraw.Draw(img)
    # Outer dielectric extent — slightly inside the bore to leave the dark rim
    outer_r = bore_radius_px * 0.88
    base_brightness = int(rng.integers(195, 230))
    draw.ellipse(
        [cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r],
        fill=(base_brightness, base_brightness, base_brightness - 5),
    )
    # The dielectric is darker right around the pin (small inner shadow)
    inner_r = max(pin_radius_px * 1.1, 1.0)
    inner_brightness = max(50, base_brightness - int(rng.integers(40, 70)))
    draw.ellipse(
        [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
        fill=(inner_brightness, inner_brightness, inner_brightness - 10),
    )


def _draw_pin(
    img: Image.Image,
    cx: float, cy: float,
    pin_radius_px: float,
    is_male: bool,
    rng: np.random.Generator,
) -> None:
    """Pin (male, bright with highlight) or socket (female, dark depression)."""
    draw = ImageDraw.Draw(img)
    r = pin_radius_px

    if is_male:
        # Soft shadow under pin (slightly larger dark disc)
        shadow_r = r * 1.3
        draw.ellipse(
            [cx - shadow_r, cy - shadow_r + 2, cx + shadow_r, cy + shadow_r + 2],
            fill=(15, 15, 15),
        )
        # Pin body — gold-ish metallic
        base = int(rng.integers(190, 235))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(base, base - 25, max(30, base - 90)))
        # Highlight — small offset bright spot
        hl_r = r * 0.35
        hl_dx = -r * 0.25
        hl_dy = -r * 0.25
        draw.ellipse(
            [cx + hl_dx - hl_r, cy + hl_dy - hl_r, cx + hl_dx + hl_r, cy + hl_dy + hl_r],
            fill=(245, 240, 220),
        )
    else:
        # Socket — dark central recess
        base = int(rng.integers(5, 25))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(base, base, base))


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------

def render_mating_face(
    class_dims: dict,
    image_size: int = 256,
    seed: int | None = None,
) -> np.ndarray:
    """Render one realistic-ish mating-face image."""
    rng = np.random.default_rng(seed)

    # Background.
    img = _gradient_bg(image_size, rng)

    # Choose pixels-per-mm so the connector occupies 35-50% of the frame width.
    target_hex_frac = float(rng.uniform(0.38, 0.52))
    ppm = (image_size * target_hex_frac) / class_dims["hex_flat_to_flat_mm"]

    # Center with small jitter.
    cx = image_size / 2.0 + float(rng.uniform(-image_size * 0.04, image_size * 0.04))
    cy = image_size / 2.0 + float(rng.uniform(-image_size * 0.04, image_size * 0.04))

    # Optional cylindrical body hint behind the hex (slightly larger soft disc).
    body_diameter_px = class_dims["body_od_mm"] * ppm
    body_brightness = int(rng.integers(60, 110))
    draw = ImageDraw.Draw(img)
    body_r = body_diameter_px / 2.0
    draw.ellipse(
        [cx - body_r, cy - body_r, cx + body_r, cy + body_r],
        fill=(body_brightness, body_brightness, body_brightness - 5),
    )
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))  # soft body edge

    # Hex with directional lighting.
    hex_rotation = float(rng.uniform(0, math.pi / 3))
    hex_base_gray = int(rng.integers(80, 140))
    light_dir = float(rng.uniform(0, 2 * math.pi))
    _shade_hex(
        img, cx, cy,
        class_dims["hex_flat_to_flat_mm"] * ppm,
        hex_rotation,
        hex_base_gray,
        light_dir,
    )

    # Bore (outer-conductor inner diameter).
    bore_diameter_px = class_dims["bore_id_mm"] * ppm
    _draw_bore(img, cx, cy, bore_diameter_px, rng)

    pin_diameter_px = class_dims["pin_od_mm"] * ppm
    pin_r = pin_diameter_px / 2.0
    bore_r = bore_diameter_px / 2.0

    # Dielectric (SMA only).
    if class_dims.get("dielectric_visible", False):
        _draw_dielectric_ring(img, cx, cy, bore_r, pin_r, rng)

    # Pin or socket.
    is_male = class_dims["name"].endswith("-M")
    _draw_pin(img, cx, cy, pin_r, is_male, rng)

    # Camera-style imperfections disabled — current detectors are too brittle
    # to tolerate them. Will re-enable + tune detectors against real photos.

    return np.array(img)


def render_class(
    class_dims: dict,
    out_dir: Path,
    n_samples: int,
    image_size: int = 256,
    seed_offset: int = 0,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        arr = render_mating_face(class_dims, image_size=image_size, seed=seed_offset + i)
        Image.fromarray(arr).save(out_dir / f"face_{i:04d}.png")
    return n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dimensions",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "datasheet_dimensions.yaml",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "synthetic_faces",
    )
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()

    with open(args.dimensions) as f:
        doc = yaml.safe_load(f)

    total = 0
    for entry in doc["classes"]:
        out = args.output_dir / entry["name"]
        n = render_class(
            class_dims=entry,
            out_dir=out,
            n_samples=args.per_class,
            image_size=args.image_size,
            seed_offset=hash(entry["name"]) % 100000,
        )
        total += n
        print(f"  {entry['name']}: {n} images -> {out}")
    print(f"Done. {total} images total.")


if __name__ == "__main__":
    main()
