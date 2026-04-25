"""
Photo-style mating-face renderer (PIL + numpy, no Blender).

Uses smooth radial-gradient shading, computed specular highlights, depth-cued
bore, metallic pin with proper highlight, procedural lab-bench background,
and a slight perspective tilt to produce images that look like phone-camera
captures of an RF connector face.

Geometry is exact (hex / aperture / pin sizes per datasheet config); the
visual treatment adds the texture, lighting, and depth cues a real photo
would have. Detectors may need re-tuning against real photos regardless,
but these images are at least a defensible visual stand-in for demo
material and richer eval than flat polygons.
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
# Material colors — chosen to match real connector finishes, not arbitrary grays.
# ---------------------------------------------------------------------------
MATERIAL_COLORS = {
    # Stainless on real RF connectors typically photographs as medium-dark gray
    # under lab lighting; keeping it darker than common bench backgrounds
    # ensures Otsu can cleanly separate the hex from the background.
    "stainless": (110, 113, 120),
    "brass":     (170, 145, 95),       # for SMA bodies (slightly darker than before)
    "gold_pin":  (240, 195, 110),
    "ptfe":      (235, 230, 215),      # cream-white
    "dark_bore": (12, 14, 16),
    "socket":    (14, 16, 18),
}


def _hex_mask(image_size: int, cx: float, cy: float, flat_to_flat_px: float, rotation_rad: float) -> np.ndarray:
    """Return a (H, W) bool mask of the hex region."""
    apothem = flat_to_flat_px / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)
    img = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(img)
    verts = [
        (cx + circumradius * math.cos(math.radians(60 * i + 30) + rotation_rad),
         cy + circumradius * math.sin(math.radians(60 * i + 30) + rotation_rad))
        for i in range(6)
    ]
    draw.polygon(verts, fill=255)
    return np.array(img) > 128


def _circle_mask(image_size: int, cx: float, cy: float, radius_px: float) -> np.ndarray:
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2


def _make_background(image_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Procedural lab-bench background: smooth gradient + low-frequency texture
    + subtle vignette. Gives the synthetic image a real-photo backdrop.
    """
    # Base gradient in HSV — slight tint variation
    base_brightness = rng.integers(195, 235)
    edge_brightness = max(60, base_brightness - rng.integers(50, 100))

    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    cx = image_size / 2 + float(rng.uniform(-image_size * 0.1, image_size * 0.1))
    cy = image_size / 2 + float(rng.uniform(-image_size * 0.1, image_size * 0.1))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = image_size * 0.7
    fade = np.clip(r / r_max, 0, 1) ** 1.4

    arr = (base_brightness * (1 - fade) + edge_brightness * fade).astype(np.float32)

    # Low-frequency Perlin-ish texture (sum of a few sines).
    tex = np.zeros_like(arr)
    for _ in range(3):
        kx = float(rng.uniform(0.005, 0.025))
        ky = float(rng.uniform(0.005, 0.025))
        phase_x = float(rng.uniform(0, 6.28))
        phase_y = float(rng.uniform(0, 6.28))
        amp = float(rng.uniform(2, 8))
        tex += amp * np.sin(kx * xx + phase_x) * np.cos(ky * yy + phase_y)

    arr = np.clip(arr + tex, 0, 255).astype(np.uint8)

    # Slight color tint (warm or cool)
    tint_r, tint_g, tint_b = rng.integers(-12, 12, size=3)
    rgb = np.stack([
        np.clip(arr.astype(np.int16) + tint_r, 0, 255),
        np.clip(arr.astype(np.int16) + tint_g, 0, 255),
        np.clip(arr.astype(np.int16) + tint_b, 0, 255),
    ], axis=-1).astype(np.uint8)
    return rgb


def _shade_hex(
    canvas: np.ndarray,
    cx: float, cy: float,
    flat_to_flat_px: float,
    rotation_rad: float,
    base_color: tuple[int, int, int],
    light_dir_rad: float,
    rng: np.random.Generator,
) -> None:
    """
    Composite a smoothly-shaded hex onto canvas in-place.

    Uses a directional-gradient shading: brightness varies smoothly across
    the hex face based on dot product with light direction. No per-face
    stepping — keeps the hex readable as a single contour for the detector.
    """
    image_size = canvas.shape[0]
    mask = _hex_mask(image_size, cx, cy, flat_to_flat_px, rotation_rad)
    if not mask.any():
        return

    # Compute a smooth shading factor across the hex region.
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    apothem = flat_to_flat_px / 2.0
    light_x = math.cos(light_dir_rad)
    light_y = math.sin(light_dir_rad)
    # Project (x - cx, y - cy) onto light direction; normalize by apothem.
    proj = ((xx - cx) * light_x + (yy - cy) * light_y) / apothem
    proj = np.clip(proj, -1.2, 1.2)
    # Shading: brighter where proj>0 (lit side), darker where proj<0.
    shade = 1.0 + 0.18 * proj  # ~0.78 to 1.22 range

    # Apply to base color.
    base = np.array(base_color, dtype=np.float32)
    shaded = (base[None, None, :] * shade[..., None]).clip(0, 255).astype(np.uint8)

    # Composite onto canvas under hex mask.
    canvas[mask] = shaded[mask]

    # Specular highlight: subtle broad sheen, not a punch-through bright spot.
    # Keep peak alpha low so Otsu thresholding still groups the highlight with
    # the rest of the hex face (a too-bright spec creates a "hole" in the
    # binary mask that shrinks the detected contour).
    spec_dist = apothem * 0.45
    sx = cx + light_x * spec_dist
    sy = cy + light_y * spec_dist
    spec_radius = apothem * 0.30
    spec_falloff = ((xx - sx) ** 2 + (yy - sy) ** 2) / (spec_radius ** 2)
    spec_alpha = np.exp(-spec_falloff * 1.0) * 0.18
    spec_alpha[~mask] = 0
    sheen = np.minimum(base * 1.25, 235.0)
    blend = (canvas.astype(np.float32) * (1 - spec_alpha[..., None])
             + sheen[None, None, :] * spec_alpha[..., None])
    canvas[:] = blend.clip(0, 255).astype(np.uint8)


def _draw_bore_depth(
    canvas: np.ndarray,
    cx: float, cy: float,
    bore_radius_px: float,
    rng: np.random.Generator,
) -> None:
    """Bore as a dark hole with subtle inner gradient (depth cue)."""
    image_size = canvas.shape[0]
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = r <= bore_radius_px

    # Gradient: darkest at center, slightly lighter at rim (suggests 3D recession)
    fade = np.clip(r / bore_radius_px, 0, 1)
    inner_dark = MATERIAL_COLORS["dark_bore"]
    rim_dark = (28, 30, 34)
    base = np.array(inner_dark, dtype=np.float32)
    rim = np.array(rim_dark, dtype=np.float32)
    grad = base[None, None, :] * (1 - fade[..., None]) + rim[None, None, :] * fade[..., None]
    canvas[mask] = grad[mask].astype(np.uint8)


def _draw_dielectric(
    canvas: np.ndarray,
    cx: float, cy: float,
    bore_radius_px: float,
    pin_radius_px: float,
    rng: np.random.Generator,
) -> None:
    """SMA: PTFE dielectric ring around the pin. Subtle inner shadow at pin edge."""
    image_size = canvas.shape[0]
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    outer_r = bore_radius_px * 0.92
    inner_r = max(pin_radius_px * 1.05, 1.0)
    ptfe_mask = (r <= outer_r) & (r >= inner_r)
    if not ptfe_mask.any():
        return
    # Slight gradient: outer rim brighter, inner darker (inner shadow at pin)
    fade = np.clip((outer_r - r) / max(outer_r - inner_r, 1e-3), 0, 1)
    base = np.array(MATERIAL_COLORS["ptfe"], dtype=np.float32)
    inner = np.array((180, 175, 160), dtype=np.float32)
    grad = base[None, None, :] * fade[..., None] + inner[None, None, :] * (1 - fade[..., None])
    canvas[ptfe_mask] = grad[ptfe_mask].astype(np.uint8)


def _draw_pin(
    canvas: np.ndarray,
    cx: float, cy: float,
    pin_radius_px: float,
    is_male: bool,
    light_dir_rad: float,
    rng: np.random.Generator,
) -> None:
    """Pin (male, gold metallic with highlight) or socket (female, dark recession)."""
    image_size = canvas.shape[0]
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = r <= pin_radius_px
    if not mask.any():
        return

    if is_male:
        base = np.array(MATERIAL_COLORS["gold_pin"], dtype=np.float32)
        # Hemispherical shading on pin: bright on lit side, darker on opposite.
        light_x = math.cos(light_dir_rad)
        light_y = math.sin(light_dir_rad)
        proj = ((xx - cx) * light_x + (yy - cy) * light_y) / pin_radius_px
        proj = np.clip(proj, -1.0, 1.0)
        shade = 1.0 + 0.30 * proj
        shaded = (base[None, None, :] * shade[..., None]).clip(0, 255)
        canvas[mask] = shaded[mask].astype(np.uint8)

        # Specular highlight on pin: bright near light direction, small radius.
        spec_dx = light_x * pin_radius_px * 0.40
        spec_dy = light_y * pin_radius_px * 0.40
        spec_falloff = ((xx - cx - spec_dx) ** 2 + (yy - cy - spec_dy) ** 2) / (pin_radius_px * 0.30) ** 2
        spec_alpha = np.exp(-spec_falloff * 2.0) * 0.85
        spec_alpha[~mask] = 0
        spec_color = np.array([255, 250, 230], dtype=np.float32)
        blend = (canvas.astype(np.float32) * (1 - spec_alpha[..., None])
                 + spec_color[None, None, :] * spec_alpha[..., None])
        canvas[:] = blend.clip(0, 255).astype(np.uint8)
    else:
        # Socket: dark central recess with very slight rim highlight at top.
        socket_color = np.array(MATERIAL_COLORS["socket"], dtype=np.float32)
        canvas[mask] = socket_color.astype(np.uint8)


def _sample_tilt_strength(rng: np.random.Generator) -> float:
    """
    Mixed tilt distribution. Most renders stay mild (detector-friendly), a
    minority push into moderate/strong tilt so the dataset spans the range
    of real-world phone-capture angles. Strong tilts will fall outside the
    measurement pipeline's framing tolerance and exercise the FramingGate
    reject path.
    """
    bucket = rng.uniform()
    if bucket < 0.60:
        return float(rng.uniform(0.0, 0.015))      # mild
    if bucket < 0.90:
        return float(rng.uniform(0.015, 0.05))     # moderate
    return float(rng.uniform(0.05, 0.10))           # strong


def _apply_perspective(img: Image.Image, rng: np.random.Generator,
                       tilt_strength: float | None = None) -> Image.Image:
    """Perspective tilt — simulates phone not perfectly perpendicular."""
    w, h = img.size
    if tilt_strength is None:
        tilt_strength = _sample_tilt_strength(rng)
    max_d = int(min(w, h) * tilt_strength)
    if max_d < 1:
        return img
    dx = lambda: int(rng.integers(-max_d, max_d + 1))
    dy = lambda: int(rng.integers(-max_d, max_d + 1))
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(dx(), dy()), (w + dx(), dy()), (w + dx(), h + dy()), (dx(), h + dy())]

    # PIL uses dst → src mapping for PERSPECTIVE
    coeffs = _find_perspective_coeffs(dst, src)
    return img.transform(
        (w, h), Image.PERSPECTIVE, coeffs,
        resample=Image.BILINEAR, fillcolor=(220, 220, 220)
    )


def _find_perspective_coeffs(src, dst):
    matrix = []
    for s, d in zip(src, dst):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0]*d[0], -s[0]*d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1]*d[0], -s[1]*d[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(src, dtype=np.float64).reshape(8)
    res = np.linalg.solve(A, B)
    return res.tolist()


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------

def render_mating_face(
    class_dims: dict,
    image_size: int = 256,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Canvas starts as the procedural background.
    canvas = _make_background(image_size, rng)

    # Geometry parameters.
    target_hex_frac = float(rng.uniform(0.40, 0.52))
    ppm = (image_size * target_hex_frac) / class_dims["hex_flat_to_flat_mm"]
    cx = image_size / 2.0 + float(rng.uniform(-image_size * 0.04, image_size * 0.04))
    cy = image_size / 2.0 + float(rng.uniform(-image_size * 0.04, image_size * 0.04))
    light_dir = float(rng.uniform(0, 2 * math.pi))

    # Hex base color depends on family material.
    is_sma = class_dims.get("dielectric_visible", False)
    base_color = MATERIAL_COLORS["brass"] if is_sma else MATERIAL_COLORS["stainless"]
    # Small per-render color jitter
    base_color = tuple(int(np.clip(c + rng.integers(-12, 13), 0, 255)) for c in base_color)

    hex_rotation = float(rng.uniform(0, math.pi / 3))
    hex_ff_px = class_dims["hex_flat_to_flat_mm"] * ppm
    _shade_hex(canvas, cx, cy, hex_ff_px, hex_rotation, base_color, light_dir, rng)

    # Bore.
    bore_radius_px = class_dims["bore_id_mm"] * ppm / 2.0
    _draw_bore_depth(canvas, cx, cy, bore_radius_px, rng)

    pin_radius_px = class_dims["pin_od_mm"] * ppm / 2.0

    # Dielectric (SMA only).
    if is_sma:
        _draw_dielectric(canvas, cx, cy, bore_radius_px, pin_radius_px, rng)

    # Pin / socket.
    is_male = class_dims["name"].endswith("-M")
    _draw_pin(canvas, cx, cy, pin_radius_px, is_male, light_dir, rng)

    # PIL conversion for post-processing.
    img = Image.fromarray(canvas)

    # Slight perspective tilt 70% of the time.
    if rng.uniform() < 0.7:
        img = _apply_perspective(img, rng)

    # Subtle defocus on the very edges (depth-of-field hint).
    if rng.uniform() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.5))))

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
