"""
Procedural mating-face image renderer (PIL-based, no Blender needed).

Produces a frontal mating-face view of a connector with the right hex,
aperture, dielectric, and pin geometry per class. Used to generate eval
images for the measurement pipeline before real connector photos arrive.

The output looks like a stylized phone-camera capture: gray hex on a light
background, dark bore, small bright pin (male) or dark socket (female),
brightness-modulated dielectric region for SMA. Adds rotation, position
jitter, brightness jitter, and noise so the eval set isn't trivial.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter


def _hex_vertices(cx: float, cy: float, flat_to_flat_px: float, rotation_rad: float) -> list[tuple[float, float]]:
    apothem = flat_to_flat_px / 2.0
    circumradius = apothem * 2.0 / math.sqrt(3)
    return [
        (cx + circumradius * math.cos(math.radians(60 * i + 30) + rotation_rad),
         cy + circumradius * math.sin(math.radians(60 * i + 30) + rotation_rad))
        for i in range(6)
    ]


def render_mating_face(
    class_dims: dict,
    image_size: int = 256,
    seed: int | None = None,
) -> np.ndarray:
    """
    Render a single mating-face image for the given class.

    `class_dims` is one entry from datasheet_dimensions.yaml — must include:
      bore_id_mm, pin_od_mm, hex_flat_to_flat_mm, body_od_mm, dielectric_visible
    plus a "name" key (used to determine gender from the suffix).
    """
    rng = np.random.default_rng(seed)

    # Choose pixels-per-mm so the connector occupies a reasonable fraction
    # of the frame. Hex flat-to-flat should be roughly 40-55% of image width.
    target_hex_frac = float(rng.uniform(0.42, 0.55))
    ppm = (image_size * target_hex_frac) / class_dims["hex_flat_to_flat_mm"]

    # Center with small jitter.
    cx = image_size / 2.0 + float(rng.uniform(-image_size * 0.04, image_size * 0.04))
    cy = image_size / 2.0 + float(rng.uniform(-image_size * 0.04, image_size * 0.04))

    # Background brightness (light, with jitter).
    bg_val = int(rng.integers(210, 245))
    img = Image.new("RGB", (image_size, image_size), (bg_val, bg_val, bg_val))
    draw = ImageDraw.Draw(img)

    # Hex coupling nut — gray tone with jitter.
    hex_gray = int(rng.integers(70, 130))
    hex_rotation = float(rng.uniform(0, math.pi / 3))  # any orientation 0..60°
    hex_verts = _hex_vertices(cx, cy, class_dims["hex_flat_to_flat_mm"] * ppm, hex_rotation)
    draw.polygon(hex_verts, fill=(hex_gray, hex_gray, hex_gray))

    # Aperture (outer-conductor bore) — dark by default.
    bore_diameter_px = class_dims["bore_id_mm"] * ppm
    r_bore = bore_diameter_px / 2.0
    draw.ellipse(
        [cx - r_bore, cy - r_bore, cx + r_bore, cy + r_bore],
        fill=(20, 20, 20),
    )

    # Dielectric: SMA has visible PTFE filling the bore (whitish ring), making
    # the annular region between bore edge and pin/socket bright instead of dark.
    pin_diameter_px = class_dims["pin_od_mm"] * ppm
    if class_dims.get("dielectric_visible", False):
        # Draw an inner white-ish ring covering the annular region.
        ptfe_outer_r = r_bore * 0.92
        ptfe_brightness = int(rng.integers(195, 235))
        draw.ellipse(
            [cx - ptfe_outer_r, cy - ptfe_outer_r, cx + ptfe_outer_r, cy + ptfe_outer_r],
            fill=(ptfe_brightness, ptfe_brightness, ptfe_brightness - 5),
        )

    # Inner conductor: pin (male, bright) or socket (female, dark).
    is_male = class_dims["name"].endswith("-M")
    r_pin = pin_diameter_px / 2.0
    if is_male:
        # Bright metallic pin
        pin_brightness = int(rng.integers(200, 245))
        draw.ellipse(
            [cx - r_pin, cy - r_pin, cx + r_pin, cy + r_pin],
            fill=(pin_brightness, pin_brightness - 20, pin_brightness - 60),
        )
    else:
        # Dark socket (recessed pin receptacle)
        socket_brightness = int(rng.integers(5, 30))
        draw.ellipse(
            [cx - r_pin, cy - r_pin, cx + r_pin, cy + r_pin],
            fill=(socket_brightness, socket_brightness, socket_brightness),
        )

    # Convert to numpy and add noise + slight blur.
    arr = np.array(img).astype(np.int16)
    noise_std = int(rng.integers(2, 8))
    noise = rng.normal(0, noise_std, size=arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    if rng.uniform() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 0.8))))
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
