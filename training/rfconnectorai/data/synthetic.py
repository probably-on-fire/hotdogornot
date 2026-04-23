from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image, ImageDraw


# Nominal outer-body diameters (mm). Families visually similar but not identical.
FAMILY_OUTER_DIAMETER_MM = {
    "sma": 6.3,
    "precision": 5.5,
}

MALE_PIN_EXTENDS_MM = 2.5
FEMALE_PIN_RECESS_MM = 1.5

# Nominal object depth and its noise for the generated depth map.
OBJECT_DEPTH_M = 0.12
BACKGROUND_DEPTH_M = 2.0
DEPTH_NOISE_STD_M = 0.002


def make_connector_mesh(gender: str, family: str) -> trimesh.Trimesh:
    """
    Build a simple connector-like mesh:
      - cylinder for the body
      - smaller cylinder as the inner pin (extends for male, capped for female)

    This mesh is not currently rendered via GL on Windows (no reliable offscreen
    context), but the trimesh object remains useful for downstream geometric
    reasoning and as a stable reference for the 2D procedural renderer below.
    """
    body_radius_mm = FAMILY_OUTER_DIAMETER_MM[family] / 2
    body_length_mm = 10.0
    pin_radius_mm = body_radius_mm * 0.25 if family == "precision" else body_radius_mm * 0.28

    body = trimesh.creation.cylinder(radius=body_radius_mm, height=body_length_mm, sections=48)

    if gender == "male":
        pin = trimesh.creation.cylinder(
            radius=pin_radius_mm, height=MALE_PIN_EXTENDS_MM, sections=24
        )
        pin.apply_translation([0, 0, body_length_mm / 2 + MALE_PIN_EXTENDS_MM / 2])
        mesh = trimesh.util.concatenate([body, pin])
    else:
        cap = trimesh.creation.cylinder(
            radius=pin_radius_mm * 1.8, height=0.2, sections=24
        )
        cap.apply_translation([0, 0, body_length_mm / 2])
        mesh = trimesh.util.concatenate([body, cap])

    # Convert mm → meters.
    mesh.apply_scale(0.001)
    return mesh


def render_connector_sample(
    gender: str,
    family: str,
    image_size: int = 384,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce a procedural Phase-0 proxy image of a connector.

    Returns (RGB uint8, depth float32 m, mask bool), all of shape
    (image_size, image_size) or (image_size, image_size, 3).

    Instead of rasterizing a 3D mesh (which needs a GL context that is not
    reliably available on headless Windows), this renders a connector-like
    appearance procedurally with PIL:
      - body: filled circle with silver/gold-ish base color
      - pin:  smaller concentric circle, darker for female, bright for male
      - background: solid noise-ish fill
      - depth: foreground = OBJECT_DEPTH_M, background = BACKGROUND_DEPTH_M,
               with Gaussian noise
      - mask:  True where the connector was drawn

    Deterministic for a given `seed`.
    """
    rng = np.random.default_rng(seed)

    # Body radius in pixels, scaled by the family diameter.
    body_outer_mm = FAMILY_OUTER_DIAMETER_MM[family]
    body_radius_px = int(image_size * 0.18 * (body_outer_mm / 6.3))
    pin_radius_px = int(
        body_radius_px * (0.25 if family == "precision" else 0.28)
    )
    if gender == "female":
        pin_radius_px = int(pin_radius_px * 1.8)

    # Slight jitter of center and radii so repeated samples differ.
    cx = image_size // 2 + int(rng.uniform(-image_size * 0.05, image_size * 0.05))
    cy = image_size // 2 + int(rng.uniform(-image_size * 0.05, image_size * 0.05))
    body_radius_px = max(8, body_radius_px + int(rng.uniform(-4, 4)))
    pin_radius_px = max(2, pin_radius_px + int(rng.uniform(-2, 2)))

    # Colors.
    bg_gray = int(rng.integers(20, 80))
    base_color = (
        int(np.clip(150 + rng.integers(-30, 30), 0, 255)),
        int(np.clip(150 + rng.integers(-30, 30), 0, 255)),
        int(np.clip(140 + rng.integers(-40, 20), 0, 255)),
    )
    pin_color = (
        (240, 220, 170) if gender == "male" else (60, 60, 60)
    )

    # --- RGB ---
    rgb_img = Image.new("RGB", (image_size, image_size), (bg_gray, bg_gray, bg_gray))
    draw = ImageDraw.Draw(rgb_img)
    draw.ellipse(
        [cx - body_radius_px, cy - body_radius_px, cx + body_radius_px, cy + body_radius_px],
        fill=base_color,
    )
    draw.ellipse(
        [cx - pin_radius_px, cy - pin_radius_px, cx + pin_radius_px, cy + pin_radius_px],
        fill=pin_color,
    )
    rgb = np.asarray(rgb_img, dtype=np.uint8)

    # Add background noise texture so the model cannot learn a constant-background shortcut.
    noise = rng.integers(-15, 15, size=rgb.shape, dtype=np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # --- Mask (foreground pixels) ---
    yy, xx = np.ogrid[:image_size, :image_size]
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= body_radius_px ** 2

    # --- Depth ---
    depth = np.full((image_size, image_size), BACKGROUND_DEPTH_M, dtype=np.float32)
    depth[mask] = OBJECT_DEPTH_M
    depth += rng.normal(0.0, DEPTH_NOISE_STD_M, size=depth.shape).astype(np.float32)
    depth = np.maximum(depth, 0.01)

    return rgb, depth, mask
