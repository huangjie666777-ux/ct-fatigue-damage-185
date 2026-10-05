"""PNG preview rendering. Preview-only scaling never alters the NPY data."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def render_png(image: np.ndarray) -> bytes:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        lo, hi = 0.0, 0.0
    else:
        lo, hi = float(finite.min()), float(finite.max())
    if hi > lo:
        scaled = (image - lo) / (hi - lo)
    else:
        scaled = np.zeros_like(image)
    pixels = np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def npy_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(image, dtype=np.float64), allow_pickle=False)
    return buffer.getvalue()


def render_check_png(ratio_map: np.ndarray) -> bytes:
    """RGB utilization preview: grayscale up to ratio 1, red where exceeded.

    Pixels outside the section (NaN) render black. Display-only; the
    quantitative data lives in the NPY stress maps and the JSON report.
    """
    ratio = np.nan_to_num(ratio_map, nan=0.0)
    gray = np.clip(np.rint(np.clip(ratio, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    exceeded = ratio_map > 1.0
    rgb[exceeded] = np.array([255, 0, 0], dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def render_damage_png(damage_map: np.ndarray) -> bytes:
    """RGB Miner-damage preview on a log10 scale.

    Damage values <= 0 render black; positive damage up to 1 (failure in a
    single block) spans grayscale; pixels at or beyond D=1 are marked red.
    NaN pixels (material absent) render black. Display-only.
    """
    rgb = np.zeros((*damage_map.shape, 3), dtype=np.uint8)
    positive = np.isfinite(damage_map) & (damage_map > 0.0)
    if positive.any():
        scaled = np.clip(np.log10(np.where(positive, damage_map, 1.0)), -6.0, 0.0)
        gray = np.clip(np.rint((scaled + 6.0) / 6.0 * 255.0), 0, 255).astype(np.uint8)
        rgb[..., 0] = np.where(positive, gray, 0)
        rgb[..., 1] = np.where(positive, gray, 0)
        rgb[..., 2] = np.where(positive, gray, 0)
    failed = np.isfinite(damage_map) & (damage_map >= 1.0)
    rgb[failed] = np.array([255, 0, 0], dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()
