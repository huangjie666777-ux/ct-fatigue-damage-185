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
