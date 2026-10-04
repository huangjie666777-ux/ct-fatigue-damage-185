"""Analytical parallel-beam projection of an off-center homogeneous disk.

Used both as an end-to-end example and to validate reconstruction accuracy.
A disk of radius R centered at (cx, cy) with attenuation mu projects to::

    p(theta, t) = 2 * mu * sqrt(R^2 - (t - p0)^2),  |t - p0| <= R

where ``p0 = cx*cos(theta) + cy*sin(theta)``.
"""

from __future__ import annotations

import numpy as np


def disk_sinogram(
    n_angles: int,
    n_det: int,
    detector_spacing: float,
    center_index: float,
    disk_center_mm: tuple[float, float],
    radius_mm: float,
    attenuation: float,
) -> np.ndarray:
    angles = np.arange(n_angles, dtype=np.float64) * (np.pi / n_angles)
    detector_t = (np.arange(n_det) - center_index) * detector_spacing
    cx, cy = disk_center_mm
    p0 = cx * np.cos(angles) + cy * np.sin(angles)
    offset = detector_t[np.newaxis, :] - p0[:, np.newaxis]
    inside = offset * offset <= radius_mm * radius_mm
    chord = 2.0 * attenuation * np.sqrt(np.maximum(radius_mm * radius_mm - offset * offset, 0.0))
    return np.where(inside, chord, 0.0)


def acquire(
    line_integrals: np.ndarray,
    dark_level: float = 500.0,
    flat_level: float = 4000.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn line integrals into realistic (dark, flat, intensity) arrays."""
    n_det = line_integrals.shape[1]
    dark = np.full(n_det, dark_level, dtype=np.float64)
    flat = np.full(n_det, flat_level, dtype=np.float64)
    intensity = dark + (flat - dark) * np.exp(-line_integrals)
    if rng is not None:
        intensity = np.maximum(intensity + rng.normal(0.0, 0.5, intensity.shape), dark + 1.0)
    return intensity, dark, flat
