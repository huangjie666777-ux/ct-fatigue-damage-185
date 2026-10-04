"""Rectangular region-of-interest validation and quantitative integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_utils import ValidationError

MAX_ROIS = 8


@dataclass(frozen=True)
class Roi:
    """Pixel-index rectangle: left/top inclusive, right/bottom exclusive."""

    name: str
    x0: int
    y0: int
    x1: int
    y1: int

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        mask[self.y0 : self.y1, self.x0 : self.x1] = True
        return mask


def validate_rois(spec: object, output_size: int) -> list[Roi]:
    """Validate 1-8 uniquely named, in-bounds, non-empty ROI rectangles."""
    if not isinstance(spec, (list, tuple)) or not (1 <= len(spec) <= MAX_ROIS):
        raise ValidationError(f"rois must be a list of 1 to {MAX_ROIS} rectangles")
    rois: list[Roi] = []
    names: set[str] = set()
    for entry in spec:
        if not isinstance(entry, dict):
            raise ValidationError("each ROI must be an object with name, x0, y0, x1, y1")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("each ROI needs a non-empty name")
        name = name.strip()
        if name in names:
            raise ValidationError(f"duplicate ROI name {name!r}")
        names.add(name)
        try:
            x0, y0, x1, y1 = (entry[k] for k in ("x0", "y0", "x1", "y1"))
        except KeyError as exc:
            raise ValidationError(f"ROI {name!r} is missing key {exc}") from exc
        coords = (x0, y0, x1, y1)
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in coords):
            raise ValidationError(f"ROI {name!r} coordinates must be integers")
        if not (0 <= x0 < x1 <= output_size and 0 <= y0 < y1 <= output_size):
            raise ValidationError(
                f"ROI {name!r} ({x0},{y0},{x1},{y1}) is empty or out of bounds "
                f"for a {output_size}x{output_size} image"
            )
        rois.append(Roi(name=name, x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1)))
    return rois


def validate_slice_thickness(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("slice_thickness_mm must be a number")
    thickness = float(value)
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValidationError("slice_thickness_mm must be finite and strictly positive")
    return thickness


def integrate_rois(
    rois: list[Roi],
    densities: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
    pixel_spacing_mm: float,
    slice_thickness_mm: float,
) -> list[dict]:
    """Integrate material masses (mg) and mean residuals (mm^-1) per ROI.

    Mass = sum(density) * pixel_area_mm^2 * slice_thickness_mm, with density
    in mg/mm^3. Negative density pixels contribute as-is; nothing is clipped.
    """
    voxel_volume = pixel_spacing_mm * pixel_spacing_mm * slice_thickness_mm
    results = []
    for roi in rois:
        mask = roi.mask(next(iter(densities.values())).shape)
        masses = {
            name: float(np.sum(density[mask]) * voxel_volume)
            for name, density in densities.items()
        }
        mean_residuals = {
            energy: float(np.mean(residual[mask]))
            for energy, residual in residuals.items()
        }
        results.append(
            {
                "name": roi.name,
                "pixels": {"x0": roi.x0, "y0": roi.y0, "x1": roi.x1, "y1": roi.y1},
                "n_pixels": int(mask.sum()),
                "mass_mg": masses,
                "mean_residual_per_mm": mean_residuals,
            }
        )
    return results
