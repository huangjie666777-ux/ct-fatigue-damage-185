"""End-to-end reconstruction pipeline shared by HTTP and CLI code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import calibrate
from .io_utils import LoadedData, validate_params
from .reconstruct import fbp


@dataclass(frozen=True)
class ReconstructionResult:
    image: np.ndarray
    line_integrals: np.ndarray
    params: dict
    stats: dict


def reconstruct_upload(data: LoadedData, params: dict) -> ReconstructionResult:
    clean = validate_params(
        params["detector_spacing_mm"],
        params["center_index"],
        params["output_size"],
        params["pixel_spacing_mm"],
        params["filter"],
    )
    sinogram = calibrate(data)
    image = fbp(
        sinogram,
        detector_spacing=clean["detector_spacing_mm"],
        center_index=clean["center_index"],
        output_size=clean["output_size"],
        pixel_spacing=clean["pixel_spacing_mm"],
        filter_name=clean["filter"],
    )
    stats = {
        "image_min": float(np.min(image)),
        "image_max": float(np.max(image)),
        "image_mean": float(np.mean(image)),
        "sinogram_min": float(np.min(sinogram)),
        "sinogram_max": float(np.max(sinogram)),
    }
    return ReconstructionResult(
        image=image,
        line_integrals=sinogram,
        params=clean,
        stats=stats,
    )
