"""FastAPI application exposing the parallel-beam CT reconstruction API."""

from __future__ import annotations

import io
import json
import hashlib
import re
import zipfile

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from .dual_service import dual_energy_pipeline
from .fatigue_service import fatigue_pipeline
from .fatigue import validate_history, validate_repetitions
from .io_utils import ValidationError, load_history_npz, load_mask_npz, load_npz
from .preview import npy_bytes, render_check_png, render_png
from .reconstruct import supported_filters
from .section_service import section_check_pipeline
from .service import reconstruct_upload

app = FastAPI(title="Parallel-beam CT FBP reconstruction", version="1.0.0")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zip_safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._") or "unnamed"
    return cleaned[:80]


def _unique_path(paths: set[str], path: str) -> str:
    candidate = path
    match = re.search(r"\.[A-Za-z0-9]+$", path)
    stem = path[: match.start()] if match else path
    suffix = path[match.start() :] if match else ""
    counter = 2
    while candidate in paths:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    paths.add(candidate)
    return candidate


@app.exception_handler(ValidationError)
async def _validation_handler(_request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "filters": list(supported_filters())}


@app.post("/reconstruct")
async def reconstruct(
    file: UploadFile = File(..., description="NPZ with intensity, dark, flat"),
    detector_spacing_mm: float = Form(..., alias="detector_spacing_mm"),
    center_index: float = Form(...),
    output_size: int = Form(...),
    pixel_spacing_mm: float = Form(...),
    filter: str = Form("ram-lak"),
) -> Response:
    payload = await file.read()
    data = load_npz(payload)
    result = reconstruct_upload(
        data,
        {
            "detector_spacing_mm": detector_spacing_mm,
            "center_index": center_index,
            "output_size": output_size,
            "pixel_spacing_mm": pixel_spacing_mm,
            "filter": filter,
        },
    )

    metadata = {
        "parameters": result.params,
        "ranges": {
            "image_min": result.stats["image_min"],
            "image_max": result.stats["image_max"],
            "image_mean": result.stats["image_mean"],
            "sinogram_min": result.stats["sinogram_min"],
            "sinogram_max": result.stats["sinogram_max"],
        },
        "units": {
            "detector_spacing_mm": "millimeter",
            "pixel_spacing_mm": "millimeter",
            "image": "linear attenuation coefficient per millimeter",
        },
        "notes": "preview.png uses a min/max linear stretch for display only; reconstruction.npy is untouched float64 data.",
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("reconstruction.npy", npy_bytes(result.image))
        zf.writestr("preview.png", render_png(result.image))
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, sort_keys=True))

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="reconstruction.zip"'},
    )


@app.post("/decompose")
async def decompose(
    low_file: UploadFile = File(..., description="NPZ with low-energy intensity, dark, flat"),
    high_file: UploadFile = File(..., description="NPZ with high-energy intensity, dark, flat"),
    detector_spacing_mm: float = Form(...),
    center_index: float = Form(...),
    output_size: int = Form(...),
    pixel_spacing_mm: float = Form(...),
    filter: str = Form("ram-lak"),
    materials: str = Form(..., description='JSON list of two material names'),
    mu_matrix: str = Form(..., description="JSON 2x2 mass attenuation matrix, mm^2/mg"),
    slice_thickness_mm: float = Form(...),
    rois: str = Form(..., description="JSON list of ROI rectangles"),
) -> Response:
    low_data = load_npz(await low_file.read())
    high_data = load_npz(await high_file.read())
    result = dual_energy_pipeline(
        low_data,
        high_data,
        {
            "detector_spacing_mm": detector_spacing_mm,
            "center_index": center_index,
            "output_size": output_size,
            "pixel_spacing_mm": pixel_spacing_mm,
            "filter": filter,
        },
        materials,
        mu_matrix,
        slice_thickness_mm,
        rois,
    )

    metadata = {
        "parameters": result.params,
        "rois": result.roi_results,
        "ranges": {
            f"density_{name}": {
                "min": float(np.min(density)),
                "max": float(np.max(density)),
            }
            for name, density in result.densities.items()
        },
        "units": {
            "density": "mg/mm^3",
            "residual": "linear attenuation per millimeter (mm^-1)",
            "mass": "milligram",
            "mu_matrix": "mm^2/mg",
        },
        "notes": "preview PNGs use a min/max linear stretch for display only; "
        "density NPY files are untouched float64 data.",
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, density in result.densities.items():
            zf.writestr(f"density_{name}.npy", npy_bytes(density))
        for energy, residual in result.residuals.items():
            zf.writestr(f"residual_{energy}.npy", npy_bytes(residual))
        for name, density in result.densities.items():
            zf.writestr(f"preview_{name}.png", render_png(density))
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, sort_keys=True))

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="decomposition.zip"'},
    )


@app.post("/section_check")
async def section_check(
    low_file: UploadFile = File(..., description="NPZ with low-energy intensity, dark, flat"),
    high_file: UploadFile = File(..., description="NPZ with high-energy intensity, dark, flat"),
    mask_file: UploadFile = File(..., description="NPZ with boolean array 'mask'"),
    detector_spacing_mm: float = Form(...),
    center_index: float = Form(...),
    output_size: int = Form(...),
    pixel_spacing_mm: float = Form(...),
    filter: str = Form("ram-lak"),
    materials: str = Form(..., description="JSON list of two materials with mechanical properties"),
    mu_matrix: str = Form(..., description="JSON 2x2 mass attenuation matrix, mm^2/mg"),
    load_cases: str = Form(..., description="JSON list of 1-8 load cases with N, Mx, My"),
) -> Response:
    low_data = load_npz(await low_file.read())
    high_data = load_npz(await high_file.read())
    mask = load_mask_npz(await mask_file.read())
    result = section_check_pipeline(
        low_data,
        high_data,
        mask,
        {
            "detector_spacing_mm": detector_spacing_mm,
            "center_index": center_index,
            "output_size": output_size,
            "pixel_spacing_mm": pixel_spacing_mm,
            "filter": filter,
        },
        materials,
        mu_matrix,
        load_cases,
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        used_names = set()
        for case_result in result.case_results:
            case_name = _zip_safe(case_result["name"])
            for material_name, entry in case_result["materials"].items():
                path = _unique_path(
                    used_names,
                    f"stress_{case_name}_{_zip_safe(material_name)}.npy",
                )
                zf.writestr(path, npy_bytes(entry["_stress_map"]))
            zf.writestr(
                _unique_path(used_names, f"exceedance_{case_name}.png"),
                render_check_png(case_result["_ratio_map"]),
            )
        zf.writestr("report.json", json.dumps(result.report, indent=2, sort_keys=True))

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="section_check.zip"'},
    )


@app.post("/fatigue")
async def fatigue(
    low_file: UploadFile = File(...),
    high_file: UploadFile = File(...),
    mask_file: UploadFile = File(...),
    history_file: UploadFile = File(..., description="NPZ with ordered N, Mx, My arrays"),
    detector_spacing_mm: float = Form(...),
    center_index: float = Form(...),
    output_size: int = Form(...),
    pixel_spacing_mm: float = Form(...),
    filter: str = Form("ram-lak"),
    materials: str = Form(..., description="JSON two materials with Sref, Nref, m, Su"),
    mu_matrix: str = Form(...),
    R: str = Form(..., description="positive integer block repetitions"),
) -> Response:
    low_payload = await low_file.read()
    high_payload = await high_file.read()
    mask_payload = await mask_file.read()
    history_payload = await history_file.read()

    low_data = load_npz(low_payload)
    high_data = load_npz(high_payload)
    mask = load_mask_npz(mask_payload)
    history = validate_history(load_history_npz(history_payload))
    repetitions = validate_repetitions(R)
    result = fatigue_pipeline(
        low_data,
        high_data,
        mask,
        history,
        {
            "detector_spacing_mm": detector_spacing_mm,
            "center_index": center_index,
            "output_size": output_size,
            "pixel_spacing_mm": pixel_spacing_mm,
            "filter": filter,
        },
        materials,
        mu_matrix,
        repetitions,
        {
            "low_file_sha256": _sha256(low_payload),
            "high_file_sha256": _sha256(high_payload),
            "mask_file_sha256": _sha256(mask_payload),
            "history_file_sha256": _sha256(history_payload),
        },
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, damage in result.damage_maps.items():
            zf.writestr(f"damage_{_zip_safe(name)}.npy", npy_bytes(damage))
        zf.writestr("damage_preview.png", render_check_png(result.combined_map))
        zf.writestr("fatigue_report.json", json.dumps(result.report, indent=2, sort_keys=True, allow_nan=False))

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="fatigue.zip"'},
    )
