"""FastAPI application exposing the parallel-beam CT reconstruction API."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from .dual_service import dual_energy_pipeline
from .fatigue import FATIGUE_MAX_OUTPUT_SIZE
from .fatigue_service import fatigue_pipeline
from .io_utils import ValidationError, load_mask_npz, load_npz
from .preview import npy_bytes, render_check_png, render_damage_png, render_png
from .reconstruct import supported_filters
from .section_service import section_check_pipeline
from .service import reconstruct_upload

app = FastAPI(title="Parallel-beam CT FBP reconstruction", version="1.0.0")


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(value: str) -> str:
    """Map an arbitrary case/material name to a safe ZIP entry stem."""
    slug = _SLUG_RE.sub("_", value.strip()).strip("._")
    return slug or "unnamed"


def _unique_zip_names(zf: zipfile.ZipFile, desired: dict[str, bytes]) -> None:
    """Write entries, disambiguating any filename collisions instead of silently
    overwriting (fixes stress ZIP collisions for names containing '_' or for
    duplicate case/material name combinations)."""
    used: set[str] = set()
    for candidate, payload in desired.items():
        final = candidate
        suffix = 1
        while final in used:
            stem, dot, extension = candidate.rpartition(".")
            base = stem if dot else candidate
            ext = f".{extension}" if dot else ""
            final = f"{base}__{suffix}{ext}"
            suffix += 1
        used.add(final)
        zf.writestr(final, payload)


def _payload_hashes(*payloads: bytes) -> list[str]:
    return [hashlib.sha256(payload).hexdigest() for payload in payloads]


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
    low_payload = await low_file.read()
    high_payload = await high_file.read()
    mask_payload = await mask_file.read()
    low_data = load_npz(low_payload)
    high_data = load_npz(high_payload)
    mask = load_mask_npz(mask_payload)
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

    planned_stress: list[tuple[str, str, str, bytes]] = []
    planned_png: list[tuple[str, str, bytes]] = []
    desired_stress_names: list[str] = []
    desired_png_names: list[str] = []
    for case_index, case_result in enumerate(result.case_results):
        case_name = case_result["name"]
        case_stem = _slugify(case_name)
        for material_index, (material_name, entry) in enumerate(
            case_result["materials"].items()
        ):
            desired = f"stress_{case_stem}_{_slugify(material_name)}.npy"
            desired_stress_names.append(desired)
            planned_stress.append(
                (case_name, material_name, desired, npy_bytes(entry["_stress_map"]))
            )
        png_desired = f"exceedance_{case_stem}.png"
        desired_png_names.append(png_desired)
        planned_png.append(
            (case_name, png_desired, render_check_png(case_result["_ratio_map"]))
        )
    stress_collision = len(set(desired_stress_names)) != len(desired_stress_names)
    png_collision = len(set(desired_png_names)) != len(desired_png_names)
    entries: dict[str, bytes] = {}
    file_links: dict[str, str] = {}
    for case_index, (case_name, material_name, desired, payload) in enumerate(
        planned_stress
    ):
        filename = (
            f"stress_{case_index:02d}_{desired[len('stress_'):]}"
            if stress_collision
            else desired
        )
        entries[filename] = payload
        file_links[f"stress:{case_name}:{material_name}"] = filename
    for case_index, (case_name, desired, payload) in enumerate(planned_png):
        filename = (
            f"exceedance_{case_index:02d}_{desired[len('exceedance_'):]}"
            if png_collision
            else desired
        )
        entries[filename] = payload
        file_links[f"exceedance:{case_name}"] = filename
    result.report["linkage"] = {
        "low_file_sha256": _payload_hashes(low_payload)[0],
        "high_file_sha256": _payload_hashes(high_payload)[0],
        "mask_file_sha256": _payload_hashes(mask_payload)[0],
        "files": file_links,
    }
    entries["report.json"] = json.dumps(
        result.report, indent=2, sort_keys=True
    ).encode()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _unique_zip_names(zf, entries)

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="section_check.zip"'},
    )


@app.post("/fatigue_check")
async def fatigue_check(
    low_file: UploadFile = File(..., description="NPZ with low-energy intensity, dark, flat"),
    high_file: UploadFile = File(..., description="NPZ with high-energy intensity, dark, flat"),
    mask_file: UploadFile = File(..., description="NPZ with boolean array 'mask'"),
    detector_spacing_mm: float = Form(...),
    center_index: float = Form(...),
    output_size: int = Form(...),
    pixel_spacing_mm: float = Form(...),
    filter: str = Form("ram-lak"),
    materials: str = Form(
        ...,
        description=(
            "JSON list of two materials with reference_density, elastic_modulus, "
            "allowables plus Sref, Nref, m, Su (stress in MPa)"
        ),
    ),
    mu_matrix: str = Form(..., description="JSON 2x2 mass attenuation matrix, mm^2/mg"),
    history: str = Form(
        ...,
        description="JSON ordered list of 2-2000 points {N, Mx, My}",
    ),
    R: int = Form(..., description="positive integer block repeat count"),
) -> Response:
    low_payload = await low_file.read()
    high_payload = await high_file.read()
    mask_payload = await mask_file.read()
    low_data = load_npz(low_payload)
    high_data = load_npz(high_payload)
    mask = load_mask_npz(mask_payload)
    result = fatigue_pipeline(
        low_data,
        high_data,
        mask,
        {
            "detector_spacing_mm": detector_spacing_mm,
            "center_index": center_index,
            "output_size": output_size,
            "pixel_spacing_mm": pixel_spacing_mm,
            "filter": filter,
            "max_output_size": FATIGUE_MAX_OUTPUT_SIZE,
        },
        materials,
        mu_matrix,
        history,
        R,
        low_payload,
        high_payload,
        mask_payload,
    )

    entries: dict[str, bytes] = {}
    file_links: dict[str, str] = {}
    for index, (name, damage_map) in enumerate(result.damage_maps.items()):
        npy_name = f"damage_{index:02d}_{_slugify(name)}.npy"
        png_name = f"damage_preview_{index:02d}_{_slugify(name)}.png"
        entries[npy_name] = npy_bytes(damage_map)
        entries[png_name] = render_damage_png(damage_map)
        file_links[f"damage:{name}"] = npy_name
        file_links[f"preview:{name}"] = png_name
    result.report["linkage"]["files"] = file_links
    entries["report.json"] = json.dumps(
        result.report, indent=2, sort_keys=True
    ).encode()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _unique_zip_names(zf, entries)

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="fatigue_check.zip"'},
    )
