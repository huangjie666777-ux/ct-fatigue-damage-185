"""FastAPI application exposing the parallel-beam CT reconstruction API."""

from __future__ import annotations

import io
import json
import zipfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from .io_utils import ValidationError, load_npz
from .preview import npy_bytes, render_png
from .reconstruct import supported_filters
from .service import reconstruct_upload

app = FastAPI(title="Parallel-beam CT FBP reconstruction", version="1.0.0")


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
