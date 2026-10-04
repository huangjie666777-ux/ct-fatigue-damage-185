import io
import json
import zipfile

import numpy as np
from fastapi.testclient import TestClient

from ctrecon.app import app
from ctrecon.synthetic import acquire, disk_sinogram

client = TestClient(app)
PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")


def _upload(intensity, dark, flat, **params):
    buffer = io.BytesIO()
    np.savez(buffer, intensity=intensity, dark=dark, flat=flat)
    files = {"file": ("scan.npz", buffer.getvalue(), "application/octet-stream")}
    return client.post("/reconstruct", files=files, data=params)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "ram-lak" in response.json()["filters"]


def test_reconstruct_endpoint_returns_zip_bundle():
    truth = disk_sinogram(90, 128, 0.5, 63.5, (4.0, 2.0), 8.0, 0.2)
    intensity, dark, flat = acquire(truth)
    response = _upload(
        intensity, dark, flat,
        detector_spacing_mm="0.5",
        center_index="63.5",
        output_size="64",
        pixel_spacing_mm="0.5",
        filter="ram-lak",
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    bundle = zipfile.ZipFile(io.BytesIO(response.content))
    assert set(bundle.namelist()) == {
        "reconstruction.npy", "preview.png", "metadata.json"}

    metadata = json.loads(bundle.read("metadata.json"))
    assert metadata["parameters"]["output_size"] == 64
    assert metadata["ranges"]["image_min"] < metadata["ranges"]["image_max"]
    assert bundle.read("preview.png")[:8] == PNG_MAGIC

    image = np.load(io.BytesIO(bundle.read("reconstruction.npy")),
                    allow_pickle=False)
    assert image.dtype == np.float64
    assert image.shape == (64, 64)


def test_bad_upload_returns_422():
    response = _upload(
        np.ones((4, 8)), np.zeros(7), np.ones(8),
        detector_spacing_mm="1", center_index="3.5",
        output_size="8", pixel_spacing_mm="1", filter="ram-lak",
    )
    assert response.status_code == 422
    assert "error" in response.json()
