import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ctrecon.app import app
from ctrecon.fatigue import (
    FatigueMaterial,
    rainflow_cycles,
    reduce_turning_points,
    validate_history,
    validate_repetitions,
)
from ctrecon.io_utils import ValidationError
from ctrecon.synthetic import acquire, disk_sinogram

client = TestClient(app)
PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")
MU = [[0.06, 0.02], [0.03, 0.018]]
DISKS = [((6.0, -4.0), 12.0, 2.7), ((-8.0, 5.0), 9.0, 1.2)]
MATERIALS = [
    {
        "name": "aluminum",
        "reference_density_mg_per_mm3": 2.7,
        "elastic_modulus_mpa": 70000.0,
        "allowable_tension_mpa": 150.0,
        "allowable_compression_mpa": 120.0,
        "Sref": 200.0,
        "Nref": 1.0e6,
        "m": 5.0,
        "Su": 300.0,
    },
    {
        "name": "plastic",
        "reference_density_mg_per_mm3": 1.2,
        "elastic_modulus_mpa": 3000.0,
        "allowable_tension_mpa": 60.0,
        "allowable_compression_mpa": 80.0,
        "Sref": 40.0,
        "Nref": 1.0e5,
        "m": 4.0,
        "Su": 80.0,
    },
]


def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _phantom_npzs():
    sinograms = [
        disk_sinogram(180, 256, 0.5, 127.5, center, radius, density)
        for center, radius, density in DISKS
    ]
    combined = np.einsum("em,mad->ead", np.asarray(MU), np.stack(sinograms))
    return [
        _npz(intensity=i, dark=d, flat=f)
        for i, d, f in (acquire(integrals) for integrals in combined)
    ]


def _mask_npz(size=64):
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    mask = (yy - center) ** 2 + (xx - center) ** 2 <= 24.0**2
    return _npz(mask=mask)


def _history_npz(n=5, scale=1000.0):
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return _npz(
        N=scale * np.sin(t),
        Mx=5000.0 * scale / 1000.0 * np.cos(t),
        My=-3000.0 * scale / 1000.0 * np.sin(2.0 * t),
    )


def _post(low, high, mask, history=None, **overrides):
    data = {
        "detector_spacing_mm": "0.5",
        "center_index": "127.5",
        "output_size": "64",
        "pixel_spacing_mm": "0.5",
        "filter": "hann",
        "materials": json.dumps(MATERIALS),
        "mu_matrix": json.dumps(MU),
        "R": "3",
    }
    data.update(overrides)
    files = {
        "low_file": ("low.npz", low, "application/octet-stream"),
        "high_file": ("high.npz", high, "application/octet-stream"),
        "mask_file": ("mask.npz", mask, "application/octet-stream"),
        "history_file": ("history.npz", history or _history_npz(), "application/octet-stream"),
    }
    return client.post("/fatigue", files=files, data=data)


def test_turning_reduction_and_rainflow_counts():
    assert reduce_turning_points(np.array([1, 1, 2, 2, 1, 1])) == [1.0, 2.0, 1.0]
    assert rainflow_cycles(np.array([0.0, 1.0])) == [(0.5, 0.5, 0.5)]
    cycles = rainflow_cycles(np.array([0.0, 1.0, 0.0]))
    assert cycles == [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    closed = rainflow_cycles(np.array([0.0, 1.0, 0.0, -1.0, 0.0]))
    assert [(a, c) for a, _, c in closed] == [
        (0.5, 0.5),
        (1.0, 0.5),
        (0.5, 0.5),
    ]


def test_history_and_repetition_validation():
    history = validate_history({
        "N": np.array([0.0, 1.0]),
        "Mx": np.array([0.0, 1.0]),
        "My": np.array([0.0, 1.0]),
    })
    assert history.axial.size == 2
    assert validate_repetitions("17") == 17
    for value in ("0", "-1", "1.0", " 1 ", "abc"):
        with pytest.raises(ValidationError):
            validate_repetitions(value)
    with pytest.raises(ValidationError):
        validate_history({
            "N": np.zeros(1),
            "Mx": np.zeros(1),
            "My": np.zeros(1),
        })
    with pytest.raises(ValidationError):
        validate_history({
            "N": np.array([0.0, 1.0]),
            "Mx": np.array([0.0]),
            "My": np.array([0.0, 1.0]),
        })


def test_fatigue_zip_and_report():
    low, high = _phantom_npzs()
    response = _post(low, high, _mask_npz())
    assert response.status_code == 200, response.text
    bundle = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(bundle.namelist())
    assert names == {
        "damage_aluminum.npy",
        "damage_plastic.npy",
        "damage_preview.png",
        "fatigue_report.json",
    }
    assert bundle.read("damage_preview.png")[:8] == PNG_MAGIC
    damage = np.load(io.BytesIO(bundle.read("damage_plastic.npy")), allow_pickle=False)
    assert damage.shape == (64, 64)
    assert damage.dtype == np.float64
    assert np.nanmax(damage) >= 0.0
    report = json.loads(bundle.read("fatigue_report.json"))
    assert report["maximum_cumulative_damage"] == pytest.approx(
        3.0 * report["maximum_damage_pixel"]["single_block_damage"]
    )
    assert report["total_block_life"] > 0.0
    assert report["source_hashes"]["history_file_sha256"]
    detail = report["cycles_at_maximum_pixel"]
    assert detail
    assert all(item["count"] in (0.5, 1.0) for item in detail)
    assert all(item["stress_amplitude_mpa"] > 0.0 for item in detail)


def test_fatigue_rejects_invalid_requests():
    low, high = _phantom_npzs()
    mask = _mask_npz()
    history = _history_npz()
    assert _post(low, high, _mask_npz(128), output_size="128").status_code == 422
    bad_history = _npz(N=np.ones(1), Mx=np.ones(1), My=np.ones(1))
    assert _post(low, high, mask, bad_history).status_code == 422
    mismatched = _npz(N=np.ones(3), Mx=np.ones(2), My=np.ones(3))
    assert _post(low, high, mask, mismatched).status_code == 422
    assert _post(low, high, mask, history, R="0").status_code == 422
    bad_materials = [
        {**MATERIALS[0], **{k: v for k, v in MATERIALS[1].items() if k != "Su"}},
        MATERIALS[1],
    ]
    assert _post(low, high, mask, history, materials=json.dumps(bad_materials)).status_code == 422
    bad_materials = [
        {**MATERIALS[0], "Su": 1.0},
        MATERIALS[1],
    ]
    assert _post(
        low, high, mask, _history_npz(scale=100000.0), materials=json.dumps(bad_materials)
    ).status_code == 422


def test_zero_damage_reports_null_life():
    low, high = _phantom_npzs()
    history = _npz(N=np.zeros(4), Mx=np.zeros(4), My=np.zeros(4))
    response = _post(low, high, _mask_npz(), history)
    assert response.status_code == 200, response.text
    report = json.loads(zipfile.ZipFile(io.BytesIO(response.content)).read("fatigue_report.json"))
    assert report["maximum_cumulative_damage"] == 0.0
    assert report["total_block_life"] is None
    assert report["zero_damage"] is True
    assert report["cycles_at_maximum_pixel"] == []
