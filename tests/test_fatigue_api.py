import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ctrecon.app import app
from ctrecon.synthetic import acquire, disk_sinogram

client = TestClient(app)
PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")

GEOMETRY = dict(
    detector_spacing_mm="0.5",
    center_index="63.5",
    output_size="48",
    pixel_spacing_mm="0.5",
    filter="hann",
)
MATERIAL_PROPS = [
    {
        "name": "aluminum",
        "reference_density_mg_per_mm3": 2.7,
        "elastic_modulus_mpa": 70000.0,
        "allowable_tension_mpa": 150.0,
        "allowable_compression_mpa": 120.0,
        "Sref": 80.0,
        "Nref": 1.0e6,
        "m": 4.0,
        "Su": 300.0,
    },
    {
        "name": "plastic",
        "reference_density_mg_per_mm3": 1.2,
        "elastic_modulus_mpa": 3000.0,
        "allowable_tension_mpa": 60.0,
        "allowable_compression_mpa": 80.0,
        "Sref": 30.0,
        "Nref": 1.0e5,
        "m": 5.0,
        "Su": 80.0,
    },
]
MU = [[0.06, 0.02], [0.03, 0.018]]
DISKS = [((4.0, -2.0), 6.0, 2.7), ((-4.0, 3.0), 5.0, 1.2)]
HISTORY = [
    {"N": 3000.0, "Mx": 40000.0, "My": -10000.0},
    {"N": -2000.0, "Mx": -30000.0, "My": 15000.0},
    {"N": 1000.0, "Mx": 20000.0, "My": 0.0},
    {"N": 3000.0, "Mx": 40000.0, "My": -10000.0},
]


def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _phantom_npzs():
    sino = [
        disk_sinogram(120, 128, 0.5, 63.5, center, radius, density)
        for center, radius, density in DISKS
    ]
    combined = np.einsum("em,mad->ead", np.asarray(MU), np.stack(sino))
    return [
        ("scan.npz", _npz(intensity=i, dark=d, flat=f), "application/octet-stream")
        for i, d, f in (acquire(integrals) for integrals in combined)
    ]


def _mask_npz(size=48):
    mask = np.zeros((size, size), dtype=bool)
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    mask[(yy - center) ** 2 + (xx - center) ** 2 <= 15.0**2] = True
    return ("mask.npz", _npz(mask=mask), "application/octet-stream")


def _post(overrides=None, geometry=None):
    params = dict(GEOMETRY)
    if geometry:
        params.update(geometry)
    params.update(
        materials=json.dumps(MATERIAL_PROPS),
        mu_matrix=json.dumps(MU),
        history=json.dumps(HISTORY),
        R="25",
    )
    if overrides:
        params.update(overrides)
    low, high = _phantom_npzs()
    return client.post(
        "/fatigue_check",
        files={"low_file": low, "high_file": high, "mask_file": _mask_npz()},
        data=params,
    )


def test_fatigue_bundle_report_and_accumulation():
    response = _post()
    assert response.status_code == 200, response.text
    bundle = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(bundle.namelist())
    assert {
        "damage_00_aluminum.npy",
        "damage_01_plastic.npy",
        "damage_preview_00_aluminum.png",
        "damage_preview_01_plastic.png",
        "report.json",
    } <= names
    assert bundle.read("damage_preview_00_aluminum.png")[:8] == PNG_MAGIC

    damage = np.load(
        io.BytesIO(bundle.read("damage_00_aluminum.npy")), allow_pickle=False
    )
    assert damage.dtype == np.float64
    assert damage.shape == (48, 48)
    assert np.isnan(damage).any()

    report = json.loads(bundle.read("report.json"))
    critical = report["critical_point"]
    assert critical["material"] in ("aluminum", "plastic")
    assert critical["accumulated_damage"] == pytest.approx(
        25 * critical["single_block_damage"]
    )
    assert critical["blocks_to_failure"] == pytest.approx(
        1.0 / critical["single_block_damage"]
    )
    assert critical["zero_damage"] is False
    assert len(critical["cycles"]) >= 1
    for cycle in critical["cycles"]:
        assert cycle["count"] in (0.5, 1.0)
        assert cycle["amplitude_mpa"] > 0.0
    linkage = report["linkage"]
    assert all(
        len(linkage[key]) == 64
        for key in (
            "low_file_sha256",
            "high_file_sha256",
            "mask_file_sha256",
            "history_sha256",
        )
    )
    assert linkage["files"]["damage:aluminum"] == "damage_00_aluminum.npy"
    for name, material in report["materials"].items():
        if material["present_pixels"]:
            assert material["max_accumulated_damage"] >= 0.0


def test_fatigue_zero_damage_returns_null_life():
    flat_history = [
        {"N": 1000.0, "Mx": 0.0, "My": 0.0},
        {"N": 1000.0, "Mx": 0.0, "My": 0.0},
    ]
    response = _post({"history": json.dumps(flat_history)})
    assert response.status_code == 200, response.text
    report = json.loads(
        zipfile.ZipFile(io.BytesIO(response.content)).read("report.json")
    )
    critical = report["critical_point"]
    assert critical["single_block_damage"] == 0.0
    assert critical["accumulated_damage"] == 0.0
    assert critical["blocks_to_failure"] is None
    assert critical["zero_damage"] is True
    assert critical["cycles"] == []


def test_fatigue_rejects_bad_inputs():
    bad_cases = [
        {"history": json.dumps([HISTORY[0]])},
        {"history": json.dumps([HISTORY[0]] * 2001)},
        {"history": json.dumps([{"N": 0, "Mx": 0}, HISTORY[0]])},
        {"history": json.dumps(
            [{"N": "x", "Mx": 0, "My": 0}, HISTORY[0]]
        )},
        {"history": "not-json"},
        {"R": "0"},
        {"R": "2.5"},
        {"R": "true"},
        {"materials": json.dumps([MATERIAL_PROPS[0]])},
        {"materials": json.dumps(
            [{**MATERIAL_PROPS[0], "Sref": -1.0}, MATERIAL_PROPS[1]]
        )},
        {"materials": json.dumps(
            [{k: v for k, v in MATERIAL_PROPS[0].items() if k != "Su"},
             MATERIAL_PROPS[1]]
        )},
        {"mu_matrix": json.dumps([[0.06, 0.02], [0.03]])},
    ]
    for overrides in bad_cases:
        response = _post(overrides)
        assert response.status_code == 422, overrides
        body = response.json()
        assert "error" in body or "detail" in body
    response = _post(geometry={"output_size": "65"})
    assert response.status_code == 422


def test_fatigue_rejects_nonpositive_goodman_denominator():
    materials = [
        {**MATERIAL_PROPS[0], "Su": 1.0},
        MATERIAL_PROPS[1],
    ]
    response = _post({"materials": json.dumps(materials)})
    assert response.status_code == 422
    assert "Goodman" in response.json()["error"]
