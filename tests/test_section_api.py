import io
import json
import zipfile

import numpy as np
from fastapi.testclient import TestClient

from ctrecon.app import app
from ctrecon.synthetic import acquire, disk_sinogram

client = TestClient(app)
PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")

GEOMETRY = dict(
    detector_spacing_mm="0.5",
    center_index="127.5",
    output_size="128",
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
    },
    {
        "name": "plastic",
        "reference_density_mg_per_mm3": 1.2,
        "elastic_modulus_mpa": 3000.0,
        "allowable_tension_mpa": 60.0,
        "allowable_compression_mpa": 80.0,
    },
]
MU = [[0.06, 0.02], [0.03, 0.018]]
DISKS = [((6.0, -4.0), 12.0, 2.7), ((-8.0, 5.0), 9.0, 1.2)]
LOAD_CASES = [
    {"name": "service", "N": 4.0e3, "Mx": 5.0e4, "My": -2.0e4},
    {"name": "overload", "N": -5.0e6, "Mx": 1.0e7, "My": 1.0e7},
]


def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _phantom_npzs():
    sino = [
        disk_sinogram(180, 256, 0.5, 127.5, center, radius, density)
        for center, radius, density in DISKS
    ]
    mu = np.asarray(MU)
    combined = np.einsum("em,mad->ead", mu, np.stack(sino))
    return [
        _npz(intensity=i, dark=d, flat=f)
        for i, d, f in (acquire(integrals) for integrals in combined)
    ]


def _mask_npz():
    mask = np.zeros((128, 128), dtype=bool)
    yy, xx = np.mgrid[0:128, 0:128]
    mask[(yy - 63.5) ** 2 + (xx - 63.5) ** 2 <= 30.0**2] = True
    return _npz(mask=mask)


def _post(low, high, mask, **overrides):
    params = dict(GEOMETRY)
    params.update(
        materials=json.dumps(MATERIAL_PROPS),
        mu_matrix=json.dumps(MU),
        load_cases=json.dumps(LOAD_CASES),
    )
    params.update(overrides)
    files = {
        "low_file": ("low.npz", low, "application/octet-stream"),
        "high_file": ("high.npz", high, "application/octet-stream"),
        "mask_file": ("mask.npz", mask, "application/octet-stream"),
    }
    return client.post("/section_check", files=files, data=params)


def test_section_check_bundle_and_report():
    low, high = _phantom_npzs()
    response = _post(low, high, _mask_npz())
    assert response.status_code == 200
    bundle = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(bundle.namelist())
    for case in LOAD_CASES:
        for material in ("aluminum", "plastic"):
            assert f"stress_{case['name']}_{material}.npy" in names
        assert f"exceedance_{case['name']}.png" in names
    assert "report.json" in names
    assert bundle.read("exceedance_service.png")[:8] == PNG_MAGIC

    stress = np.load(
        io.BytesIO(bundle.read("stress_service_aluminum.npy")), allow_pickle=False
    )
    assert stress.dtype == np.float64
    assert stress.shape == (128, 128)
    assert np.isnan(stress).any()  # outside material/mask is NaN

    report = json.loads(bundle.read("report.json"))
    stiffness = np.asarray(report["section"]["stiffness_matrix"])
    assert stiffness.shape == (3, 3)
    assert np.all(np.isfinite(stiffness))
    by_case = {case["name"]: case for case in report["cases"]}
    service = by_case["service"]
    assert set(service["strain_curvature"]) == {"eps0", "kx_per_mm", "ky_per_mm"}
    for key in ("N", "Mx", "My"):
        assert abs(service["equilibrium_residual"][key]) < 1e-3
    alu = service["materials"]["aluminum"]
    assert alu["utilization"] >= 0.0
    assert -63.75 <= alu["worst_position_mm"]["x"] <= 63.75
    # The overload case must fail somewhere; the mild case must pass.
    assert by_case["overload"]["passed"] is False
    assert service["passed"] is True
    assert report["passed"] is False


def test_section_check_rejects_bad_inputs():
    low, high = _phantom_npzs()
    mask = _mask_npz()
    bad_mask_float = _npz(mask=np.ones((128, 128)))
    bad_mask_shape = _npz(mask=np.ones((64, 64), dtype=bool))
    bad_mask_empty = _npz(mask=np.zeros((128, 128), dtype=bool))
    bad_mask_missing = _npz(other=np.zeros((128, 128), dtype=bool))
    for bad in (bad_mask_float, bad_mask_shape, bad_mask_empty, bad_mask_missing):
        assert _post(low, high, bad).status_code == 422

    bad_params = [
        {"materials": json.dumps([MATERIAL_PROPS[0]])},
        {"materials": json.dumps([MATERIAL_PROPS[0], MATERIAL_PROPS[0]])},
        {"materials": json.dumps(
            [MATERIAL_PROPS[0], {**MATERIAL_PROPS[1], "elastic_modulus_mpa": -1.0}]
        )},
        {"mu_matrix": json.dumps([[0.06, 0.02], [0.03]])},  # ragged rows
        {"load_cases": json.dumps([])},
        {"load_cases": json.dumps(
            [{"name": f"c{i}", "N": 0, "Mx": 0, "My": 0} for i in range(9)]
        )},
        {"load_cases": json.dumps(
            [
                {"name": "a", "N": 0, "Mx": 0, "My": 0},
                {"name": "a", "N": 1, "Mx": 0, "My": 0},
            ]
        )},
        {"load_cases": json.dumps([{"name": "a", "N": float("nan"), "Mx": 0, "My": 0}])},
        {"load_cases": json.dumps([{"name": "a", "N": 0, "Mx": 0}])},
        {"load_cases": "not json"},
    ]
    for overrides in bad_params:
        response = _post(low, high, mask, **overrides)
        assert response.status_code == 422, overrides
        assert "error" in response.json()


def test_section_check_rejects_singular_section():
    low, high = _phantom_npzs()
    # Single-pixel mask: no bending stiffness.
    mask = np.zeros((128, 128), dtype=bool)
    mask[64, 64] = True
    response = _post(low, high, _npz(mask=mask))
    assert response.status_code == 422


def test_section_check_zip_keeps_sanitized_duplicate_names():
    low, high = _phantom_npzs()
    duplicate_cases = [
        {"name": "service?", "N": 0.0, "Mx": 0.0, "My": 0.0},
        {"name": "service!", "N": 0.0, "Mx": 0.0, "My": 0.0},
    ]
    response = _post(low, high, _mask_npz(), load_cases=json.dumps(duplicate_cases))
    assert response.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(response.content)).namelist())
    assert "stress_service_aluminum.npy" in names
    assert "stress_service_aluminum_2.npy" in names
    assert "exceedance_service.png" in names
    assert "exceedance_service_2.png" in names
