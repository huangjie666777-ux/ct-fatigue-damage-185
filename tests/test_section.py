import numpy as np
import pytest

from ctrecon.io_utils import ValidationError
from ctrecon.section import (
    check_load_case,
    pixel_coordinates,
    section_stiffness,
    validate_load_cases,
    validate_mask,
    validate_section_materials,
    volume_fractions,
)

STEEL = {
    "name": "steel",
    "reference_density_mg_per_mm3": 7.8,
    "elastic_modulus_mpa": 200000.0,
    "allowable_tension_mpa": 250.0,
    "allowable_compression_mpa": 250.0,
}
ALU = {
    "name": "alu",
    "reference_density_mg_per_mm3": 2.7,
    "elastic_modulus_mpa": 70000.0,
    "allowable_tension_mpa": 150.0,
    "allowable_compression_mpa": 100.0,
}


def _materials():
    return validate_section_materials([STEEL, ALU])


def test_validate_section_materials_rules():
    mats = _materials()
    assert [m.name for m in mats] == ["steel", "alu"]
    bad = [
        [STEEL],
        [STEEL, STEEL],
        [STEEL, {**ALU, "elastic_modulus_mpa": 0}],
        [STEEL, {**ALU, "reference_density_mg_per_mm3": -1}],
        [STEEL, {**ALU, "allowable_tension_mpa": float("nan")}],
        [STEEL, {k: v for k, v in ALU.items() if k != "allowable_compression_mpa"}],
    ]
    for spec in bad:
        with pytest.raises(ValidationError):
            validate_section_materials(spec)


def test_validate_load_cases_rules():
    cases = validate_load_cases([{"name": "lc1", "N": 1.0, "Mx": 0.0, "My": -2.0}])
    assert cases[0].my == -2.0
    bad = [
        [],
        [{"name": f"c{i}", "N": 0, "Mx": 0, "My": 0} for i in range(9)],
        [{"name": "a", "N": 0, "Mx": 0, "My": 0}, {"name": "a", "N": 1, "Mx": 0, "My": 0}],
        [{"name": "a", "N": float("inf"), "Mx": 0, "My": 0}],
        [{"name": "a", "N": 0, "Mx": 0}],
    ]
    for spec in bad:
        with pytest.raises(ValidationError):
            validate_load_cases(spec)


def test_validate_mask_rules():
    mask = np.ones((4, 4), dtype=bool)
    assert validate_mask(mask, 4).shape == (4, 4)
    for bad in (
        np.ones((4, 4)),            # not boolean
        np.ones((3, 4), dtype=bool),  # wrong shape
        np.zeros((4, 4), dtype=bool),  # empty
    ):
        with pytest.raises(ValidationError):
            validate_mask(bad, 4)


def test_volume_fractions_normalize_and_keep_void():
    materials = _materials()
    mask = np.ones((2, 2), dtype=bool)
    densities = {
        "steel": np.full((2, 2), 7.8 * 0.8),   # phi 0.8
        "alu": np.full((2, 2), 2.7 * 0.5),     # phi 0.5 -> sum 1.3
    }
    fractions = volume_fractions(densities, materials, mask)
    assert fractions["steel"][0, 0] == pytest.approx(0.8 / 1.3)
    assert fractions["alu"][0, 0] == pytest.approx(0.5 / 1.3)

    densities = {"steel": np.full((2, 2), 7.8 * 0.3), "alu": np.zeros((2, 2))}
    fractions = volume_fractions(densities, materials, mask)
    assert fractions["steel"][0, 0] == pytest.approx(0.3)  # void kept

    mask[0, 0] = False
    fractions = volume_fractions(densities, materials, mask)
    assert fractions["steel"][0, 0] == 0.0
    assert densities["steel"][0, 0] == pytest.approx(7.8 * 0.3)  # input untouched


def _homogeneous_section(n=16, spacing=1.0, e=200000.0):
    """Single-material square section, phi=1 inside a full mask."""
    fractions = {"steel": np.ones((n, n)), "alu": np.zeros((n, n))}
    materials = validate_section_materials([STEEL, ALU])
    x_map, y_map = pixel_coordinates(n, spacing)
    stiffness = section_stiffness(fractions, materials, x_map, y_map, spacing**2)
    return materials, fractions, stiffness, x_map, y_map, n, spacing


def test_pure_axial_matches_analytic():
    materials, fractions, stiffness, x_map, y_map, n, spacing = _homogeneous_section()
    e = STEEL["elastic_modulus_mpa"]
    area = (n * spacing) ** 2
    n_load = 50000.0
    result = check_load_case(
        validate_load_cases([{"name": "ax", "N": n_load, "Mx": 0, "My": 0}])[0],
        fractions, materials, stiffness, x_map, y_map,
    )
    eps0 = result["strain_curvature"]["eps0"]
    assert eps0 == pytest.approx(n_load / (e * area), rel=1e-9)
    steel = result["materials"]["steel"]
    assert steel["max_tension_mpa"] == pytest.approx(n_load / area, rel=1e-9)
    assert result["equilibrium_residual"]["N"] == pytest.approx(0.0, abs=1e-6)


def test_pure_bending_matches_ei():
    materials, fractions, stiffness, x_map, y_map, n, spacing = _homogeneous_section()
    e = STEEL["elastic_modulus_mpa"]
    side = n * spacing
    inertia = side**4 / 12.0
    mx = 1.0e6
    result = check_load_case(
        validate_load_cases([{"name": "bend", "N": 0, "Mx": mx, "My": 0}])[0],
        fractions, materials, stiffness, x_map, y_map,
    )
    kx = result["strain_curvature"]["kx_per_mm"]
    # Pixel-center integration gives I = side^2*(side^2-spacing^2)/12.
    assert kx == pytest.approx(mx / (e * inertia), rel=5e-3)
    # Extreme fibre stress at y = +side/2 approximated by outermost pixel center.
    y_max = (n - 1) / 2 * spacing
    steel = result["materials"]["steel"]
    assert steel["max_tension_mpa"] == pytest.approx(e * kx * y_max, rel=1e-9)
    assert steel["max_compression_mpa"] == pytest.approx(-e * kx * y_max, rel=1e-9)
    assert steel["worst_position_mm"]["y"] == pytest.approx(y_max)


def test_eccentric_composite_section_utilization_and_position():
    # Steel on the right half, aluminum on the left; tension + My shifts the
    # neutral axis and must overload the stiffer steel side.
    materials = validate_section_materials(
        [STEEL, {**ALU, "allowable_tension_mpa": 1.0e9, "allowable_compression_mpa": 1.0e9}]
    )
    n, spacing = 16, 1.0
    mask = np.ones((n, n), dtype=bool)
    fractions = {
        "steel": np.zeros((n, n)),
        "alu": np.zeros((n, n)),
    }
    fractions["steel"][:, n // 2 :] = 1.0
    fractions["alu"][:, : n // 2] = 1.0
    x_map, y_map = pixel_coordinates(n, spacing)
    stiffness = section_stiffness(fractions, materials, x_map, y_map, spacing**2)
    case = validate_load_cases([{"name": "combo", "N": 1.0e5, "Mx": 2.0e5, "My": -3.0e5}])[0]
    result = check_load_case(case, fractions, materials, stiffness, x_map, y_map)

    # Joint solve reproduces the load resultants (no dropped coupling terms).
    for key in ("N", "Mx", "My"):
        assert result["equilibrium_residual"][key] == pytest.approx(0.0, abs=1e-4)
    steel = result["materials"]["steel"]
    assert steel["utilization"] > 0.0
    assert steel["worst_position_mm"]["x"] > 0.0  # steel occupies +x half
    assert result["passed"] is (
        steel["utilization"] <= 1.0 and result["materials"]["alu"]["utilization"] <= 1.0
    )


def test_singular_section_rejected():
    materials = _materials()
    n = 8
    fractions = {"steel": np.zeros((n, n)), "alu": np.zeros((n, n))}
    fractions["steel"][3, 3] = 1.0  # single pixel: no bending stiffness
    x_map, y_map = pixel_coordinates(n, 1.0)
    with pytest.raises(ValidationError):
        section_stiffness(fractions, materials, x_map, y_map, 1.0)
