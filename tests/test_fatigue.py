import numpy as np
import pytest

from ctrecon.fatigue import (
    FatigueMaterial,
    block_damage,
    rainflow_cycles,
    turning_points,
    validate_fatigue_materials,
    validate_history,
    validate_repeat_count,
)
from ctrecon.io_utils import ValidationError
from ctrecon.section import SectionMaterial


def _material(**overrides):
    values = dict(
        name="steel",
        reference_density=7.85,
        elastic_modulus=200000.0,
        allowable_tension=250.0,
        allowable_compression=250.0,
        s_ref=100.0,
        n_ref=1.0e6,
        sn_exponent=4.0,
        su=400.0,
    )
    values.update(overrides)
    base = SectionMaterial(
        name=values["name"],
        reference_density=values["reference_density"],
        elastic_modulus=values["elastic_modulus"],
        allowable_tension=values["allowable_tension"],
        allowable_compression=values["allowable_compression"],
    )
    return FatigueMaterial(
        base=base,
        s_ref=values["s_ref"],
        n_ref=values["n_ref"],
        sn_exponent=values["sn_exponent"],
        su=values["su"],
    )


def test_turning_points_keep_endpoints_and_drop_repeats():
    series = np.array([0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0])
    assert turning_points(series).tolist() == [0.0, 5.0, 0.0]
    series = np.array([1.0, 1.0, 1.0])
    assert turning_points(series).tolist() == [1.0]


def test_rainflow_counts_closed_and_residual_cycles():
    # Interior small reversal closes first; the outer ranges stay residual.
    series = np.array([0.0, 10.0, 5.0, 10.0, 0.0])
    cycles = rainflow_cycles(turning_points(series))
    closed = [c for c in cycles if c.count == 1.0]
    halves = [c for c in cycles if c.count == 0.5]
    assert len(closed) == 1
    assert closed[0].amplitude == pytest.approx(2.5)
    assert closed[0].mean == pytest.approx(7.5)
    assert len(halves) == 2

    # Head and tail are never joined: [0,10,0,10] gives three half cycles.
    cycles = rainflow_cycles(turning_points(np.array([0.0, 10.0, 0.0, 10.0])))
    assert all(c.count == 0.5 for c in cycles)
    assert len(cycles) == 3

    # Monotonic two-point block: one residual half cycle only.
    cycles = rainflow_cycles(np.array([2.0, 5.0]))
    assert [(c.count, c.amplitude) for c in cycles] == [(0.5, 1.5)]


def test_rainflow_astm_example():
    # ASTM E1049 worked example extrema: counts by (range, mean).
    extrema = [-2, 1, -3, 5, -1, 3, -4, 4, -2]
    cycles = rainflow_cycles(np.asarray(extrema, dtype=float))
    by_key = {}
    for cycle in cycles:
        key = (round(2 * cycle.amplitude, 6), round(cycle.mean, 6))
        by_key[key] = by_key.get(key, 0.0) + cycle.count
    assert by_key[(4.0, 1.0)] == pytest.approx(1.0)
    assert by_key[(3.0, -0.5)] == pytest.approx(0.5)
    assert by_key[(4.0, -1.0)] == pytest.approx(0.5)
    assert by_key[(8.0, 1.0)] == pytest.approx(0.5)
    assert by_key[(9.0, 0.5)] == pytest.approx(0.5)
    assert by_key[(8.0, 0.0)] == pytest.approx(0.5)
    assert by_key[(6.0, 1.0)] == pytest.approx(0.5)


def test_block_damage_goodman_and_zero_amplitude():
    material = _material()
    # Zero amplitude history contributes nothing and does not divide by zero.
    damage, details = block_damage(np.array([7.0, 7.0, 7.0]), material)
    assert damage == 0.0
    assert details == []

    # Closed inner cycle range 50 (a=25, mean 75), outer residuals range 100.
    history = np.array([0.0, 100.0, 50.0, 100.0, 0.0])
    damage, details = block_damage(history, material)
    closed = next(d for d in details if d["count"] == 1.0)
    corrected = 25.0 / (1.0 - 75.0 / 400.0)
    n_fail = 1.0e6 * (100.0 / corrected) ** 4
    assert closed["amplitude_mpa"] == pytest.approx(25.0)
    assert closed["mean_mpa"] == pytest.approx(75.0)
    assert closed["goodman_amplitude_mpa"] == pytest.approx(corrected)
    assert closed["Nf"] == pytest.approx(n_fail)
    assert damage == pytest.approx(sum(d["damage"] for d in details))


def test_block_damage_rejects_nonpositive_goodman_denominator():
    material = _material(su=100.0)
    with pytest.raises(ValidationError, match="Goodman"):
        block_damage(np.array([0.0, 200.0, 0.0, 200.0]), material)
    material = _material(su=100.0)  # mean == Su on a non-zero cycle
    with pytest.raises(ValidationError, match="Goodman"):
        block_damage(np.array([0.0, 200.0]), material)


def test_history_and_repeat_validation():
    valid = [{"N": 1.0, "Mx": 2.0, "My": 3.0}, {"N": 0.0, "Mx": 0.0, "My": 0.0}]
    assert len(validate_history(valid)) == 2
    with pytest.raises(ValidationError):
        validate_history([valid[0]])
    with pytest.raises(ValidationError):
        validate_history([valid[0]] * 2001)
    with pytest.raises(ValidationError):
        validate_history(
            [{"N": float("nan"), "Mx": 0.0, "My": 0.0}, valid[0]]
        )
    with pytest.raises(ValidationError):
        validate_history([{"N": 0, "Mx": 0}, valid[0]])
    assert validate_repeat_count(3) == 3
    for bad in (0, -1, 1.5, True, "3"):
        with pytest.raises(ValidationError):
            validate_repeat_count(bad)


def test_material_validation_requires_fatigue_fields():
    spec = [
        {
            "name": "a",
            "reference_density_mg_per_mm3": 1.0,
            "elastic_modulus_mpa": 1.0,
            "allowable_tension_mpa": 1.0,
            "allowable_compression_mpa": 1.0,
        },
        {
            "name": "b",
            "reference_density_mg_per_mm3": 1.0,
            "elastic_modulus_mpa": 1.0,
            "allowable_tension_mpa": 1.0,
            "allowable_compression_mpa": 1.0,
            "Sref": 1.0,
            "Nref": 1.0,
            "m": 1.0,
            "Su": 1.0,
        },
    ]
    with pytest.raises(ValidationError, match="Sref"):
        validate_fatigue_materials(spec)
    spec[0].update(Sref=0.0, Nref=1.0, m=1.0, Su=1.0)
    with pytest.raises(ValidationError, match="strictly positive"):
        validate_fatigue_materials(spec)
