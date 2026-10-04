import numpy as np
import pytest

from ctrecon.decompose import (
    decompose,
    nnls_2x2,
    validate_materials,
    validate_mu_matrix,
)
from ctrecon.io_utils import ValidationError
from ctrecon.roi import integrate_rois, validate_rois, validate_slice_thickness

A = np.array([[0.5, 0.2], [0.1, 0.4]])


def test_validate_materials_rejects_duplicates_and_empty():
    assert validate_materials([" a ", "b"]) == ("a", "b")
    for bad in (["a", "a"], ["a", ""], ["a"], "ab", [["a"], "b"]):
        with pytest.raises(ValidationError):
            validate_materials(bad)


def test_validate_mu_matrix_rules():
    assert validate_mu_matrix([[0.5, 0.2], [0.1, 0.4]]).shape == (2, 2)
    for bad in (
        [[1.0, 2.0]],                 # not 2x2
        [[0.5, 0.2], [0.1, 0.0]],     # non-positive
        [[0.5, 0.2], [0.1, np.inf]],  # non-finite
        [[1.0, 1.0], [1.0, 1.0 + 1e-9]],  # singular-ish: cond > 10000
    ):
        with pytest.raises(ValidationError):
            validate_mu_matrix(bad)


def test_nnls_matches_unconstrained_when_positive():
    low = np.array([[0.7]])
    high = np.array([[0.5]])
    rho1, rho2 = nnls_2x2(A, low, high)
    assert rho1[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert rho2[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_nnls_does_not_truncate_unconstrained_solution():
    # Unconstrained solution has rho1 < 0, rho2 large; naive truncation at
    # zero is NOT the constrained optimum.
    b = A @ np.array([-1.0, 2.0])
    rho1, rho2 = nnls_2x2(A, b[0:1].reshape(1, 1), b[1:2].reshape(1, 1))
    assert rho1[0, 0] == 0.0
    # Boundary optimum: best fit of b with material 2 alone.
    expected = float(A[:, 1] @ b / (A[:, 1] @ A[:, 1]))
    assert rho2[0, 0] == pytest.approx(expected, abs=1e-12)
    assert rho2[0, 0] != pytest.approx(2.0, abs=1e-3)


def test_nnls_satisfies_kkt_conditions():
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(2, 4000))
    rho1, rho2 = nnls_2x2(A, obs[0:1], obs[1:2])
    rho = np.vstack([rho1.ravel(), rho2.ravel()])
    assert (rho >= 0).all()
    grad = A.T @ (A @ rho - obs)
    assert np.all(grad >= -1e-9)
    assert np.all(np.abs(grad * rho) < 1e-9)


def test_decompose_residuals_are_prediction_minus_observation():
    low = np.array([[0.7, 0.1]])
    high = np.array([[0.5, 0.2]])
    rho1, rho2, res_low, res_high = decompose(A, low, high)
    assert np.allclose(A @ np.vstack([rho1[0], rho2[0]]) - np.vstack([low[0], high[0]]),
                       np.vstack([res_low[0], res_high[0]]))


def test_validate_rois_rules():
    rois = validate_rois([{"name": "r", "x0": 0, "y0": 0, "x1": 4, "y1": 3}], 8)
    assert rois[0].mask((8, 8)).sum() == 12
    for bad in (
        [],                                                    # too few
        [{"name": f"r{i}", "x0": 0, "y0": 0, "x1": 1, "y1": 1} for i in range(9)],
        [{"name": "a", "x0": 0, "y0": 0, "x1": 2, "y1": 2},
         {"name": "a", "x0": 0, "y0": 0, "x1": 1, "y1": 1}],   # duplicate name
        [{"name": "a", "x0": 2, "y0": 0, "x1": 2, "y1": 2}],   # empty
        [{"name": "a", "x0": 0, "y0": 0, "x1": 9, "y1": 2}],   # out of bounds
        [{"name": "a", "x0": -1, "y0": 0, "x1": 2, "y1": 2}],  # negative
        [{"name": "a", "x0": 0.5, "y0": 0, "x1": 2, "y1": 2}], # non-integer
    ):
        with pytest.raises(ValidationError):
            validate_rois(bad, 8)


def test_integrate_rois_mass_and_mean_residual():
    density = np.full((4, 4), 2.0)  # mg/mm^3
    residual = np.full((4, 4), 0.5)
    rois = validate_rois([{"name": "q", "x0": 1, "y0": 1, "x1": 3, "y1": 3}], 4)
    (result,) = integrate_rois(
        rois, {"m": density}, {"low": residual},
        pixel_spacing_mm=0.5, slice_thickness_mm=2.0,
    )
    # 4 pixels * 2 mg/mm^3 * (0.5*0.5*2) mm^3 = 4 mg
    assert result["mass_mg"]["m"] == pytest.approx(4.0)
    assert result["mean_residual_per_mm"]["low"] == pytest.approx(0.5)
    assert result["n_pixels"] == 4


def test_slice_thickness_validation():
    assert validate_slice_thickness(1.5) == 1.5
    for bad in (0, -1.0, float("nan"), "1.0", True):
        with pytest.raises(ValidationError):
            validate_slice_thickness(bad)
