import io
import zipfile

import numpy as np
import pytest

from ctrecon.io_utils import (
    MAX_DETECTORS,
    MAX_OUTPUT_SIZE,
    ValidationError,
    load_npz,
    validate_params,
)


def _npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _good_payload(n_angles=4, n_det=8):
    return _npz(
        intensity=np.ones((n_angles, n_det)),
        dark=np.zeros(n_det),
        flat=np.ones(n_det) * 2,
    )


def test_valid_load():
    loaded = load_npz(_good_payload())
    assert loaded.intensity.shape == (4, 8)
    assert loaded.intensity.dtype == np.float64


def test_missing_array():
    with pytest.raises(ValidationError):
        load_npz(_npz(intensity=np.ones((2, 2))))


def test_shape_mismatch():
    payload = _npz(
        intensity=np.ones((4, 8)),
        dark=np.zeros(7),
        flat=np.ones(8),
    )
    with pytest.raises(ValidationError):
        load_npz(payload)


def test_non_finite_rejected():
    intensity = np.ones((4, 8))
    intensity[0, 0] = np.nan
    payload = _npz(
        intensity=intensity, dark=np.zeros(8), flat=np.ones(8))
    with pytest.raises(ValidationError):
        load_npz(payload)


def test_object_array_rejected():
    payload = _npz(
        intensity=np.array([["a", "b"]], dtype=object),
        dark=np.array(["c", "d"], dtype=object),
        flat=np.array(["e", "f"], dtype=object),
    )
    with pytest.raises(ValidationError):
        load_npz(payload)


def test_limit_violations():
    too_many_angles = _good_payload(n_angles=MAX_DETECTORS + 1, n_det=8)
    with pytest.raises(ValidationError):
        load_npz(too_many_angles)
    too_wide = _good_payload(n_angles=4, n_det=MAX_DETECTORS + 1)
    with pytest.raises(ValidationError):
        load_npz(too_wide)


def test_not_a_zip():
    with pytest.raises(ValidationError):
        load_npz(b"this is definitely not a zip")


def test_uncompressed_size_limit():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("intensity.npy", b"x" * (64 * 1024 * 1024 + 1))
    with pytest.raises(ValidationError):
        load_npz(buf.getvalue())


@pytest.mark.parametrize("kwargs", [
    dict(detector_spacing=0, center_index=0, output_size=8,
         pixel_spacing=1, filter_name="ram-lak"),
    dict(detector_spacing=1, center_index=float("nan"), output_size=8,
         pixel_spacing=1, filter_name="ram-lak"),
    dict(detector_spacing=1, center_index=0, output_size=MAX_OUTPUT_SIZE + 1,
         pixel_spacing=1, filter_name="ram-lak"),
    dict(detector_spacing=1, center_index=0, output_size=8,
         pixel_spacing=-1, filter_name="ram-lak"),
    dict(detector_spacing=1, center_index=0, output_size=8,
         pixel_spacing=1, filter_name="shepp"),
])
def test_bad_params(kwargs):
    with pytest.raises(ValidationError):
        validate_params(**kwargs)


def test_good_params():
    clean = validate_params(1.0, 3.5, 8, 0.5, "hann")
    assert clean["output_size"] == 8
