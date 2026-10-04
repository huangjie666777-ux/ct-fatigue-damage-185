import numpy as np
import pytest

from ctrecon.calibration import calibrate
from ctrecon.io_utils import LoadedData, ValidationError


def test_basic_log_conversion():
    dark = np.array([100.0, 100.0])
    flat = np.array([500.0, 500.0])
    intensity = np.array([[300.0, 500.0]])
    sino = calibrate(LoadedData(intensity, dark, flat))
    assert sino.shape == (1, 2)
    assert sino[0, 0] == pytest.approx(np.log(2.0))
    assert sino[0, 1] == pytest.approx(0.0)


def test_transmission_above_one_keeps_negative():
    dark = np.array([0.0])
    flat = np.array([100.0])
    intensity = np.array([[200.0]])
    sino = calibrate(LoadedData(intensity, dark, flat))
    assert sino[0, 0] == pytest.approx(-np.log(2.0))
    assert sino[0, 0] < 0.0


def test_flat_not_above_dark_rejected():
    dark = np.array([10.0, 10.0])
    flat = np.array([10.0, 20.0])
    intensity = np.zeros((2, 2))
    with pytest.raises(ValidationError):
        calibrate(LoadedData(intensity, dark, flat))


def test_nonpositive_transmission_rejected():
    dark = np.array([5.0])
    flat = np.array([10.0])
    intensity = np.array([[4.0]])
    with pytest.raises(ValidationError):
        calibrate(LoadedData(intensity, dark, flat))


def test_per_detector_gain():
    dark = np.array([10.0, 20.0])
    flat = np.array([110.0, 220.0])
    intensity = np.array([[60.0, 120.0]])
    sino = calibrate(LoadedData(intensity, dark, flat))
    assert np.allclose(sino, np.log(2.0))
