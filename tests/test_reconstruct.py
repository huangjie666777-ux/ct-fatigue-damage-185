import numpy as np
import pytest

from ctrecon.calibration import calibrate
from ctrecon.io_utils import LoadedData
from ctrecon.reconstruct import fbp
from ctrecon.synthetic import acquire, disk_sinogram


@pytest.mark.parametrize("name", ["ram-lak", "hann"])
def test_centered_disk_recovers_attenuation(name):
    spacing = 0.5
    n_angles, n_det = 180, 256
    center = 127.5
    truth = disk_sinogram(
        n_angles, n_det, spacing, center,
        disk_center_mm=(0.0, 0.0), radius_mm=10.0, attenuation=0.2,
    )
    image = fbp(truth, spacing, center, 128, spacing, filter_name=name)
    side = (128 - 1) / 2
    col, row = np.meshgrid(np.arange(128), np.arange(128))
    x = (col - side) * spacing
    y = (side - row) * spacing
    interior = (x * x + y * y) <= 6.0**2
    interior_mean = image[interior].mean()
    assert interior_mean == pytest.approx(0.2, abs=0.03)
    assert image.min() < 0.0


def test_offcenter_disk_position_and_orientation():
    spacing = 0.5
    n_angles, n_det = 180, 256
    center = 127.5
    cx, cy = 8.0, -5.0
    truth = disk_sinogram(
        n_angles, n_det, spacing, center,
        disk_center_mm=(cx, cy), radius_mm=10.0, attenuation=0.3,
    )
    n = 128
    image = fbp(truth, spacing, center, n, spacing, filter_name="hann")
    side = (n - 1) / 2
    col, row = np.meshgrid(np.arange(n), np.arange(n))
    mask = image > 0.15
    rows, cols = np.nonzero(mask)
    centroid_x = (cols.mean() - side) * spacing
    centroid_y = (side - rows.mean()) * spacing
    assert centroid_x == pytest.approx(cx, abs=1.0)
    assert centroid_y == pytest.approx(cy, abs=1.0)


def test_reconstruction_units_scale_with_spacing():
    common = dict(n_angles=180, n_det=256, center_index=127.5,
                  disk_center_mm=(0.0, 0.0), radius_mm=10.0, attenuation=0.25)
    truth = disk_sinogram(detector_spacing=0.5, **common)
    image = fbp(truth, 0.5, 127.5, 128, 0.5, "ram-lak")
    center_value = image[64, 64]
    assert center_value == pytest.approx(0.25, abs=0.05)


def test_end_to_end_via_calibration():
    spacing = 0.5
    truth = disk_sinogram(180, 256, spacing, 127.5, (0.0, 0.0), 8.0, 0.15)
    intensity, dark, flat = acquire(truth)
    sino = calibrate(LoadedData(intensity, dark, flat))
    assert np.allclose(sino, truth, atol=1e-9)
    image = fbp(sino, spacing, 127.5, 96, spacing, "ram-lak")
    assert image.shape == (96, 96)


def test_unknown_filter_rejected():
    sino = np.zeros((4, 8))
    with pytest.raises(ValueError):
        fbp(sino, 1.0, 3.5, 8, 1.0, filter_name="cosine")
