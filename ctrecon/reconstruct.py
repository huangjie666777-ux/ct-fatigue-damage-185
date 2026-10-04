"""Filtered back-projection implemented directly with NumPy."""

from __future__ import annotations

import numpy as np

FILTERS = ("ram-lak", "hann")


def supported_filters() -> tuple[str, ...]:
    return FILTERS


def _next_fft_length(n: int) -> int:
    size = 1
    # Zero-pad to at least 2n samples so linear convolution is not aliased.
    while size < 2 * n:
        size *= 2
    return size


def _ramp_filter(n_fft: int, detector_spacing: float, name: str) -> np.ndarray:
    """Frequency response sampled on the FFT grid (cycles/mm)."""
    frequencies = np.fft.fftfreq(n_fft, d=detector_spacing)
    nyquist = 0.5 / detector_spacing
    # Response |f| in cycles/mm: the FFT/ifft pair already carries the
    # 1/(N*d) and d factors of the continuous Fourier transform, so no
    # extra detector-spacing factor may be applied to the filtered rows.
    ramp = np.abs(frequencies)
    if name == "hann":
        window = np.where(
            np.abs(frequencies) <= nyquist,
            0.5 * (1.0 + np.cos(np.pi * frequencies / nyquist)),
            0.0,
        )
        ramp = ramp * window
    else:
        # Ram-Lak is zero past the physical detector Nyquist frequency.
        ramp = np.where(np.abs(frequencies) <= nyquist, ramp, 0.0)
    return ramp


def filter_sinogram(sinogram: np.ndarray, detector_spacing: float, name: str) -> np.ndarray:
    """Apply the ramp filter row-wise with FFT zero-padding."""
    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}")
    n_det = sinogram.shape[1]
    n_fft = _next_fft_length(n_det)
    response = _ramp_filter(n_fft, detector_spacing, name)
    spectrum = np.fft.fft(sinogram, n=n_fft, axis=1)
    filtered = np.fft.ifft(spectrum * response[np.newaxis, :], axis=1).real
    return filtered[:, :n_det]


def fbp(
    sinogram: np.ndarray,
    detector_spacing: float,
    center_index: float,
    output_size: int,
    pixel_spacing: float,
    filter_name: str = "ram-lak",
) -> np.ndarray:
    """Parallel-beam FBP reconstruction.

    Coordinate convention: the geometric center of the image is the origin;
    columns point to +x, rows point to +y. At angle 0 the ray normal is +x,
    so detector coordinate ``t = x*cos(theta) + y*sin(theta)`` with
    ``t = (detector_index - center_index) * detector_spacing``.

    Returns an ``output_size x output_size`` array of linear attenuation
    coefficients in mm^-1.
    """
    sinogram = np.asarray(sinogram, dtype=np.float64)
    if sinogram.ndim != 2:
        raise ValueError("sinogram must be 2-D")
    n_angles, n_det = sinogram.shape

    filtered = filter_sinogram(sinogram, detector_spacing, filter_name)

    side = (output_size - 1) / 2.0
    coords = (np.arange(output_size) - side) * pixel_spacing
    x = coords[np.newaxis, :]                    # x varies with column
    y = coords[::-1][:, np.newaxis]             # row 0 is +y

    angles = np.arange(n_angles, dtype=np.float64) * (np.pi / n_angles)
    angle_step = np.pi / n_angles

    detector_indices = np.arange(n_det, dtype=np.float64)
    reconstruction = np.zeros((output_size, output_size), dtype=np.float64)

    for k, theta in enumerate(angles):
        t = x * np.cos(theta) + y * np.sin(theta)
        sample_index = t / detector_spacing + center_index
        # Linear interpolation; samples outside the detector range are zero.
        back = np.interp(sample_index, detector_indices, filtered[k], left=0.0, right=0.0)
        reconstruction += back

    # Integrate over theta in radians. No per-image normalization is applied.
    return reconstruction * angle_step
