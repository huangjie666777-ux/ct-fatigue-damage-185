"""Dark/flat calibration: raw intensities -> line-integral sinogram."""

from __future__ import annotations

import numpy as np

from .io_utils import LoadedData, ValidationError


def calibrate(data: LoadedData) -> np.ndarray:
    """Return line integrals ``p = -ln((I - dark) / (flat - dark))``.

    ``flat`` must be strictly greater than ``dark`` at every detector and the
    resulting transmission must be strictly positive. Transmissions above 1
    are kept, producing *negative* line integrals; nothing is clipped.
    """
    gain = data.flat - data.dark
    if np.any(gain <= 0.0):
        raise ValidationError("flat field must be strictly greater than dark field at every detector")

    transmission = (data.intensity - data.dark[np.newaxis, :]) / gain[np.newaxis, :]
    if np.any(~np.isfinite(transmission)) or np.any(transmission <= 0.0):
        raise ValidationError("transmission must be finite and strictly positive")

    return -np.log(transmission)
