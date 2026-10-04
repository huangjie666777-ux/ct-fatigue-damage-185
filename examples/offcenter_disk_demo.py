"""Generate an off-center disk NPZ and print ready-to-use curl commands.

Run: .venv/bin/python examples/offcenter_disk_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctrecon.synthetic import acquire, disk_sinogram


def main() -> None:
    n_angles, n_det = 180, 256
    detector_spacing = 0.5
    center_index = 127.5
    pixel_spacing = 0.5
    output_size = 128

    truth = disk_sinogram(
        n_angles=n_angles,
        n_det=n_det,
        detector_spacing=detector_spacing,
        center_index=center_index,
        disk_center_mm=(8.0, -5.0),
        radius_mm=12.0,
        attenuation=0.2,
    )
    rng = np.random.default_rng(0)
    intensity, dark, flat = acquire(truth, rng=rng)

    out_dir = Path(__file__).resolve().parent
    np.savez(
        out_dir / "offcenter_disk.npz",
        intensity=intensity,
        dark=dark,
        flat=flat,
    )
    print("wrote", out_dir / "offcenter_disk.npz")
    print(
        "curl -s -X POST http://127.0.0.1:8000/reconstruct \\\n"
        "  -F 'file=@examples/offcenter_disk.npz' \\\n"
        f"  -F 'detector_spacing_mm={detector_spacing}' \\\n"
        f"  -F 'center_index={center_index}' \\\n"
        f"  -F 'output_size={output_size}' \\\n"
        f"  -F 'pixel_spacing_mm={pixel_spacing}' \\\n"
        "  -F 'filter=hann' -o reconstruction.zip"
    )


if __name__ == "__main__":
    main()
