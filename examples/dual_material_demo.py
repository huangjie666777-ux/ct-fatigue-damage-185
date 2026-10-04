"""Generate aligned low/high energy NPZ scans of an analytic two-material phantom.

Phantom: two overlapping uniform disks of different materials with known
partial densities (mg/mm^3). Each material's analytic projection is the disk
chord-length formula scaled by its density; the energy-e line integral is the
linear combination sum_m mu_matrix[e, m] * projection_m (linear two-material
assumption, no beam-hardening).

Run: .venv/bin/python examples/dual_material_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctrecon.synthetic import acquire, disk_sinogram

OUT_DIR = Path(__file__).resolve().parent

GEOMETRY = dict(
    n_angles=180,
    n_det=256,
    detector_spacing_mm=0.5,
    center_index=127.5,
    output_size=128,
    pixel_spacing_mm=0.5,
    slice_thickness_mm=1.0,
)

MATERIALS = ["aluminum", "plastic"]
# Rows: low/high energy; columns: materials. Units: mm^2/mg.
MU_MATRIX = [[0.06, 0.02], [0.03, 0.018]]
# (center_mm, radius_mm, density mg/mm^3) per material disk.
DISKS = [
    ((6.0, -4.0), 12.0, 2.7),    # aluminum
    ((-8.0, 5.0), 9.0, 1.2),     # plastic
]

ROIS = [
    {"name": "al_disk", "x0": 48, "y0": 44, "x1": 102, "y1": 98},
    {"name": "plastic_disk", "x0": 26, "y0": 32, "x1": 68, "y1": 82},
    {"name": "background", "x0": 100, "y0": 100, "x1": 120, "y1": 120},
]


def main() -> None:
    g = GEOMETRY
    material_sinograms = [
        disk_sinogram(
            n_angles=g["n_angles"],
            n_det=g["n_det"],
            detector_spacing=g["detector_spacing_mm"],
            center_index=g["center_index"],
            disk_center_mm=center,
            radius_mm=radius,
            attenuation=density,
        )
        for center, radius, density in DISKS
    ]
    mu = np.asarray(MU_MATRIX)
    stacked = np.stack(material_sinograms)  # (material, angle, detector)
    line_integrals = np.einsum("em,mad->ead", mu, stacked)  # (energy, ...)

    rng = np.random.default_rng(0)
    for label, integrals in zip(("low", "high"), line_integrals):
        intensity, dark, flat = acquire(integrals, rng=rng)
        path = OUT_DIR / f"dual_material_{label}.npz"
        np.savez(path, intensity=intensity, dark=dark, flat=flat)
        print("wrote", path)

    print(
        "curl -s -X POST http://127.0.0.1:8000/decompose \
"
        "  -F 'low_file=@examples/dual_material_low.npz' \
"
        "  -F 'high_file=@examples/dual_material_high.npz' \
"
        f"  -F 'detector_spacing_mm={g['detector_spacing_mm']}' \
"
        f"  -F 'center_index={g['center_index']}' \
"
        f"  -F 'output_size={g['output_size']}' \
"
        f"  -F 'pixel_spacing_mm={g['pixel_spacing_mm']}' \
"
        "  -F 'filter=hann' \
"
        f"  -F 'materials={json.dumps(MATERIALS)}' \
"
        f"  -F 'mu_matrix={json.dumps(MU_MATRIX)}' \
"
        f"  -F 'slice_thickness_mm={g['slice_thickness_mm']}' \
"
        f"  -F 'rois={json.dumps(ROIS)}' \
"
        "  -o decomposition.zip"
    )


if __name__ == "__main__":
    main()
