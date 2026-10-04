"""Generate inputs and print curl commands for POST /section_check.

Phantom: an eccentric two-material section (off-center aluminum disk and
plastic disk inside a circular mask). The script writes the low/high NPZ
scans (reusing the dual-material phantom) plus a boolean mask NPZ, then
prints a ready-to-run curl command.

Run: .venv/bin/python examples/section_check_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.dual_material_demo import GEOMETRY, MU_MATRIX, acquire, disk_sinogram

OUT_DIR = Path(__file__).resolve().parent

MATERIAL_PROPS = [
    {
        "name": "aluminum",
        "reference_density_mg_per_mm3": 2.7,
        "elastic_modulus_mpa": 70000.0,
        "allowable_tension_mpa": 150.0,
        "allowable_compression_mpa": 120.0,
    },
    {
        "name": "plastic",
        "reference_density_mg_per_mm3": 1.2,
        "elastic_modulus_mpa": 3000.0,
        "allowable_tension_mpa": 60.0,
        "allowable_compression_mpa": 80.0,
    },
]
# Eccentric disks: (center_mm, radius_mm, density mg/mm^3).
DISKS = [
    ((10.0, -6.0), 11.0, 2.7),   # aluminum, off-center
    ((-9.0, 7.0), 8.0, 1.2),     # plastic, off-center the other way
]
LOAD_CASES = [
    {"name": "service", "N": 2.0e3, "Mx": 2.5e4, "My": -1.0e4},
    {"name": "overload", "N": -5.0e6, "Mx": 1.0e7, "My": 1.0e7},
]


def main() -> None:
    g = GEOMETRY
    stacked = np.stack(
        [
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
    )
    mu = np.asarray(MU_MATRIX)
    line_integrals = np.einsum("em,mad->ead", mu, stacked)
    rng = np.random.default_rng(1)
    for label, integrals in zip(("low", "high"), line_integrals):
        intensity, dark, flat = acquire(integrals, rng=rng)
        path = OUT_DIR / f"section_{label}.npz"
        np.savez(path, intensity=intensity, dark=dark, flat=flat)
        print("wrote", path)

    n = g["output_size"]
    yy, xx = np.mgrid[0:n, 0:n]
    center = (n - 1) / 2.0
    mask = (yy - center) ** 2 + (xx - center) ** 2 <= 30.0**2
    mask_path = OUT_DIR / "section_mask.npz"
    np.savez(mask_path, mask=mask)
    print("wrote", mask_path)

    curl = [
        "curl -s -o section_check.zip",
        "  -F low_file=@examples/section_low.npz",
        "  -F high_file=@examples/section_high.npz",
        "  -F mask_file=@examples/section_mask.npz",
        f"  -F detector_spacing_mm={g['detector_spacing_mm']}",
        f"  -F center_index={g['center_index']}",
        f"  -F output_size={g['output_size']}",
        f"  -F pixel_spacing_mm={g['pixel_spacing_mm']}",
        "  -F filter=hann",
        "  -F materials=" + json.dumps(json.dumps(MATERIAL_PROPS)),
        "  -F mu_matrix=" + json.dumps(json.dumps(MU_MATRIX)),
        "  -F load_cases=" + json.dumps(json.dumps(LOAD_CASES)),
        "  http://127.0.0.1:8000/section_check",
    ]
    print("\nstart the server:\n")
    print("  .venv/bin/python -m uvicorn ctrecon.app:app --host 127.0.0.1 --port 8000")
    print("\nthen run:\n")
    print(" " + "\\\n".join(line.strip() for line in curl))


if __name__ == "__main__":
    main()
