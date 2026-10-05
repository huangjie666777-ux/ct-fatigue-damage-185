"""Generate inputs and print a curl command for POST /fatigue_check.

Phantom: the same eccentric two-material section used by the section check
demo, reconstructed on a 48x48 grid (the fatigue endpoint caps the grid at
64x64). The ordered history contains 8 (N, Mx, My) points; the block is
repeated R = 100 times independently.

Run: .venv/bin/python examples/fatigue_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.dual_material_demo import MU_MATRIX, acquire, disk_sinogram

OUT_DIR = Path(__file__).resolve().parent

GEOMETRY = {
    "n_angles": 120,
    "n_det": 128,
    "detector_spacing_mm": 0.5,
    "center_index": 63.5,
    "output_size": 48,
    "pixel_spacing_mm": 0.5,
}

MATERIAL_PROPS = [
    {
        "name": "aluminum",
        "reference_density_mg_per_mm3": 2.7,
        "elastic_modulus_mpa": 70000.0,
        "allowable_tension_mpa": 150.0,
        "allowable_compression_mpa": 120.0,
        "Sref": 80.0,
        "Nref": 1.0e6,
        "m": 4.0,
        "Su": 300.0,
    },
    {
        "name": "plastic",
        "reference_density_mg_per_mm3": 1.2,
        "elastic_modulus_mpa": 3000.0,
        "allowable_tension_mpa": 60.0,
        "allowable_compression_mpa": 80.0,
        "Sref": 30.0,
        "Nref": 1.0e5,
        "m": 5.0,
        "Su": 80.0,
    },
]
# Eccentric disks: (center_mm, radius_mm, density mg/mm^3).
DISKS = [((4.0, -2.0), 6.0, 2.7), ((-4.0, 3.0), 5.0, 1.2)]
HISTORY = [
    {"N": 3000.0, "Mx": 40000.0, "My": -10000.0},
    {"N": -2000.0, "Mx": -30000.0, "My": 15000.0},
    {"N": 1000.0, "Mx": 20000.0, "My": 0.0},
    {"N": 3000.0, "Mx": 40000.0, "My": -10000.0},
    {"N": -4000.0, "Mx": -50000.0, "My": 20000.0},
    {"N": 500.0, "Mx": 10000.0, "My": -5000.0},
    {"N": 3000.0, "Mx": 40000.0, "My": -10000.0},
    {"N": -2000.0, "Mx": -30000.0, "My": 15000.0},
]
R = 100


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
    line_integrals = np.einsum(
        "em,mad->ead", np.asarray(MU_MATRIX), stacked
    )
    rng = np.random.default_rng(7)
    for label, integrals in zip(("low", "high"), line_integrals):
        intensity, dark, flat = acquire(integrals, rng=rng)
        path = OUT_DIR / f"fatigue_{label}.npz"
        np.savez(path, intensity=intensity, dark=dark, flat=flat)
        print("wrote", path)

    n = g["output_size"]
    yy, xx = np.mgrid[0:n, 0:n]
    center = (n - 1) / 2.0
    mask = (yy - center) ** 2 + (xx - center) ** 2 <= 15.0**2
    mask_path = OUT_DIR / "fatigue_mask.npz"
    np.savez(mask_path, mask=mask)
    print("wrote", mask_path)

    curl = [
        "curl -s -o fatigue_check.zip",
        "  -F low_file=@examples/fatigue_low.npz",
        "  -F high_file=@examples/fatigue_high.npz",
        "  -F mask_file=@examples/fatigue_mask.npz",
        f"  -F detector_spacing_mm={g['detector_spacing_mm']}",
        f"  -F center_index={g['center_index']}",
        f"  -F output_size={g['output_size']}",
        f"  -F pixel_spacing_mm={g['pixel_spacing_mm']}",
        "  -F filter=hann",
        "  -F materials=" + json.dumps(json.dumps(MATERIAL_PROPS)),
        "  -F mu_matrix=" + json.dumps(json.dumps(MU_MATRIX)),
        "  -F history=" + json.dumps(json.dumps(HISTORY)),
        f"  -F R={R}",
        "  http://127.0.0.1:8000/fatigue_check",
    ]
    print("\nstart the server:\n")
    print("  .venv/bin/python -m uvicorn ctrecon.app:app --host 127.0.0.1 --port 8000")
    print("\nthen run:\n")
    print(" " + "\\\n".join(line.strip() for line in curl))


if __name__ == "__main__":
    main()
