"""Generate a 64x64 fatigue example and print a curl command for /fatigue."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctrecon.synthetic import acquire, disk_sinogram

OUT_DIR = Path(__file__).resolve().parent

GEOMETRY = {
    "n_angles": 180,
    "n_det": 256,
    "detector_spacing_mm": 0.5,
    "center_index": 127.5,
    "output_size": 64,
    "pixel_spacing_mm": 0.5,
}
MU_MATRIX = [[0.06, 0.02], [0.03, 0.018]]
DISKS = [((6.0, -4.0), 12.0, 2.7), ((-8.0, 5.0), 9.0, 1.2)]
MATERIALS = [
    {
        "name": "aluminum",
        "reference_density_mg_per_mm3": 2.7,
        "elastic_modulus_mpa": 70000.0,
        "allowable_tension_mpa": 150.0,
        "allowable_compression_mpa": 120.0,
        "Sref": 200.0,
        "Nref": 1.0e6,
        "m": 5.0,
        "Su": 300.0,
    },
    {
        "name": "plastic",
        "reference_density_mg_per_mm3": 1.2,
        "elastic_modulus_mpa": 3000.0,
        "allowable_tension_mpa": 60.0,
        "allowable_compression_mpa": 80.0,
        "Sref": 40.0,
        "Nref": 1.0e5,
        "m": 4.0,
        "Su": 80.0,
    },
]


def main() -> None:
    sinograms = np.stack([
        disk_sinogram(
            GEOMETRY["n_angles"],
            GEOMETRY["n_det"],
            GEOMETRY["detector_spacing_mm"],
            GEOMETRY["center_index"],
            center,
            radius,
            density,
        )
        for center, radius, density in DISKS
    ])
    projections = np.einsum("em,mad->ead", np.asarray(MU_MATRIX), sinograms)
    rng = np.random.default_rng(2)
    for label, projection in zip(("low", "high"), projections):
        intensity, dark, flat = acquire(projection, rng=rng)
        path = OUT_DIR / f"fatigue_{label}.npz"
        np.savez(path, intensity=intensity, dark=dark, flat=flat)
        print("wrote", path)

    size = GEOMETRY["output_size"]
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    mask = (yy - center) ** 2 + (xx - center) ** 2 <= 24.0**2
    mask_path = OUT_DIR / "fatigue_mask.npz"
    np.savez(mask_path, mask=mask)
    print("wrote", mask_path)

    time = np.linspace(0.0, 2.0 * np.pi, 33, endpoint=True)
    history_path = OUT_DIR / "fatigue_history.npz"
    np.savez(
        history_path,
        N=1.0e3 * np.sin(time),
        Mx=5.0e3 * np.cos(time),
        My=-3.0e3 * np.sin(2.0 * time),
    )
    print("wrote", history_path)

    curl = [
        "curl -s -o fatigue.zip",
        "  -F low_file=@examples/fatigue_low.npz",
        "  -F high_file=@examples/fatigue_high.npz",
        "  -F mask_file=@examples/fatigue_mask.npz",
        "  -F history_file=@examples/fatigue_history.npz",
        f"  -F detector_spacing_mm={GEOMETRY['detector_spacing_mm']}",
        f"  -F center_index={GEOMETRY['center_index']}",
        f"  -F output_size={GEOMETRY['output_size']}",
        f"  -F pixel_spacing_mm={GEOMETRY['pixel_spacing_mm']}",
        "  -F filter=hann",
        "  -F materials=" + json.dumps(json.dumps(MATERIALS)),
        "  -F mu_matrix=" + json.dumps(json.dumps(MU_MATRIX)),
        "  -F R=10",
        "  http://127.0.0.1:8000/fatigue",
    ]
    print("\nstart the server:\n")
    print("  .venv/bin/python -m uvicorn ctrecon.app:app --host 127.0.0.1 --port 8000")
    print("\nthen run:\n")
    print(" " + "\\\n".join(line.strip() for line in curl))


if __name__ == "__main__":
    main()
