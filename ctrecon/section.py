"""Composite beam cross-section check under axial force and biaxial bending.

Assumptions: linear elasticity, full bond between materials, plane sections
remain plane. Strain over the section is

    eps(x, y) = eps0 + kx * y - ky * x

and the effective stress at a pixel is the volume-fraction weighted sum
sum_m phi_m * E_m * eps. Resultants follow the sign convention

    N  = integral sigma dA
    Mx = integral  y * sigma dA
    My = integral -x * sigma dA

with x to the right, y up, origin at the image center, sampled at pixel
centers. The 3x3 stiffness system is solved jointly; coupling terms are
never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_utils import ValidationError

MAX_LOAD_CASES = 8
MAX_STIFFNESS_COND = 1e12


@dataclass(frozen=True)
class SectionMaterial:
    name: str
    reference_density: float      # mg/mm^3
    elastic_modulus: float        # MPa (N/mm^2)
    allowable_tension: float      # MPa
    allowable_compression: float  # MPa (positive magnitude)


@dataclass(frozen=True)
class LoadCase:
    name: str
    axial: float   # N, tension positive
    mx: float      # N*mm
    my: float      # N*mm


def _positive_finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValidationError(f"{field} must be finite and strictly positive")
    return result


def validate_section_materials(spec: object) -> list[SectionMaterial]:
    """Validate the two materials with mechanical properties."""
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValidationError("materials must be a list of two material objects")
    materials: list[SectionMaterial] = []
    names: set[str] = set()
    for entry in spec:
        if not isinstance(entry, dict):
            raise ValidationError("each material must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("each material needs a non-empty name")
        name = name.strip()
        if name in names:
            raise ValidationError(f"duplicate material name {name!r}")
        names.add(name)
        try:
            material = SectionMaterial(
                name=name,
                reference_density=_positive_finite(
                    entry["reference_density_mg_per_mm3"],
                    f"material {name!r} reference_density_mg_per_mm3",
                ),
                elastic_modulus=_positive_finite(
                    entry["elastic_modulus_mpa"],
                    f"material {name!r} elastic_modulus_mpa",
                ),
                allowable_tension=_positive_finite(
                    entry["allowable_tension_mpa"],
                    f"material {name!r} allowable_tension_mpa",
                ),
                allowable_compression=_positive_finite(
                    entry["allowable_compression_mpa"],
                    f"material {name!r} allowable_compression_mpa",
                ),
            )
        except KeyError as exc:
            raise ValidationError(f"material {name!r} is missing key {exc}") from exc
        materials.append(material)
    return materials


def validate_load_cases(spec: object) -> list[LoadCase]:
    """Validate 1-8 uniquely named load cases with finite resultants."""
    if not isinstance(spec, (list, tuple)) or not (1 <= len(spec) <= MAX_LOAD_CASES):
        raise ValidationError(
            f"load_cases must be a list of 1 to {MAX_LOAD_CASES} cases"
        )
    cases: list[LoadCase] = []
    names: set[str] = set()
    for entry in spec:
        if not isinstance(entry, dict):
            raise ValidationError("each load case must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("each load case needs a non-empty name")
        name = name.strip()
        if name in names:
            raise ValidationError(f"duplicate load case name {name!r}")
        names.add(name)
        values = []
        for key in ("N", "Mx", "My"):
            if key not in entry:
                raise ValidationError(f"load case {name!r} is missing key {key!r}")
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"load case {name!r} {key} must be a number")
            value = float(value)
            if not np.isfinite(value):
                raise ValidationError(f"load case {name!r} {key} must be finite")
            values.append(value)
        cases.append(LoadCase(name=name, axial=values[0], mx=values[1], my=values[2]))
    return cases


def validate_mask(mask: object, output_size: int) -> np.ndarray:
    """Validate the boolean section mask; must match the image size."""
    arr = np.asarray(mask)
    if arr.dtype != np.bool_:
        raise ValidationError("mask array must be boolean")
    if arr.shape != (output_size, output_size):
        raise ValidationError(
            f"mask shape {arr.shape} does not match the {output_size}x{output_size} "
            "reconstruction"
        )
    if not arr.any():
        raise ValidationError("mask selects no pixels")
    return arr


def volume_fractions(
    densities: dict[str, np.ndarray],
    materials: list[SectionMaterial],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Partial densities -> volume fractions, normalized where the sum > 1.

    Where fractions sum to less than 1 the remainder is void and kept as
    such. Pixels outside the mask contribute nothing. The input density
    maps are not modified.
    """
    fractions = {}
    total = np.zeros_like(mask, dtype=np.float64)
    for material in materials:
        phi = densities[material.name] / material.reference_density
        phi = np.where(mask, np.maximum(phi, 0.0), 0.0)
        fractions[material.name] = phi
        total += phi
    over = total > 1.0
    if over.any():
        for name in fractions:
            fractions[name][over] /= total[over]
    return fractions


def pixel_coordinates(
    output_size: int, pixel_spacing_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-center coordinates (mm): origin at image center, x right, y up."""
    center = (output_size - 1) / 2.0
    x = (np.arange(output_size) - center) * pixel_spacing_mm
    y = (center - np.arange(output_size)) * pixel_spacing_mm
    return np.meshgrid(x, y)  # (x_map, y_map), row 0 is +y


def section_stiffness(
    fractions: dict[str, np.ndarray],
    materials: list[SectionMaterial],
    x_map: np.ndarray,
    y_map: np.ndarray,
    pixel_area_mm2: float,
) -> np.ndarray:
    """Assemble the 3x3 section stiffness for [eps0, kx, ky] -> [N, Mx, My]."""
    weight = np.zeros_like(x_map)
    for material in materials:
        weight += fractions[material.name] * material.elastic_modulus
    basis = np.stack(
        [np.ones_like(x_map), y_map, -x_map]
    )  # rows: eps0, kx (times y), ky (times -x)
    stiffness = pixel_area_mm2 * np.einsum(
        "xy,ixy,jxy->ij", weight, basis, basis
    )
    cond = float(np.linalg.cond(stiffness))
    if not np.isfinite(cond) or cond > MAX_STIFFNESS_COND:
        raise ValidationError(
            "section is singular: masked material distribution cannot equilibrate "
            "axial force and biaxial bending"
        )
    return stiffness


def check_load_case(
    case: LoadCase,
    fractions: dict[str, np.ndarray],
    materials: list[SectionMaterial],
    stiffness: np.ndarray,
    x_map: np.ndarray,
    y_map: np.ndarray,
) -> dict:
    """Solve one load case and evaluate per-material allowables."""
    load = np.array([case.axial, case.mx, case.my], dtype=np.float64)
    solution = np.linalg.solve(stiffness, load)
    eps0, kx, ky = (float(v) for v in solution)
    strain = eps0 + kx * y_map - ky * x_map
    residual = stiffness @ solution - load

    per_material = {}
    passed = True
    ratio_map = np.full(strain.shape, np.nan)
    for material in materials:
        present = fractions[material.name] > 0.0
        stress = material.elastic_modulus * strain
        map_out = np.where(present, stress, np.nan)
        if present.any():
            values = stress[present]
            tension = float(values.max())
            compression = float(values.min())
            ratio_t = tension / material.allowable_tension if tension > 0.0 else 0.0
            ratio_c = (
                -compression / material.allowable_compression
                if compression < 0.0
                else 0.0
            )
            ratio = max(ratio_t, ratio_c)
            utilization = np.where(
                stress >= 0.0,
                stress / material.allowable_tension,
                -stress / material.allowable_compression,
            )
            current = np.where(present, utilization, np.nan)
            ratio_map = np.fmax(np.nan_to_num(ratio_map, nan=-np.inf), np.nan_to_num(current, nan=-np.inf))
            if ratio == ratio_t and ratio_t > 0.0:
                idx = int(np.argmax(values))
                mode = "tension"
            else:
                idx = int(np.argmin(values))
                mode = "compression"
            rows, cols = np.nonzero(present)
            row, col = int(rows[idx]), int(cols[idx])
            material_pass = ratio <= 1.0
            passed = passed and material_pass
            per_material[material.name] = {
                "max_tension_mpa": tension,
                "max_compression_mpa": compression,
                "utilization": float(ratio),
                "governing": mode,
                "worst_pixel": {"row": row, "col": col},
                "worst_position_mm": {
                    "x": float(x_map[row, col]),
                    "y": float(y_map[row, col]),
                },
                "passed": bool(material_pass),
            }
        else:
            per_material[material.name] = {
                "max_tension_mpa": None,
                "max_compression_mpa": None,
                "utilization": None,
                "governing": None,
                "worst_pixel": None,
                "worst_position_mm": None,
                "passed": True,
                "note": "material absent inside the mask",
            }
        per_material[material.name]["_stress_map"] = map_out

    return {
        "name": case.name,
        "load": {"N": case.axial, "Mx": case.mx, "My": case.my},
        "strain_curvature": {"eps0": eps0, "kx_per_mm": kx, "ky_per_mm": ky},
        "equilibrium_residual": {
            "N": float(residual[0]),
            "Mx": float(residual[1]),
            "My": float(residual[2]),
        },
        "materials": per_material,
        "passed": bool(passed),
        "_ratio_map": np.where(np.isfinite(ratio_map), ratio_map, np.nan),
    }
