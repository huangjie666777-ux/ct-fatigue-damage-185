"""Fatigue-analysis pipeline.

Reuses the dual-energy material decomposition, volume fractions and the
coupled 3x3 section stiffness. For each ordered history point (N, Mx, My)
the strain state is solved jointly, then the per-material stress history
is reconstructed at every pixel where that material exists. The combined
stress history goes through a single ASTM E1049 rainflow count; load
components are never counted separately.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .dual_service import _parse_json_field, reconstruct_densities
from .fatigue import (
    FATIGUE_MAX_OUTPUT_SIZE,
    block_damage,
    validate_fatigue_materials,
    validate_history,
    validate_repeat_count,
)
from .io_utils import LoadedData, ValidationError
from .section import (
    pixel_coordinates,
    section_stiffness,
    validate_mask,
    volume_fractions,
)


@dataclass(frozen=True)
class FatigueResult:
    damage_maps: dict[str, np.ndarray]  # accumulated damage R*D per material
    block_maps: dict[str, np.ndarray]   # single-block damage per material
    mask: np.ndarray
    params: dict
    report: dict


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fatigue_pipeline(
    low_data: LoadedData,
    high_data: LoadedData,
    mask: np.ndarray,
    geometry: dict,
    materials_raw: str,
    mu_matrix_raw: str,
    history_raw: str,
    repeat_count: int,
    low_payload: bytes,
    high_payload: bytes,
    mask_payload: bytes,
) -> FatigueResult:
    materials = validate_fatigue_materials(
        _parse_json_field(materials_raw, "materials")
    )
    history = validate_history(_parse_json_field(history_raw, "history"))
    repeats = validate_repeat_count(repeat_count)
    names = (material.base.name for material in materials)
    _, densities, params, _, _ = reconstruct_densities(
        low_data, high_data, geometry, json.dumps(list(names)), mu_matrix_raw
    )

    output_size = params["output_size"]
    if output_size > FATIGUE_MAX_OUTPUT_SIZE:
        raise ValidationError(
            f"fatigue analysis output_size must be <= {FATIGUE_MAX_OUTPUT_SIZE}"
        )
    mask = validate_mask(mask, output_size)
    pixel_spacing = params["pixel_spacing_mm"]
    pixel_area = pixel_spacing * pixel_spacing

    section_materials = [material.base for material in materials]
    fractions = volume_fractions(densities, section_materials, mask)
    if not any((fractions[m.base.name] > 0.0).any() for m in materials):
        raise ValidationError(
            "no material is present (volume fraction > 0) inside the mask"
        )
    x_map, y_map = pixel_coordinates(output_size, pixel_spacing)
    stiffness = section_stiffness(
        fractions, section_materials, x_map, y_map, pixel_area
    )

    # One coupled solve per ordered history point -> strain history field.
    loads = np.array(
        [[point.n_force, point.mx, point.my] for point in history],
        dtype=np.float64,
    )
    solutions = np.linalg.solve(stiffness, loads.T).T  # (T, 3)
    if not np.all(np.isfinite(solutions)):
        raise ValidationError("non-finite strain solution for the load history")
    t_points = len(history)
    strain_history = (
        solutions[:, 0][:, None, None]
        + solutions[:, 1][:, None, None] * y_map[None, :, :]
        - solutions[:, 2][:, None, None] * x_map[None, :, :]
    )

    block_maps: dict[str, np.ndarray] = {}
    damage_maps: dict[str, np.ndarray] = {}
    material_reports: dict[str, dict] = {}
    overall_argmax = None
    overall_max = -1.0

    for material in materials:
        present = fractions[material.base.name] > 0.0
        stress_history = material.base.elastic_modulus * strain_history
        block_map = np.full((output_size, output_size), np.nan, dtype=np.float64)
        rows, cols = np.nonzero(present)
        for row, col in zip(rows.tolist(), cols.tolist()):
            label = (
                f"material {material.base.name!r} pixel (row={row}, col={col})"
            )
            damage, _ = block_damage(
                stress_history[:, row, col], material, position_label=label
            )
            block_map[row, col] = damage
        block_maps[material.base.name] = block_map
        damage_maps[material.base.name] = repeats * block_map

        present_finite = present & np.isfinite(block_map)
        if present_finite.any():
            flat_index = int(np.argmax(np.where(present_finite, block_map, -np.inf)))
            worst_row, worst_col = np.unravel_index(flat_index, block_map.shape)
            worst_row, worst_col = int(worst_row), int(worst_col)
            _, cycles_detail = block_damage(
                stress_history[:, worst_row, worst_col],
                material,
                position_label=f"material {material.base.name!r} worst pixel",
            )
            block_d = float(block_map[worst_row, worst_col])
            accumulated = repeats * block_d
            material_report = {
                "present_pixels": int(present.sum()),
                "max_block_damage": block_d,
                "max_accumulated_damage": accumulated,
                "worst_pixel": {"row": worst_row, "col": worst_col},
                "worst_position_mm": {
                    "x": float(x_map[worst_row, worst_col]),
                    "y": float(y_map[worst_row, worst_col]),
                },
                "blocks_to_failure": (1.0 / block_d) if block_d > 0.0 else None,
                "zero_damage": block_d == 0.0,
                "cycles": cycles_detail,
            }
            if accumulated > overall_max:
                overall_max = accumulated
                overall_argmax = (material.base.name, worst_row, worst_col)
        else:
            material_report = {
                "present_pixels": 0,
                "max_block_damage": None,
                "max_accumulated_damage": None,
                "worst_pixel": None,
                "worst_position_mm": None,
                "blocks_to_failure": None,
                "zero_damage": True,
                "cycles": [],
                "note": "material absent inside the mask",
            }
        material_reports[material.base.name] = material_report

    # Maximum accumulated damage across all materials and pixels.
    max_name, max_row, max_col = overall_argmax
    max_material = next(m for m in materials if m.base.name == max_name)
    max_stress = max_material.base.elastic_modulus * strain_history
    max_block_d = float(block_maps[max_name][max_row, max_col])
    max_accumulated = repeats * max_block_d
    _, max_cycles = block_damage(
        max_stress[:, max_row, max_col],
        max_material,
        position_label=f"material {max_name!r} critical pixel",
    )

    report = {
        "parameters": {
            **params,
            "materials": [
                {
                    "name": m.base.name,
                    "reference_density_mg_per_mm3": m.base.reference_density,
                    "elastic_modulus_mpa": m.base.elastic_modulus,
                    "allowable_tension_mpa": m.base.allowable_tension,
                    "allowable_compression_mpa": m.base.allowable_compression,
                    "Sref_mpa": m.s_ref,
                    "Nref_cycles": m.n_ref,
                    "m": m.sn_exponent,
                    "Su_mpa": m.su,
                }
                for m in materials
            ],
        },
        "history": {
            "point_count": t_points,
            "points": [
                {"N": p.n_force, "Mx": p.mx, "My": p.my} for p in history
            ],
            "block_repeats_R": repeats,
            "independent_block_assumption": (
                "each block repeats the identical ordered history independently; "
                "accumulated damage is R times the single-block Miner damage"
            ),
        },
        "section": {
            "masked_pixels": int(mask.sum()),
            "pixel_area_mm2": float(pixel_area),
            "stiffness_matrix": stiffness.tolist(),
        },
        "materials": material_reports,
        "critical_point": {
            "material": max_name,
            "pixel": {"row": max_row, "col": max_col},
            "position_mm": {
                "x": float(x_map[max_row, max_col]),
                "y": float(y_map[max_row, max_col]),
            },
            "single_block_damage": max_block_d,
            "accumulated_damage": max_accumulated,
            "blocks_to_failure": (1.0 / max_block_d) if max_block_d > 0.0 else None,
            "zero_damage": max_block_d == 0.0,
            "cycles": max_cycles,
        },
        "units": {
            "stress": "MPa (N/mm^2)",
            "force": "N",
            "moment": "N*mm",
            "position": "mm",
        },
        "assumptions": (
            "linear elasticity, full bond, plane sections remain plane; one "
            "coupled stiffness solve per ordered history point; ASTM E1049 "
            "rainflow on the reconstructed combined stress history (load "
            "components are not counted separately); closed cycles count 1, "
            "residual ranges count 0.5, head and tail are not joined; "
            "Goodman mean-stress correction and Miner linear damage"
        ),
        "linkage": {
            "low_file_sha256": _sha256(low_payload),
            "high_file_sha256": _sha256(high_payload),
            "mask_file_sha256": _sha256(mask_payload),
            "history_sha256": _sha256(
                json.dumps(
                    [(p.n_force, p.mx, p.my) for p in history], sort_keys=True
                ).encode()
            ),
        },
    }
    return FatigueResult(
        damage_maps=damage_maps,
        block_maps=block_maps,
        mask=mask,
        params=params,
        report=report,
    )
