"""Fatigue pipeline built on the existing CT section mechanics."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .dual_service import _parse_json_field, reconstruct_densities
from .fatigue import (
    MAX_FATIGUE_GRID,
    LoadHistory,
    miner_damage,
    rainflow_cycles,
    section_stress_history,
    validate_fatigue_materials,
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
    damage_maps: dict[str, np.ndarray]
    combined_map: np.ndarray
    mask: np.ndarray
    params: dict
    report: dict


def _cycle_entry(cycle):
    return {
        "stress_amplitude_mpa": cycle.amplitude,
        "mean_stress_mpa": cycle.mean,
        "goodman_amplitude_mpa": cycle.corrected_amplitude,
        "count": cycle.count,
        "cycles_to_failure": cycle.life,
        "damage": cycle.damage,
    }


def fatigue_pipeline(
    low_data: LoadedData,
    high_data: LoadedData,
    mask: np.ndarray,
    history: LoadHistory,
    geometry: dict,
    materials_raw: str,
    mu_matrix_raw: str,
    repetitions: int,
    source_hashes: dict[str, str],
) -> FatigueResult:
    materials = validate_fatigue_materials(_parse_json_field(materials_raw, "materials"))
    names = tuple(material.name for material in materials)
    _, densities, params, _, _ = reconstruct_densities(
        low_data, high_data, geometry, json.dumps(list(names)), mu_matrix_raw
    )
    output_size = params["output_size"]
    if output_size > MAX_FATIGUE_GRID:
        raise ValidationError(f"fatigue grid output_size must be <= {MAX_FATIGUE_GRID}, got {output_size}")

    mask = validate_mask(mask, output_size)
    pixel_spacing = params["pixel_spacing_mm"]
    pixel_area = pixel_spacing * pixel_spacing
    fractions = volume_fractions(densities, materials, mask)
    x_map, y_map = pixel_coordinates(output_size, pixel_spacing)
    stiffness = section_stiffness(fractions, materials, x_map, y_map, pixel_area)
    _, first_stress, second_stress = section_stress_history(
        history, materials, stiffness, x_map, y_map
    )
    stress_histories = {
        materials[0].name: first_stress,
        materials[1].name: second_stress,
    }

    damage_maps = {}
    block_maps = {}
    worst = None
    for material in materials:
        present = fractions[material.name] > 0.0
        block_map = np.full((output_size, output_size), np.nan, dtype=np.float64)
        rows, cols = np.nonzero(present)
        for row, col in zip(rows.tolist(), cols.tolist()):
            cycles = rainflow_cycles(stress_histories[material.name][:, row, col])
            block_damage, details = miner_damage(cycles, material)
            cumulative_damage = repetitions * block_damage
            block_map[row, col] = block_damage
            candidate = (
                cumulative_damage,
                row,
                col,
                material,
                details,
                block_damage,
            )
            if worst is None or candidate[0] > worst[0]:
                worst = candidate
        block_maps[material.name] = block_map
        damage_maps[material.name] = repetitions * block_map

    finite_values = [value for value in block_maps.values() if np.isfinite(value).any()]
    if finite_values:
        combined_block = np.fmax.reduce([np.nan_to_num(v, nan=-np.inf) for v in finite_values])
        combined_block = np.where(np.isfinite(combined_block), combined_block, np.nan)
    else:
        combined_block = np.full((output_size, output_size), np.nan)
    combined_block = np.where(mask, combined_block, np.nan)
    combined_map = repetitions * combined_block

    if worst is None:
        raise ValidationError("mask selects pixels but neither material is present")
    cumulative_damage, row, col, material, details, block_damage = worst
    life_per_blocks = 1.0 / block_damage if block_damage > 0.0 else None
    cumulative_life = None
    if life_per_blocks is not None:
        cumulative_life = life_per_blocks / repetitions
    report = {
        "parameters": {
            **params,
            "repetitions": repetitions,
            "history_points": int(history.axial.size),
            "materials": [
                {
                    "name": m.name,
                    "reference_density_mg_per_mm3": m.reference_density,
                    "elastic_modulus_mpa": m.elastic_modulus,
                    "allowable_tension_mpa": m.allowable_tension,
                    "allowable_compression_mpa": m.allowable_compression,
                    "Sref": m.s_ref,
                    "Nref": m.n_ref,
                    "m": m.sn_slope,
                    "Su": m.ultimate,
                }
                for m in materials
            ],
        },
        "source_hashes": source_hashes,
        "section": {
            "masked_pixels": int(mask.sum()),
            "pixel_area_mm2": float(pixel_area),
            "stiffness_matrix": stiffness.tolist(),
        },
        "maximum_cumulative_damage": float(cumulative_damage),
        "maximum_damage_pixel": {
            "material": material.name,
            "row": int(row),
            "col": int(col),
            "position_mm": {"x": float(x_map[row, col]), "y": float(y_map[row, col])},
            "single_block_damage": float(block_damage),
        },
        "total_block_life": life_per_blocks,
        "total_block_life_note": (
            "null: single-block damage at every counted material pixel is zero"
            if life_per_blocks is None
            else "independent repetitions; blocks to failure for the supplied block"
        ),
        "cumulative_life_in_repeated_blocks": cumulative_life,
        "cycles_at_maximum_pixel": [_cycle_entry(cycle) for cycle in details],
        "zero_damage": life_per_blocks is None,
        "units": {
            "stress": "MPa",
            "force": "N",
            "moment": "N*mm",
            "position": "mm",
            "damage": "Miner dimensionless fraction",
        },
        "assumptions": (
            "same bonded linear-elastic composite section and coupled 3x3 stiffness as /section_check; "
            "the complete synchronized N/Mx/My load vector creates one common strain history; "
            "ASTM E1049 rainflow counting is applied per pixel/material to stress, not to load components; "
            "closed cycles count 1 and residual reversals count 0.5; repeated blocks are independent"
        ),
    }
    return FatigueResult(damage_maps, combined_map, mask, params, report)
