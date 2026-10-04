"""Section-check pipeline: dual-energy densities + mask + loads -> report."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .dual_service import _parse_json_field, reconstruct_densities
from .io_utils import LoadedData
from .section import (
    check_load_case,
    pixel_coordinates,
    section_stiffness,
    validate_load_cases,
    validate_mask,
    validate_section_materials,
    volume_fractions,
)


@dataclass(frozen=True)
class SectionCheckResult:
    fractions: dict[str, np.ndarray]   # material name -> volume fraction map
    case_results: list[dict]           # per load case, incl. _stress_map entries
    stiffness: np.ndarray              # 3x3, [eps0, kx, ky] -> [N, Mx, My]
    mask: np.ndarray
    params: dict
    report: dict


def section_check_pipeline(
    low_data: LoadedData,
    high_data: LoadedData,
    mask: np.ndarray,
    geometry: dict,
    materials_raw: str,
    mu_matrix_raw: str,
    load_cases_raw: str,
) -> SectionCheckResult:
    materials = validate_section_materials(
        _parse_json_field(materials_raw, "materials")
    )
    cases = validate_load_cases(_parse_json_field(load_cases_raw, "load_cases"))
    names = (materials[0].name, materials[1].name)
    _, densities, params, _, _ = reconstruct_densities(
        low_data, high_data, geometry, json.dumps(list(names)), mu_matrix_raw
    )

    output_size = params["output_size"]
    mask = validate_mask(mask, output_size)
    pixel_spacing = params["pixel_spacing_mm"]
    pixel_area = pixel_spacing * pixel_spacing

    fractions = volume_fractions(densities, materials, mask)
    x_map, y_map = pixel_coordinates(output_size, pixel_spacing)
    stiffness = section_stiffness(fractions, materials, x_map, y_map, pixel_area)

    case_results = [
        check_load_case(case, fractions, materials, stiffness, x_map, y_map)
        for case in cases
    ]

    report = {
        "parameters": {
            **params,
            "materials": [
                {
                    "name": m.name,
                    "reference_density_mg_per_mm3": m.reference_density,
                    "elastic_modulus_mpa": m.elastic_modulus,
                    "allowable_tension_mpa": m.allowable_tension,
                    "allowable_compression_mpa": m.allowable_compression,
                }
                for m in materials
            ],
        },
        "section": {
            "masked_pixels": int(mask.sum()),
            "pixel_area_mm2": float(pixel_area),
            "stiffness_matrix": stiffness.tolist(),
            "stiffness_units": "[[N, N*mm, N*mm]; ...] for [eps0, kx(1/mm), ky(1/mm)]",
        },
        "cases": [
            {
                key: value
                for key, value in case_result.items()
                if key != "materials" and not key.startswith("_")
            }
            | {
                "materials": {
                    name: {
                        k: v for k, v in entry.items() if not k.startswith("_")
                    }
                    for name, entry in case_result["materials"].items()
                },
            }
            for case_result in case_results
        ],
        "passed": all(case_result["passed"] for case_result in case_results),
        "units": {
            "stress": "MPa (N/mm^2)",
            "force": "N",
            "moment": "N*mm",
            "position": "mm",
        },
        "assumptions": (
            "linear elasticity, full bond, plane sections remain plane; "
            "volume fractions normalized where their sum exceeds 1, void kept "
            "where below 1; pixels outside the mask do not participate"
        ),
    }
    return SectionCheckResult(
        fractions=fractions,
        case_results=case_results,
        stiffness=stiffness,
        mask=mask,
        params=params,
        report=report,
    )
