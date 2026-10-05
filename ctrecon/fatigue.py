"""Rain-flow fatigue counting and Miner damage for beam-section histories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .io_utils import ValidationError
from .section import (
    SectionMaterial,
    _positive_finite,
)

MAX_HISTORY_POINTS = 2000
MIN_HISTORY_POINTS = 2
MAX_FATIGUE_GRID = 64
MAX_BLOCK_REPETITIONS = 1000000000


@dataclass(frozen=True)
class FatigueMaterial(SectionMaterial):
    s_ref: float
    n_ref: float
    sn_slope: float
    ultimate: float


@dataclass(frozen=True)
class LoadHistory:
    axial: np.ndarray
    mx: np.ndarray
    my: np.ndarray


@dataclass(frozen=True)
class RainflowCycle:
    amplitude: float
    mean: float
    corrected_amplitude: float
    count: float
    life: float
    damage: float


def validate_fatigue_materials(spec: object) -> list[FatigueMaterial]:
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValidationError("materials must be a list of two material objects")
    materials = []
    names = set()
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
            materials.append(
                FatigueMaterial(
                    name=name,
                    reference_density=_positive_finite(entry["reference_density_mg_per_mm3"], f"material {name!r} reference_density_mg_per_mm3"),
                    elastic_modulus=_positive_finite(entry["elastic_modulus_mpa"], f"material {name!r} elastic_modulus_mpa"),
                    allowable_tension=_positive_finite(entry["allowable_tension_mpa"], f"material {name!r} allowable_tension_mpa"),
                    allowable_compression=_positive_finite(entry["allowable_compression_mpa"], f"material {name!r} allowable_compression_mpa"),
                    s_ref=_positive_finite(entry["Sref"], f"material {name!r} Sref"),
                    n_ref=_positive_finite(entry["Nref"], f"material {name!r} Nref"),
                    sn_slope=_positive_finite(entry["m"], f"material {name!r} m"),
                    ultimate=_positive_finite(entry["Su"], f"material {name!r} Su"),
                )
            )
        except KeyError as exc:
            raise ValidationError(f"material {name!r} is missing key {exc}") from exc
    return materials


def validate_repetitions(value: str) -> int:
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise ValidationError("R must be a positive integer string")
    if not value.isascii() or not value.isdigit():
        raise ValidationError("R must be a positive integer")
    result = int(value)
    if not 1 <= result <= MAX_BLOCK_REPETITIONS:
        raise ValidationError(
            f"R must be a positive integer no greater than {MAX_BLOCK_REPETITIONS}"
        )
    return result


def validate_history(arrays: dict[str, np.ndarray]) -> LoadHistory:
    cleaned = []
    for name in ("N", "Mx", "My"):
        if name not in arrays:
            raise ValidationError(f"history NPZ is missing required array '{name}'")
        arr = np.asarray(arrays[name])
        if arr.dtype == np.bool_ or arr.dtype.kind not in "iuf":
            raise ValidationError(f"history array '{name}' must be numeric")
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        if arr.ndim != 1:
            raise ValidationError(f"history array '{name}' must be 1-D")
        if not np.all(np.isfinite(arr)):
            raise ValidationError(f"history array '{name}' contains non-finite values")
        cleaned.append(arr)
    n_points = cleaned[0].size
    if not (MIN_HISTORY_POINTS <= n_points <= MAX_HISTORY_POINTS):
        raise ValidationError(f"history length must be in [{MIN_HISTORY_POINTS}, {MAX_HISTORY_POINTS}], got {n_points}")
    if any(arr.size != n_points for arr in cleaned[1:]):
        raise ValidationError("history arrays N, Mx and My must have equal length")
    return LoadHistory(cleaned[0], cleaned[1], cleaned[2])


def reduce_turning_points(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    keep = [float(values[0])]
    for value in (float(v) for v in values[1:]):
        if value != keep[-1]:
            keep.append(value)
    if len(keep) <= 2:
        return keep
    result = [keep[0]]
    for prev, value, nxt in zip(keep, keep[1:], keep[2:]):
        if (value > prev and value > nxt) or (value < prev and value < nxt):
            result.append(value)
    result.append(keep[-1])
    return result


def rainflow_cycles(values: np.ndarray | Sequence[float]) -> list[tuple[float, float, float]]:
    """Count cycles with the ASTM E1049 three-reversal online algorithm."""
    points = reduce_turning_points(np.asarray(values, dtype=np.float64))
    if len(points) < 2:
        return []
    cycles = []
    pending = [points[0]]
    for point in points[1:]:
        pending.append(point)
        while len(pending) >= 3:
            first, middle, last = pending[-3], pending[-2], pending[-1]
            x_range = abs(last - middle)
            y_range = abs(middle - first)
            if x_range < y_range:
                break
            if len(pending) == 3:
                cycles.append((y_range / 2.0, (first + middle) / 2.0, 0.5))
                pending.pop(0)
            else:
                cycles.append((y_range / 2.0, (first + middle) / 2.0, 1.0))
                end = pending.pop()
                pending.pop()
                pending.pop()
                pending.append(end)
    for left, right in zip(pending, pending[1:]):
        cycles.append((abs(right - left) / 2.0, (left + right) / 2.0, 0.5))
    return cycles


def miner_damage(
    cycles: list[tuple[float, float, float]], material: FatigueMaterial
) -> tuple[float, list[RainflowCycle]]:
    total = 0.0
    details = []
    for amplitude, mean, count in cycles:
        if amplitude <= 0.0:
            continue
        denominator = 1.0 - max(mean, 0.0) / material.ultimate
        if denominator <= 0.0:
            raise ValidationError(
                f"Goodman denominator is non-positive for material {material.name!r} at mean stress {mean:g} MPa and Su {material.ultimate:g} MPa"
            )
        corrected = amplitude / denominator
        life = material.n_ref * (material.s_ref / corrected) ** material.sn_slope
        if not np.isfinite(life) or life <= 0.0:
            raise ValidationError(f"non-finite or non-positive fatigue life for material {material.name!r}")
        damage = count / life
        total += damage
        details.append(RainflowCycle(float(amplitude), float(mean), float(corrected), float(count), float(life), float(damage)))
    if not np.isfinite(total):
        raise ValidationError(f"fatigue damage is non-finite for material {material.name!r}")
    return float(total), details


def section_stress_history(
    history: LoadHistory,
    materials: list[FatigueMaterial],
    stiffness: np.ndarray,
    x_map: np.ndarray,
    y_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loads = np.stack([history.axial, history.mx, history.my], axis=0)
    solution = np.linalg.solve(stiffness, loads)
    eps0, kx, ky = solution[0], solution[1], solution[2]
    strain_history = (
        eps0[:, np.newaxis, np.newaxis]
        + kx[:, np.newaxis, np.newaxis] * y_map[np.newaxis, :, :]
        - ky[:, np.newaxis, np.newaxis] * x_map[np.newaxis, :, :]
    )
    stress_histories = [material.elastic_modulus * strain_history for material in materials]
    return strain_history, stress_histories[0], stress_histories[1]
