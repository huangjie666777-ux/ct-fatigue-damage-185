"""Fatigue damage from ordered load histories (ASTM E1049 rainflow + Goodman/Miner).

Loads (N, Mx, My) are given as one ordered history and the stress at each
pixel is reconstructed first; rainflow counting is then applied to that
stress history. Load components are never counted separately and added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io_utils import ValidationError
from .section import SectionMaterial

MIN_HISTORY_POINTS = 2
MAX_HISTORY_POINTS = 2000
FATIGUE_MAX_OUTPUT_SIZE = 64


@dataclass(frozen=True)
class FatigueMaterial:
    base: SectionMaterial
    s_ref: float   # MPa
    n_ref: float   # cycles
    sn_exponent: float
    su: float      # ultimate tensile strength, MPa


@dataclass(frozen=True)
class HistoryPoint:
    n_force: float  # N
    mx: float       # N*mm
    my: float       # N*mm


@dataclass(frozen=True)
class RainflowCycle:
    count: float          # 1.0 closed cycle, 0.5 residual
    amplitude: float      # half range, MPa
    mean: float           # MPa


def _positive_finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValidationError(f"{field} must be finite and strictly positive")
    return result


def validate_fatigue_materials(spec: object) -> list[FatigueMaterial]:
    """Validate the two materials, extending section props with S-N curves."""
    from .section import validate_section_materials

    bases = validate_section_materials(spec)
    if not isinstance(spec, (list, tuple)):
        raise ValidationError("materials must be a list of two material objects")
    materials: list[FatigueMaterial] = []
    for base, entry in zip(bases, spec):
        if not isinstance(entry, dict):
            raise ValidationError("each material must be an object")
        try:
            materials.append(
                FatigueMaterial(
                    base=base,
                    s_ref=_positive_finite(entry["Sref"], f"material {base.name!r} Sref"),
                    n_ref=_positive_finite(entry["Nref"], f"material {base.name!r} Nref"),
                    sn_exponent=_positive_finite(entry["m"], f"material {base.name!r} m"),
                    su=_positive_finite(entry["Su"], f"material {base.name!r} Su"),
                )
            )
        except KeyError as exc:
            raise ValidationError(
                f"material {base.name!r} is missing key {exc}"
            ) from exc
    return materials


def validate_history(spec: object) -> list[HistoryPoint]:
    """Validate the ordered (N, Mx, My) history, 2 to 2000 finite points."""
    if not isinstance(spec, (list, tuple)) or not (
        MIN_HISTORY_POINTS <= len(spec) <= MAX_HISTORY_POINTS
    ):
        raise ValidationError(
            f"history must be a list of {MIN_HISTORY_POINTS} to "
            f"{MAX_HISTORY_POINTS} points"
        )
    points: list[HistoryPoint] = []
    for index, entry in enumerate(spec):
        if not isinstance(entry, dict):
            raise ValidationError(f"history point {index} must be an object")
        values = []
        for key in ("N", "Mx", "My"):
            if key not in entry:
                raise ValidationError(
                    f"history point {index} is missing key {key!r}"
                )
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(
                    f"history point {index} {key} must be a number"
                )
            value = float(value)
            if not np.isfinite(value):
                raise ValidationError(
                    f"history point {index} {key} must be finite"
                )
            values.append(value)
        points.append(HistoryPoint(values[0], values[1], values[2]))
    return points


def validate_repeat_count(value: object) -> int:
    """R must be a positive integer (booleans rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("R must be a positive integer")
    if value < 1:
        raise ValidationError("R must be a positive integer")
    return value


def turning_points(values: np.ndarray) -> np.ndarray:
    """Collapse consecutive repeats, then keep endpoints and reversals.

    Per ASTM E1049 X2.2 flat regions reduce to a single point first; the
    endpoints are always retained and only interior sign changes of slope
    survive.
    """
    series = np.asarray(values, dtype=np.float64)
    if series.size < 2:
        return series.copy()
    keep = np.ones(series.size, dtype=bool)
    keep[1:] = series[1:] != series[:-1]
    series = series[keep]
    if series.size <= 2:
        return series
    is_reversal = np.ones(series.size, dtype=bool)
    diffs = np.diff(series)
    is_reversal[1:-1] = (diffs[:-1] * diffs[1:]) < 0.0
    return series[is_reversal]


def rainflow_cycles(extrema: np.ndarray) -> list[RainflowCycle]:
    """ASTM E1049-85 (2017) rainflow counting, three-point online form.

    Closed cycles count 1; each value left in the stack at the end counts
    0.5 (residual). The sequence is never closed head-to-tail. Only the
    two newer stack points are dropped on a count, so the older range is
    re-examined on the following iteration (equivalent to restarting the
    scan in the standard's list formulation).
    """
    points = list(np.asarray(extrema, dtype=np.float64))
    if len(points) < 2:
        return []
    stack: list[float] = []
    cycles: list[RainflowCycle] = []
    for value in points:
        stack.append(value)
        while len(stack) >= 3:
            x1, x2, x3 = stack[-3], stack[-2], stack[-1]
            x_range = abs(x3 - x2)   # newer range X
            y_range = abs(x2 - x1)   # previous range Y
            if x_range < y_range:
                break
            if len(stack) == 3:
                # Y contains the starting point: count it as a half cycle
                # and discard the first point.
                cycles.append(
                    RainflowCycle(
                        count=0.5,
                        amplitude=0.5 * y_range,
                        mean=0.5 * (stack[0] + stack[1]),
                    )
                )
                del stack[0]
            else:
                # X >= Y with at least four points: Y is a closed cycle.
                cycles.append(
                    RainflowCycle(
                        count=1.0,
                        amplitude=0.5 * y_range,
                        mean=0.5 * (x1 + x2),
                    )
                )
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for start, end in zip(stack[:-1], stack[1:]):
        cycles.append(
            RainflowCycle(
                count=0.5,
                amplitude=0.5 * abs(end - start),
                mean=0.5 * (start + end),
            )
        )
    return cycles


def block_damage(
    stress_history: np.ndarray,
    material: FatigueMaterial,
    position_label: str | None = None,
) -> tuple[float, list[dict]]:
    """Miner single-block damage for one pixel's ordered stress history.

    Returns (damage, cycle details with Goodman-corrected amplitudes).
    A non-positive Goodman denominator (mean >= Su) on a non-zero cycle is
    rejected outright. Zero-amplitude cycles contribute no damage.
    """
    extrema = turning_points(stress_history)
    cycles = rainflow_cycles(extrema)
    damage = 0.0
    details: list[dict] = []
    for cycle in cycles:
        amplitude = cycle.amplitude
        if amplitude <= 0.0:
            continue
        denominator = 1.0 - max(cycle.mean, 0.0) / material.su
        if denominator <= 0.0:
            location = f" at {position_label}" if position_label else ""
            raise ValidationError(
                f"Goodman correction denominator {denominator:g} is not positive "
                f"(mean stress {cycle.mean:g} MPa vs Su {material.su:g} MPa)"
                f"{location}"
            )
        corrected = amplitude / denominator
        ratio = material.s_ref / corrected
        if ratio <= 0.0 or not np.isfinite(ratio):
            raise ValidationError("non-finite Goodman-corrected stress amplitude")
        n_fail = material.n_ref * ratio ** material.sn_exponent
        if not np.isfinite(n_fail) or n_fail <= 0.0:
            raise ValidationError("non-finite computed fatigue life Nf")
        damage += cycle.count / n_fail
        details.append(
            {
                "count": cycle.count,
                "amplitude_mpa": amplitude,
                "mean_mpa": cycle.mean,
                "goodman_amplitude_mpa": corrected,
                "Nf": n_fail,
                "damage": cycle.count / n_fail,
            }
        )
    return damage, details
