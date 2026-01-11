from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoltageBoundsKV:
    v_min_kv: float
    v_max_kv: float


def compute_voltage_bounds_kv(v_min: float, v_max: float, ub_kv: float) -> VoltageBoundsKV:
    """
    Normalize voltage bounds into kV.

    Backward-compatible rule:
    - If v_min/v_max look like per-unit (<= ~2.0), convert using UB (kV).
    - Otherwise treat them as kV already.
    """
    v_min_f = float(v_min)
    v_max_f = float(v_max)
    ub_f = float(ub_kv)

    if v_min_f <= 2.0 and v_max_f <= 2.0:
        return VoltageBoundsKV(v_min_kv=v_min_f * ub_f, v_max_kv=v_max_f * ub_f)
    return VoltageBoundsKV(v_min_kv=v_min_f, v_max_kv=v_max_f)

