"""Finite numeric coercion shared by crawler runtime domains."""

from __future__ import annotations

import math
from typing import SupportsFloat, SupportsIndex, cast

from typing_extensions import Buffer


def finite_float_or_none(value: object) -> float | None:
    """Return a finite float, or "None" when the value is invalid."""

    try:
        parsed = float(
            cast(str | Buffer | SupportsFloat | SupportsIndex, value)
        )
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def coerce_finite_float(
    value: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce an external numeric value to a finite, bounded float."""

    fallback = finite_float_or_none(default)
    parsed = finite_float_or_none(value)
    result = fallback if parsed is None else parsed
    if result is None:
        result = 0.0

    lower_bound = finite_float_or_none(minimum)
    upper_bound = finite_float_or_none(maximum)
    if lower_bound is not None:
        result = max(result, lower_bound)
    if upper_bound is not None:
        result = min(result, upper_bound)
    return result
