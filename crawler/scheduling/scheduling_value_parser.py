"""Scheduling-specific value coercion helpers.

These are domain-specific to avoid depending on crawler.runtime for scalar parsing.
"""

from __future__ import annotations

from typing import Any, cast

from crawler.numeric import finite_float_or_none


def coerce_int(
    value: object,
    *,
    default: int | None = None,
    allow_bool: bool = False,
) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(cast(Any, value)) if allow_bool else default
    try:
        if isinstance(value, str) and "." in value.strip():
            return int(float(value.strip()))
        return int(cast(Any, value))
    except (TypeError, ValueError, OverflowError):
        return default


def coerce_float(
    value: object,
    *,
    default: float | None = None,
    allow_bool: bool = False,
) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(cast(Any, value)) if allow_bool else default
    parsed = finite_float_or_none(cast(Any, value))
    return default if parsed is None else parsed


def coerce_bool(
    value: object,
    *,
    default: bool | None = None,
) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        return default
    return bool(value) if default is None else default


def coerce_lower_str(value: object) -> str | None:
    text = coerce_str(value, strip=True, empty_as_none=True)
    if text is None:
        return None
    return text.lower()


def coerce_str(
    value: object,
    *,
    strip: bool = True,
    empty_as_none: bool = True,
) -> str | None:
    if value is None:
        return None
    text = str(value)
    if strip:
        text = text.strip()
    if empty_as_none and not text:
        return None
    return text


def first_int_attribute(
    owner: object,
    *,
    names: tuple[str, ...],
    default: int,
) -> int:
    for name in names:
        parsed = coerce_int(getattr(owner, name, None))
        if parsed is not None:
            return parsed
    return int(default)
