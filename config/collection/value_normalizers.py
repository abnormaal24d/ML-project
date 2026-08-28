"""Shared collection config value normalization helpers.

Exports: as_candidates, normalize_string_tuple,
    normalize_status_code_tuple.
"""

from __future__ import annotations

from collections.abc import Iterable


def as_candidates(value: object) -> tuple[object, ...]:
    """Return candidate values from scalar or iterable input."""

    if value is None:
        return ()

    if isinstance(value, (str, bytes, bytearray)):
        return (value,)

    if isinstance(value, Iterable):
        return tuple(value)

    return (value,)


def normalize_string_tuple(
    value: object,
    *,
    lowercase: bool = True,
    require_prefix: str | None = None,
) -> tuple[str, ...]:
    """Normalize string candidates with optional lowercasing and prefixing."""

    normalized: list[str] = []
    seen: set[str] = set()

    for candidate in as_candidates(value):
        item = str(candidate).strip()

        if lowercase:
            item = item.lower()

        if require_prefix and item and not item.startswith(require_prefix):
            item = f"{require_prefix}{item}"

        if not item or item in seen:
            continue

        seen.add(item)
        normalized.append(item)

    return tuple(normalized)


def normalize_status_code_tuple(
    value: object,
    *,
    field_name: str,
) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()

    for candidate in as_candidates(value):
        if isinstance(candidate, bool):
            raise ValueError(
                f"{field_name} must contain integer HTTP status codes"
            )

        if isinstance(candidate, int):
            status_code = candidate
        elif isinstance(candidate, float):
            if not candidate.is_integer():
                raise ValueError(
                    f"{field_name} must contain integer HTTP status codes"
                )
            status_code = int(candidate)
        elif isinstance(candidate, str):
            status_code = int(candidate.strip())
        elif isinstance(candidate, bytes):
            status_code = int(candidate.decode().strip())
        elif isinstance(candidate, bytearray):
            status_code = int(bytes(candidate).decode().strip())
        else:
            raise ValueError(
                f"{field_name} must contain integer HTTP status codes"
            )

        if status_code < 100 or status_code > 599:
            raise ValueError(
                f"{field_name} must contain valid HTTP status codes"
            )

        if status_code in seen:
            continue

        seen.add(status_code)
        normalized.append(status_code)

    return tuple(normalized)
