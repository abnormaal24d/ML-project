"""Shared text normalization helpers for augmentation identity checks."""

from __future__ import annotations

from functools import lru_cache


def text_field_value(value: object) -> str:
    """Return text content while preserving internal formatting."""

    if value is None:
        return ""
    return str(value).strip()


def single_line_value(value: object) -> str:
    """Return normalized single-line text."""

    if value is None:
        return ""
    return " ".join(str(value).split())


def text_identity(value: object) -> str:
    """Return normalized identity for duplicate detection only."""

    if value is None:
        return ""
    return _cached_text_identity(str(value))


@lru_cache(maxsize=8192)
def _cached_text_identity(value: str) -> str:
    """Cache normalized identities to avoid repeated text normalization."""

    return " ".join(value.split()).casefold()
