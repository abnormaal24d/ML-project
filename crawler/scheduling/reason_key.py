"""Stable serialization of scheduler admission and completion reasons."""

from __future__ import annotations

from enum import Enum


def reason_key(reason: object) -> str:
    """Return a stable string key for enum or scalar decision reasons."""

    return str(reason.value) if isinstance(reason, Enum) else str(reason)


__all__ = ["reason_key"]
