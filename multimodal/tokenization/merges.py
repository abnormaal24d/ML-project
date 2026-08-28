"""Shared byte-level BPE symbol operations."""

from __future__ import annotations


def apply_merge(
    sequence: list[bytes],
    *,
    left: bytes,
    right: bytes,
    merged: bytes,
) -> list[bytes]:
    """Replace adjacent symbol pairs with one merged symbol."""
    result: list[bytes] = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == left
            and sequence[index + 1] == right
        ):
            result.append(merged)
            index += 2
        else:
            result.append(sequence[index])
            index += 1
    return result


def symbol_key(symbol: bytes) -> str:
    """Return the stable serialized key for one byte symbol."""
    return f"<bytes:{symbol.hex()}>"


__all__ = ["apply_merge", "symbol_key"]
