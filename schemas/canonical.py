"""Canonical JSON serialization shared across package boundaries."""

from __future__ import annotations

import hashlib
import json


def canonical_json(payload: object) -> bytes:
    """Serialize a JSON-compatible payload with canonical ordering."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def stable_payload_fingerprint(payload: object) -> str:
    """Return a canonical SHA-256 digest for a JSON-compatible payload."""

    return hashlib.sha256(canonical_json(payload)).hexdigest()


__all__ = [
    "canonical_json",
    "stable_payload_fingerprint",
]
