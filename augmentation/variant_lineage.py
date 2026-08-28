"""Deterministic identity and cryptographic lineage for media augmentation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from preprocessing.provenance import hash_file

MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION = "augmentation-media-v4"
MEDIA_AUGMENTATION_IMPLEMENTATION_HASH = hashlib.sha256(
    MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION.encode("utf-8")
).hexdigest()


def file_sha256(*, path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a regular file."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    return hash_file(path, chunk_size=chunk_size)


def media_variant_id(
    *,
    source_sample_id: str,
    operation: str,
    source_sha256: str,
    config_hash: str,
    prefix: str = "sample_media_aug",
) -> str:
    """Build a deterministic ID bound to source, config, and implementation."""

    payload = json.dumps(
        {
            "config_hash": config_hash,
            "implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
            "operation": operation,
            "source_sample_id": source_sample_id,
            "source_sha256": source_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def valid_sha256(value: object) -> bool:
    """Return whether a value is a canonical lowercase SHA-256 digest."""

    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)
