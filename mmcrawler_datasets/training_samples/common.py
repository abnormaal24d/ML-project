"""Shared deterministic training-sample coercion and digest helpers."""

from __future__ import annotations

import hashlib
import mimetypes
import re
from pathlib import Path

from mmcrawler_datasets.training_samples.artifact_path import (
    ValidatedArtifactPath,
)

_TOKEN_ESTIMATE_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def stable_sample_id(
    entity_id: str,
    split: str,
    prefix: str = "sample",
) -> str:
    """Build a deterministic sample identifier from entity and split."""

    digest = hashlib.sha256(f"{entity_id}|{split}".encode()).hexdigest()
    return f"{prefix}_{digest[:24]}"


def estimate_token_count(text: object) -> int:
    """Count deterministic lexical tokens for all text-bearing samples."""

    if not text:
        return 0
    return len(_TOKEN_ESTIMATE_PATTERN.findall(str(text)))


def _validated_object_sha256(object_path: ValidatedArtifactPath) -> str:
    path = object_path.resolved_path
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a SHA-256 digest") from exc


def as_opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def as_opt_str_normalized(value: object) -> str | None:
    text = as_opt_str(value)
    return text.lower() if text else None


def as_opt_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(
        value,
        (str, bytes, bytearray, int, float),
    ):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        return None
    return None


def normalize_training_text(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text if text else None


def guess_mime_type(path: str | Path) -> str | None:
    if not path:
        return None
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type
