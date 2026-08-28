"""Checkpoint contract: headers and blob-store exclusivity.

A checkpoint contract declares the mandatory integrity properties of every
checkpoint in one workflow execution:

* ``checkpoint_headers``   - every checkpoint must carry a standalone
  ``<name>.headers.json`` document that captures the model digest and the
  reproducibility fingerprint schema without loading the model file.
* ``checkpoint_blob_storage`` - every checkpoint must be persisted in one
  external content-addressable store via the canonical blob manifest.

The contract is enforced fail-closed: production workflows cannot start
without the required options, and checkpoint loading refuses checkpoints
that violate an active contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from training.runtime.checkpoint.io import (
    _atomic_write_json,
    checkpoint_sha256,
    resolve_checkpoint_model_path,
    safe_torch_load,
)

HEADERS_SCHEMA_VERSION = 1
HEADERS_SUFFIX = ".headers.json"


@dataclass(frozen=True, slots=True)
class CheckpointContract:
    """Declared checkpoint integrity requirements for one workflow run."""

    checkpoint_headers: bool = False
    checkpoint_blob_storage: Path | None = None
    staging_lock: Path | None = None


def checkpoint_headers_path(checkpoint_path: Path) -> Path:
    """Return the headers sidecar path for a checkpoint."""
    return checkpoint_path.with_name(checkpoint_path.name + HEADERS_SUFFIX)


def checkpoint_headers_present(checkpoint_path: Path) -> bool:
    """Return whether a valid headers sidecar already exists."""
    path = checkpoint_headers_path(checkpoint_path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("schema_version") == HEADERS_SCHEMA_VERSION
        and isinstance(payload.get("sha256"), str)
        and len(payload["sha256"]) == 64
        and isinstance(payload.get("checkpoint_schema"), dict)
    )


def write_checkpoint_headers(
    *,
    checkpoint_path: Path,
    metadata: dict[str, object],
    sha256: str,
    artifact_version: str,
    model_family: str,
) -> Path:
    """Write the standalone headers sidecar next to a checkpoint.

    The headers document is the machine-readable contract record of one
    checkpoint. It is written atomically after the model file, so readers can
    trust headers only when the matching model digest is present.
    """

    normalized = str(sha256).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("checkpoint sha256 must be a 64-character hex digest")
    schema = metadata.get("checkpoint_schema")
    if not isinstance(schema, dict) or not schema:
        raise ValueError(
            "checkpoint headers require a non-empty checkpoint_schema"
        )

    payload: dict[str, object] = {
        "schema_version": HEADERS_SCHEMA_VERSION,
        "sha256": normalized,
        "artifact_version": artifact_version,
        "model_family": model_family,
        "epochs": metadata.get("epochs"),
        "final_loss": metadata.get("final_loss"),
        "sample_count": metadata.get("sample_count"),
        "dataset_root": metadata.get("dataset_root"),
        "checkpoint_schema": dict(schema),
    }
    path = checkpoint_headers_path(checkpoint_path)
    _atomic_write_json(path=path, payload=payload)
    return path


def require_checkpoint_headers(
    *,
    checkpoint_path: Path,
    expected_model_family: str | None = None,
    expected_artifact_version: str | None = None,
) -> Path:
    """Fail closed when the checkpoint lacks a valid headers sidecar.

    The headers digest is cross-checked against the actual model digest so
    that a stale or mismatched headers document never passes. Identity
    fields (model_family, artifact_version) are cross-checked against the
    checkpoint payload when expected values are provided.
    """

    headers_path = checkpoint_headers_path(checkpoint_path)
    if not checkpoint_headers_present(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint contract requires headers: {headers_path}"
        )
    actual_sha256 = checkpoint_sha256(checkpoint_path)
    headers_payload = json.loads(headers_path.read_text(encoding="utf-8"))
    if headers_payload["sha256"] != actual_sha256:
        raise ValueError(
            "checkpoint headers digest does not match the checkpoint "
            f"model: {checkpoint_path}"
        )

    if (
        expected_model_family is not None
        or expected_artifact_version is not None
    ):
        cp_payload = safe_torch_load(checkpoint_path)
        if not isinstance(cp_payload, dict):
            raise ValueError("checkpoint payload must be a dictionary")
        if expected_model_family is not None:
            actual_family = cp_payload.get("model_family")
            if actual_family != expected_model_family:
                raise ValueError(
                    f"checkpoint headers model_family {expected_model_family!r} "
                    f"does not match payload {actual_family!r}"
                )
        if expected_artifact_version is not None:
            actual_version = cp_payload.get("artifact_version")
            if actual_version != expected_artifact_version:
                raise ValueError(
                    f"checkpoint headers artifact_version {expected_artifact_version!r} "
                    f"does not match payload {actual_version!r}"
                )

    return headers_path


def require_blob_checkpoint(
    *,
    checkpoint_path: Path,
    blob_storage: Path,
) -> Path:
    """Fail closed unless the checkpoint lives in the configured blob store."""

    manifest = _read_canonical_manifest(checkpoint_path)
    if manifest.get("kind") != "blob":
        raise FileNotFoundError(
            f"checkpoint is not a blob-backed checkpoint: {checkpoint_path}"
        )
    configured = manifest.get("blob_storage")
    expected_root = blob_storage.resolve()
    actual_root = (
        Path(str(configured)).resolve() if configured is not None else None
    )
    if actual_root != expected_root:
        raise ValueError(
            f"checkpoint blob store does not match the configured store: "
            f"{checkpoint_path}"
        )
    model_path = resolve_checkpoint_model_path(checkpoint_path)
    if model_path is None:
        raise ValueError("checkpoint blob pointer is invalid")
    try:
        resolved_model_path = model_path.resolve(strict=True)
    except OSError as error:
        raise ValueError("checkpoint blob pointer is invalid") from error
    if not resolved_model_path.is_relative_to(expected_root):
        raise ValueError("checkpoint blob pointer escapes the blob store")
    checksum_path = resolved_model_path.with_name(
        resolved_model_path.name + ".sha256"
    )
    if not resolved_model_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            f"checkpoint blob is unavailable: {resolved_model_path}"
        )
    return resolved_model_path


def read_checkpoint_headers_payload(
    *,
    checkpoint_path: Path,
    blob_storage: Path | None = None,
    expected_model_family: str | None = None,
    expected_artifact_version: str | None = None,
) -> dict[str, object]:
    """Return the validated headers payload for one checkpoint.

    When *blob_storage* is configured the checkpoint must resolve into that
    store before its headers are trusted. Identity fields are cross-checked
    against the checkpoint payload when expected values are provided.
    """

    if blob_storage is not None:
        require_blob_checkpoint(
            checkpoint_path=checkpoint_path,
            blob_storage=blob_storage,
        )
    require_checkpoint_headers(
        checkpoint_path=checkpoint_path,
        expected_model_family=expected_model_family,
        expected_artifact_version=expected_artifact_version,
    )
    headers_path = checkpoint_headers_path(checkpoint_path)
    payload = _read_json(headers_path)
    if payload is None:
        raise ValueError("checkpoint headers payload must be a JSON object")
    return payload


def _read_canonical_manifest(checkpoint_path: Path) -> dict[str, object]:
    data = _read_json(checkpoint_path)
    if data is None:
        raise FileNotFoundError(
            f"checkpoint is not a canonical manifest: {checkpoint_path}"
        )
    return data


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


__all__ = [
    "HEADERS_SCHEMA_VERSION",
    "CheckpointContract",
    "checkpoint_headers_path",
    "checkpoint_headers_present",
    "read_checkpoint_headers_payload",
    "require_blob_checkpoint",
    "require_checkpoint_headers",
    "write_checkpoint_headers",
]
