"""Validate and bind a training-snapshot tokenizer artifact to settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from config.multimodal.training_settings import TrainingSettings


def bind_snapshot_tokenizer(
    *,
    training: TrainingSettings,
    manifest_filename: str,
    training_root: Path,
) -> TrainingSettings:
    """Return immutable training settings bound to a snapshot tokenizer."""
    root = Path(training_root).resolve()
    identity = _load_tokenizer_identity(
        root=root,
        manifest_filename=manifest_filename,
    )

    artifact_path = _resolve_snapshot_artifact(
        root=root,
        relative_path=identity["path"],
    )
    observed_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    if observed_sha256 != identity["sha256"]:
        raise ValueError(
            "training snapshot tokenizer SHA-256 mismatch: "
            f"expected={identity['sha256']}, observed={observed_sha256}"
        )

    training_payload = training.model_dump()
    training_payload.update(
        {
            "text_tokenizer_path": str(artifact_path),
            "text_tokenizer_sha256": observed_sha256,
            "text_tokenizer_artifact_version": identity["artifact_version"],
            "text_tokenizer_vocab_size": identity["vocab_size"],
            "text_tokenizer_special_tokens": identity["special_tokens"],
        }
    )

    return TrainingSettings.model_validate(training_payload)


def _load_tokenizer_identity(
    *,
    root: Path,
    manifest_filename: str,
) -> dict[str, Any]:
    manifest_path = root / manifest_filename

    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"training snapshot manifest is unreadable: {manifest_path}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError("training snapshot manifest must be an object")

    identity = value.get("tokenizer")
    if not isinstance(identity, dict):
        raise ValueError(
            "training snapshot manifest has no tokenizer identity"
        )

    required = {
        "path",
        "sha256",
        "artifact_version",
        "vocab_size",
        "special_tokens",
    }
    missing = sorted(required - identity.keys())
    if missing:
        raise ValueError(f"tokenizer identity is missing fields: {missing}")

    return {
        "path": str(identity["path"]),
        "sha256": str(identity["sha256"]).strip().lower(),
        "artifact_version": str(identity["artifact_version"]),
        "vocab_size": int(identity["vocab_size"]),
        "special_tokens": {
            str(token): int(token_id)
            for token, token_id in dict(identity["special_tokens"]).items()
        },
    }


def _resolve_snapshot_artifact(
    *,
    root: Path,
    relative_path: str,
) -> Path:
    path = Path(relative_path)
    if not relative_path or path.is_absolute():
        raise ValueError("snapshot tokenizer path must be relative")

    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "snapshot tokenizer path escapes training root"
        ) from exc

    if not candidate.is_file():
        raise ValueError(
            f"snapshot tokenizer artifact is missing: {candidate}"
        )

    return candidate


__all__ = ["bind_snapshot_tokenizer"]
