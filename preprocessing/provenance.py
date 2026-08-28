"""Producer provenance and stable hashing for preprocessing artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Mapping


class ProducerType(StrEnum):
    """Kinds of producers allowed to create preprocessing artifacts."""

    BUILTIN = "builtin"
    EXTERNAL_MODEL = "external_model"
    EXTERNAL_TOOL = "external_tool"
    SOURCE = "source"
    HUMAN = "human"


@dataclass(frozen=True, slots=True)
class ProducerProvenance:
    """Canonical provenance for one derived preprocessing artifact.

    Model-specific values are optional because deterministic, source and
    human producers have no model.  Production configuration is responsible
    for requiring ``model_id``, ``model_revision`` and ``artifact_hash`` for
    learned producers.
    """

    producer_type: ProducerType
    producer_name: str
    producer_version: str
    parameters_hash: str
    source_hash: str
    output_hash: str
    model_id: str | None = None
    model_revision: str | None = None
    artifact_hash: str | None = None
    confidence: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text("producer_name", self.producer_name)
        _require_text("producer_version", self.producer_version)
        _require_sha256("parameters_hash", self.parameters_hash)
        _require_sha256("source_hash", self.source_hash)
        _require_sha256("output_hash", self.output_hash)
        if self.artifact_hash is not None:
            _require_sha256("artifact_hash", self.artifact_hash)
        if self.model_id is not None:
            _require_text("model_id", self.model_id)
        if self.model_revision is not None:
            _require_text("model_revision", self.model_revision)
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("warnings may not contain blank values")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-ready representation."""

        return {
            "producer_type": self.producer_type.value,
            "producer_name": self.producer_name,
            "producer_version": self.producer_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "artifact_hash": self.artifact_hash,
            "parameters_hash": self.parameters_hash,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "source_hash": self.source_hash,
            "output_hash": self.output_hash,
        }


def hash_bytes(value: bytes) -> str:
    """Return a lowercase SHA-256 digest for source or output bytes."""

    return sha256(value).hexdigest()


def hash_text(value: str) -> str:
    """Hash normalized text as UTF-8."""

    return hash_bytes(value.encode("utf-8"))


def hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into a SHA-256 digest without loading it into memory."""

    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_parameters(parameters: Mapping[str, object]) -> str:
    """Hash backend parameters using canonical JSON serialization."""

    encoded = json.dumps(
        parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hash_bytes(encoded)


def stable_int_hash(*, value: str) -> int:
    """Return a deterministic 64-bit integer hash for bucket signals."""

    digest = hashlib.blake2b(
        value.encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def stable_identifier(*, prefix: str, parts: tuple[str, ...]) -> str:
    """Return a deterministic short identifier for preprocessing entities."""

    payload = "\x1f".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _require_text(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_sha256(field_name: str, value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
