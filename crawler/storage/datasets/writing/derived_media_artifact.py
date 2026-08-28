"""Transactional preparation of derived media artifacts.

Normalized image/video files and selected keyframes are produced as temporary
scratch by the processing handlers. ``DatasetWritePipeline`` is the single
transactional owner of every persistent derived artifact: it relocates them
into run-owned relative paths, registers them in the existing write journal's
``tracked_paths``, and commits them atomically with the record manifest.

This module only prepares, commits, and consumes those copies. It introduces
no separate media transaction machinery.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

_DERIVED_RELATIVE_ROOT = Path("media") / "derived"
_COPY_BUFFER_SIZE = 1024 * 1024
_SAFE_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")


class DerivedMediaArtifactError(ValueError):
    """Raised when a derived media artifact cannot be trusted."""


@dataclass(frozen=True, slots=True)
class PreparedDerivedMediaWrite:
    """Run-owned artifact copy prepared for a later transactional commit."""

    relative_path: str
    absolute_path: Path
    source_path: Path
    sha256: str


class DerivedMediaArtifactWriter:
    """Prepare and commit derived media copies under a crawl run directory."""

    def __init__(self, *, run_directory: Path) -> None:
        self._run_directory = run_directory.resolve()

    def prepare(
        self,
        *,
        source_path: Path,
        fetch_record_id: str,
        artifact_name: str,
    ) -> PreparedDerivedMediaWrite:
        """Verify the scratch source and compute its run-owned destination.

        Raises DerivedMediaArtifactError when the source scratch file is
        missing, so an enrichment path never silently leaks into the record.
        """

        source = source_path.resolve()
        if not source.is_file():
            raise DerivedMediaArtifactError(
                f"derived media artifact source missing: {source}"
            )

        digest = _sha256_file(source)
        safe_id = _safe_id(fetch_record_id)
        safe_name = _safe_artifact_name(artifact_name)
        content_addressed_name = _name_with_digest(
            artifact_name=safe_name,
            digest=digest,
        )
        relative = (
            _DERIVED_RELATIVE_ROOT / safe_id / content_addressed_name
        ).as_posix()
        absolute = self._run_directory / relative
        return PreparedDerivedMediaWrite(
            relative_path=relative,
            absolute_path=absolute,
            source_path=source,
            sha256=digest,
        )

    def commit(self, *, prepared: PreparedDerivedMediaWrite) -> None:
        """Persist a prepared artifact copy with fsync of file and parent dir."""

        absolute = prepared.absolute_path
        absolute.parent.mkdir(parents=True, exist_ok=True)
        temp_path = absolute.with_suffix(absolute.suffix + ".tmp")
        try:
            with (
                temp_path.open("wb") as target,
                prepared.source_path.open("rb") as source,
            ):
                shutil.copyfileobj(source, target, length=_COPY_BUFFER_SIZE)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_path, absolute)
            _fsync_directory(absolute.parent)
        finally:
            # Remove leftover temp when replace/fsync fails mid-commit.
            temp_path.unlink(missing_ok=True)


def consume_derived_media_source(
    prepared: PreparedDerivedMediaWrite,
) -> None:
    """Best-effort removal of the temporary scratch source after consumption."""

    prepared.source_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in str(value).strip()
    )
    if not cleaned or cleaned in {".", ".."}:
        raise DerivedMediaArtifactError(
            "fetch_record_id is not a safe path segment"
        )
    return cleaned


def _safe_artifact_name(value: str) -> str:
    cleaned = "".join(
        ch if ch in _SAFE_NAME_CHARS else "_"
        for ch in str(value).strip().lower()
    )
    if not cleaned or cleaned.startswith(".") or ".." in cleaned:
        raise DerivedMediaArtifactError(
            f"invalid derived media artifact name: {value!r}"
        )
    return cleaned


def _name_with_digest(*, artifact_name: str, digest: str) -> str:
    """Build a bounded content-addressed name for a derived artifact.

    Scratch filenames can already contain random identifiers and normalization
    suffixes.  Carrying that complete stem into the durable path pushes valid
    Windows workspaces beyond the legacy 260-character path limit.  The full
    digest is the durable identity, so retaining only the media extension keeps
    names deterministic, collision-resistant, and bounded.
    """

    suffix = Path(artifact_name).suffix
    if not suffix:
        return digest
    return f"{digest}{suffix}"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
