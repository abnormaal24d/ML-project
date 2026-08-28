"""Content-addressed cache for generated augmentation media outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from logger.project_logger import ProjectLogger

_AUGMENTATION_CACHE_CODE_VERSION = "augmentation-cache-v4"


class AugmentationCache:
    """Reuse generated media outputs by source hash and operation settings."""

    def __init__(
        self,
        *,
        enabled: bool,
        cache_directory: str | Path,
        logger: ProjectLogger,
    ) -> None:
        self._enabled = enabled
        self._cache_directory = Path(cache_directory)
        self._logger = logger
        self._logger.debug("augmentation_cache_initialized")

    def cache_key(
        self,
        *,
        source_path: Path,
        operation: str,
        settings_digest: str,
    ) -> str:
        """Build a stable cache key for one source and operation."""

        source_hash = _file_hash(path=source_path)
        payload = "|".join(
            (
                source_hash,
                operation,
                settings_digest,
                _AUGMENTATION_CACHE_CODE_VERSION,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def restore(
        self,
        *,
        dataset_root: Path,
        cache_key: str,
        output_path: Path,
        expected_metadata: dict[str, object],
    ) -> bool:
        """Restore an existing cached output into the dataset if available."""

        if not self._enabled:
            return False
        cached_path = self._cache_output_path(
            dataset_root=dataset_root,
            cache_key=cache_key,
            suffix=output_path.suffix,
        )
        metadata_path = cached_path.with_suffix(cached_path.suffix + ".json")
        if not cached_path.is_file() or not metadata_path.is_file():
            return False
        metadata = _read_cache_metadata(path=metadata_path)
        if metadata is None:
            return False
        if metadata.get("cache_key") != cache_key:
            return False
        if (
            metadata.get("cache_code_version")
            != _AUGMENTATION_CACHE_CODE_VERSION
        ):
            return False
        if metadata.get("output_sha256") != _file_hash(path=cached_path):
            return False
        if metadata.get("output_byte_size") != cached_path.stat().st_size:
            return False
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_atomic(src=cached_path, dst=output_path)
        return output_path.is_file() and output_path.stat().st_size > 0

    def store(
        self,
        *,
        dataset_root: Path,
        cache_key: str,
        output_path: Path,
        cache_metadata: dict[str, object],
    ) -> None:
        """Store generated output and cache metadata for future runs."""

        if not self._enabled or not output_path.is_file():
            return
        cached_path = self._cache_output_path(
            dataset_root=dataset_root,
            cache_key=cache_key,
            suffix=output_path.suffix,
        )
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_atomic(src=output_path, dst=cached_path)
        _write_json_atomic(
            path=cached_path.with_suffix(cached_path.suffix + ".json"),
            payload={
                **cache_metadata,
                "cache_key": cache_key,
                "cache_code_version": _AUGMENTATION_CACHE_CODE_VERSION,
                "output_sha256": _file_hash(path=cached_path),
                "output_byte_size": cached_path.stat().st_size,
            },
        )

    def _cache_output_path(
        self,
        *,
        dataset_root: Path,
        cache_key: str,
        suffix: str,
    ) -> Path:
        root = _resolve_cache_root(
            dataset_root=dataset_root,
            configured=self._cache_directory,
        )
        return root / cache_key[:2] / f"{cache_key}{suffix}"


def settings_fingerprint(payload: object) -> str:
    """Hash a settings payload for cache keys."""

    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolve_cache_root(*, dataset_root: Path, configured: Path) -> Path:
    """Resolve cache root relative to dataset or project directories."""

    if configured.is_absolute():
        return configured
    if configured.parts and configured.parts[0] == "data":
        for candidate in (dataset_root, *dataset_root.parents):
            has_project_marker = (candidate / "pyproject.toml").exists()
            has_data_directory = (candidate / "data").exists()
            if has_project_marker or has_data_directory:
                return candidate / configured
    return dataset_root / configured


def _file_hash(*, path: Path) -> str:
    """Return a SHA-256 digest for a file read in chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_atomic(*, src: Path, dst: Path) -> None:
    """Copy a cache object through a private temporary file and replace."""

    temp_path = dst.with_name(f"{dst.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(src, temp_path)
        os.replace(temp_path, dst)
    finally:
        temp_path.unlink(missing_ok=True)


def _read_cache_metadata(*, path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(*, path: Path, payload: dict[str, object]) -> None:
    """Write JSON atomically via a temporary file and replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
