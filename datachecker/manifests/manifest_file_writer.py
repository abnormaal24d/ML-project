"""Atomic JSON manifest writing and shared writer base helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from config.environment.default_values import (
    DEFAULT_MANIFEST_REPLACE_RETRY_ATTEMPTS,
    DEFAULT_MANIFEST_REPLACE_RETRY_DELAY_SECONDS,
    DEFAULT_MANIFEST_REPLACE_RETRY_JITTER_SECONDS,
)
from config.path_resolution.resolved_config_paths import (
    relativize_payload_paths,
)
from schemas.versions import WORKFLOW_MANIFEST_SCHEMA_VERSION

if TYPE_CHECKING:
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from datachecker.manifests.artifact_manifest import RunArtifactIdentity
    from logger.project_logger import ProjectLogger


Now = Callable[[], datetime]
GenerateId = Callable[[], str]

_REPLACE_LOCK = Lock()

_ARTIFACT_PATH_FIELDS = frozenset(
    {
        "raw_run_directory",
        "raw_records_manifest_path",
        "raw_errors_manifest_path",
        "run_summary_path",
        "curated_snapshot_directory",
        "curated_snapshot_manifest_path",
        "training_snapshot_directory",
        "training_dataset_manifest_path",
        "augmented_training_directory",
        "augmented_dataset_manifest_path",
        "input_dataset_root",
        "checkpoint_path",
        "metrics_path",
        "last_successful_manifest_path",
    }
)


class ManifestWriterBase:
    """Shared manifest serialization, validation, and writing."""

    def __init__(
        self,
        *,
        artifact_path_registry: ArtifactPathRegistry,
        logger: ProjectLogger,
        project_root: Path | None,
        file_writer: ManifestFileWriter,
        artifact_identity: RunArtifactIdentity,
        now: Now,
    ) -> None:
        self._artifact_path_registry = artifact_path_registry
        self._logger = logger
        self._project_root = (
            Path(project_root).resolve() if project_root is not None else None
        )
        self._file_writer = file_writer
        self._artifact_identity = artifact_identity
        self._now = now
        self._write_lock = Lock()

    def _utc_now_iso(self) -> str:
        """Return the injected clock as an ISO 8601 timestamp."""

        return self._now().isoformat()

    def _identity_fields(self) -> dict[str, str]:
        """Return the active workflow artifact identity."""

        return self._artifact_identity.manifest_fields()

    def _require_same_generation(self, manifest: object) -> None:
        """Reject a dependency from another workflow generation."""

        identity_fields = getattr(manifest, "identity_fields", None)

        if not callable(identity_fields):
            raise TypeError(
                "Workflow dependency must be an artifact manifest."
            )

        if identity_fields() != self._identity_fields():
            raise RuntimeError(
                "Workflow artifact identity does not match the active "
                "generation."
            )

    def _write_manifest(
        self,
        *,
        path: Path,
        payload: Mapping[str, object],
    ) -> None:
        """Serialize, validate, and atomically write one manifest."""

        portable_payload: dict[str, object] = {
            "manifest_schema_version": (WORKFLOW_MANIFEST_SCHEMA_VERSION),
            **dict(payload),
        }

        resolved_payload = relativize_payload_paths(
            portable_payload,
            project_root=self._project_root,
        )

        if not isinstance(resolved_payload, dict):
            raise TypeError("Path relativizer must return a dictionary.")

        serialized_payload = {
            str(key): value for key, value in resolved_payload.items()
        }

        for field_name, configured_value in self._iter_artifact_paths(
            serialized_payload
        ):
            artifact_path = Path(configured_value)

            if (
                not artifact_path.is_absolute()
                and self._project_root is not None
            ):
                artifact_path = self._project_root / artifact_path

            if not artifact_path.exists():
                raise FileNotFoundError(
                    "Manifest references a missing artifact for "
                    f"{field_name}: {configured_value}; "
                    f"resolved_path={artifact_path}; "
                    f"project_root={self._project_root}; "
                    f"cwd={Path.cwd()}"
                )

            if not os.access(artifact_path, os.R_OK):
                raise PermissionError(
                    "Manifest references an unreadable artifact for "
                    f"{field_name}: {configured_value}; "
                    f"resolved_path={artifact_path}; "
                    f"project_root={self._project_root}; "
                    f"cwd={Path.cwd()}"
                )

        with self._write_lock:
            self._file_writer.write(
                path=path,
                payload=serialized_payload,
            )

    @classmethod
    def _iter_artifact_paths(
        cls,
        payload: object,
    ) -> Iterator[tuple[str, str]]:
        """Yield artifact paths from a nested manifest payload once."""

        if isinstance(payload, dict):
            for key, value in payload.items():
                field_name = str(key)

                if field_name in _ARTIFACT_PATH_FIELDS and cls._is_path_value(
                    value
                ):
                    yield field_name, str(value)

                if isinstance(value, (dict, list)):
                    yield from cls._iter_artifact_paths(value)

        elif isinstance(payload, list):
            for value in payload:
                if isinstance(value, (dict, list)):
                    yield from cls._iter_artifact_paths(value)

    @staticmethod
    def _is_path_value(value: object) -> bool:
        """Return whether a payload value represents a local path."""

        if not isinstance(value, str):
            return False

        normalized = value.strip()

        if not normalized or normalized in {".", "./"}:
            return False

        lowered = normalized.lower()

        return "://" not in lowered and not lowered.startswith(
            ("mailto:", "urn:")
        )


class ManifestFileWriter:
    """Write JSON manifests using atomic same-directory replacement."""

    def __init__(
        self,
        *,
        now: Now,
        generate_id: GenerateId,
        replace_retry_attempts: int = (
            DEFAULT_MANIFEST_REPLACE_RETRY_ATTEMPTS
        ),
        replace_retry_delay_seconds: float = (
            DEFAULT_MANIFEST_REPLACE_RETRY_DELAY_SECONDS
        ),
        replace_retry_jitter_seconds: float = (
            DEFAULT_MANIFEST_REPLACE_RETRY_JITTER_SECONDS
        ),
    ) -> None:
        if replace_retry_attempts < 1:
            raise ValueError("replace_retry_attempts must be at least 1.")

        if replace_retry_delay_seconds < 0:
            raise ValueError(
                "replace_retry_delay_seconds must not be negative."
            )

        if replace_retry_jitter_seconds < 0:
            raise ValueError(
                "replace_retry_jitter_seconds must not be negative."
            )

        self._now = now
        self._generate_id = generate_id
        self._replace_retry_attempts = replace_retry_attempts
        self._replace_retry_delay_seconds = replace_retry_delay_seconds
        self._replace_retry_jitter_seconds = replace_retry_jitter_seconds

    def write(
        self,
        *,
        path: Path,
        payload: dict[str, object],
        preserve_previous: bool = True,
    ) -> None:
        """Atomically write one JSON manifest."""

        self._write_json_object(
            path=path,
            payload=payload,
            preserve_previous=preserve_previous,
        )

    async def awrite(
        self,
        *,
        path: Path,
        payload: dict[str, object],
        preserve_previous: bool = True,
    ) -> None:
        """Write one JSON manifest without blocking the event loop."""

        await asyncio.wait_for(
            asyncio.to_thread(
                self._write_json_object,
                path=path,
                payload=payload,
                preserve_previous=preserve_previous,
            ),
            timeout=60.0,
        )

    @staticmethod
    def read_json_object(
        *,
        path: Path,
    ) -> dict[str, object]:
        """Read a manifest, returning an empty mapping when invalid."""

        if not path.is_file():
            return {}

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        return payload if isinstance(payload, dict) else {}

    def _write_json_object(
        self,
        *,
        path: Path,
        payload: dict[str, object],
        preserve_previous: bool,
    ) -> None:
        """Serialize and atomically replace one JSON object file."""

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary_path = Path(temporary_handle.name)

        try:
            with temporary_handle:
                json.dump(
                    payload,
                    temporary_handle,
                    indent=2,
                    sort_keys=True,
                )
                temporary_handle.flush()
                os.fsync(temporary_handle.fileno())

            if preserve_previous:
                self._preserve_previous_file(path)

            self._replace_with_retry(
                temporary_path=temporary_path,
                target_path=path,
            )
            self._fsync_directory(path.parent)

        except Exception:
            if temporary_path.exists():
                self._unlink_temporary_file(
                    temporary_path=temporary_path,
                    target_path=path,
                )
            raise

    def _preserve_previous_file(
        self,
        path: Path,
    ) -> None:
        """Copy the current manifest to its history directory."""

        if not path.is_file():
            return

        history_directory = path.parent / ".history"
        history_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = self._now().strftime("%Y%m%dT%H%M%S%fZ")
        identifier = self._generate_id()[:12]
        history_path = (
            history_directory / f"{path.name}.{timestamp}.{identifier}.bak"
        )

        shutil.copy2(path, history_path)

    def _replace_with_retry(
        self,
        *,
        temporary_path: Path,
        target_path: Path,
    ) -> None:
        """Replace a target file, retrying transient permission errors."""

        last_error: PermissionError | None = None

        for attempt in range(
            1,
            self._replace_retry_attempts + 1,
        ):
            try:
                with _REPLACE_LOCK:
                    os.replace(
                        temporary_path,
                        target_path,
                    )
                return

            except PermissionError as exc:
                last_error = exc

                if attempt >= self._replace_retry_attempts:
                    break

                retry_delay = (
                    self._replace_retry_delay_seconds * attempt
                    + self._retry_jitter(
                        temporary_path=temporary_path,
                        target_path=target_path,
                        attempt=attempt,
                    )
                )
                time.sleep(retry_delay)

            except (FileNotFoundError, IsADirectoryError) as exc:
                raise type(exc)(
                    f"Cannot replace JSON file {target_path}: {exc}"
                ) from exc

            except OSError as exc:
                raise OSError(
                    f"Cannot replace JSON file {target_path}: {exc}"
                ) from exc

        self._unlink_temporary_file(
            temporary_path=temporary_path,
            target_path=target_path,
        )

        raise PermissionError(
            f"Cannot replace JSON file {target_path}: {last_error}"
        ) from last_error

    def _retry_jitter(
        self,
        *,
        temporary_path: Path,
        target_path: Path,
        attempt: int,
    ) -> float:
        """Return deterministic retry jitter for one replacement attempt."""

        if self._replace_retry_jitter_seconds == 0:
            return 0.0

        seed_payload = (
            f"{temporary_path.as_posix()}\0{target_path.as_posix()}\0{attempt}"
        ).encode("utf-8")

        seed = int(
            hashlib.sha256(seed_payload).hexdigest()[:16],
            16,
        )

        # This jitter is deliberately reproducible, not security-sensitive.
        return random.Random(seed).uniform(  # nosec B311
            0.0,
            self._replace_retry_jitter_seconds,
        )

    @staticmethod
    def _unlink_temporary_file(
        *,
        temporary_path: Path,
        target_path: Path,
    ) -> None:
        """Safely remove a temporary manifest file."""

        if temporary_path.parent != target_path.parent:
            raise RuntimeError(
                "Refusing to remove a temporary file outside the "
                "target directory."
            )

        if not temporary_path.name.startswith(f"{target_path.name}."):
            raise RuntimeError(
                "Refusing to remove an unexpected temporary file."
            )

        if temporary_path.suffix != ".tmp":
            raise RuntimeError("Refusing to remove a non-temporary file.")

        temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Synchronize directory metadata after atomic replacement."""

        if os.name == "nt":
            return

        try:
            directory_descriptor = os.open(
                path,
                os.O_RDONLY,
            )
        except OSError:
            return

        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
