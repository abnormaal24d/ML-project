"""Concrete pinned faster-whisper backend adapter."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from config.errors import RuntimeDependencyError
from config.preprocessing.media_settings import TranscriptionSettings
from preprocessing.provenance import (
    ProducerProvenance,
    ProducerType,
    hash_file,
)


@dataclass(frozen=True, slots=True)
class WhisperModelReport:
    """Exact backend identity included in ASR artifact provenance."""

    backend_name: str
    backend_version: str
    model_id: str
    model_revision: str | None
    artifact_hash: str | None
    device: str
    compute_type: str
    cache_directory: str | None


def installed_backend_version() -> str:
    """Return the installed faster-whisper distribution version."""

    try:
        return version("faster-whisper")
    except PackageNotFoundError:
        return "unavailable"


def validate_whisper_artifact(
    *,
    settings: TranscriptionSettings,
    observed_backend_version: str,
) -> None:
    """Validate the pinned faster-whisper version and local model."""

    expected_version = settings.backend_version
    if (
        expected_version is not None
        and expected_version != observed_backend_version
    ):
        raise RuntimeDependencyError(
            "configured faster-whisper backend_version does not match "
            f"installed version: expected {expected_version!r}, "
            f"observed {observed_backend_version!r}",
            setting="preprocessing.transcription.backend_version",
            issue="backend_version_mismatch",
        )
    if not settings.local_files_only:
        return

    model_directory = Path(settings.model_name)
    if not settings.production_mode:
        if not model_directory.is_dir():
            raise RuntimeDependencyError(
                "configured local Whisper model directory is missing",
                setting="preprocessing.transcription.model_name",
                required_artifact=model_directory.name or "model_directory",
                issue="required_local_artifact_missing",
            )
        return

    expected_hash = settings.model_artifact_hash
    if expected_hash is None:
        raise RuntimeDependencyError(
            "production Whisper artifact hash is missing",
            setting="preprocessing.transcription.model_artifact_hash",
            issue="artifact_hash_missing",
        )
    if not model_directory.is_absolute():
        raise RuntimeDependencyError(
            "production Whisper model_name must be an absolute local path",
            setting="preprocessing.transcription.model_name",
            issue="local_artifact_path_invalid",
        )
    artifact_path = model_directory / "model.bin"
    if not artifact_path.is_file():
        raise RuntimeDependencyError(
            "production Whisper artifact is missing",
            setting="preprocessing.transcription.model_name",
            required_artifact="model.bin",
            issue="required_local_artifact_missing",
        )
    observed_hash = hash_file(artifact_path)
    if observed_hash != expected_hash:
        raise RuntimeDependencyError(
            "production Whisper artifact hash mismatch: "
            f"expected {expected_hash}, observed {observed_hash}",
            setting="preprocessing.transcription.model_artifact_hash",
            required_artifact="model.bin",
            issue="artifact_hash_mismatch",
        )


class WhisperModelLoader:
    """Thread-safe loader and lifecycle owner for faster-whisper."""

    def __init__(
        self,
        *,
        settings: TranscriptionSettings,
    ) -> None:
        if not settings.enabled or settings.backend != "whisper":
            raise ValueError(
                "WhisperModelLoader requires enabled whisper settings"
            )
        self._settings = settings
        self._model: Any | None = None
        self._lock = threading.Lock()
        self._backend_version = installed_backend_version()

    @property
    def report(self) -> WhisperModelReport:
        return WhisperModelReport(
            backend_name="faster-whisper",
            backend_version=self._backend_version,
            model_id=self._settings.model_name,
            model_revision=self._settings.model_revision,
            artifact_hash=self._settings.model_artifact_hash,
            device=self._settings.device,
            compute_type=self._settings.compute_type,
            cache_directory=self._settings.cache_directory,
        )

    def get_model(self) -> Any:
        """Return the cached model, loading and verifying it once."""

        cached_model = self._model
        if cached_model is not None:
            return cached_model

        with self._lock:
            cached_model = self._model
            if cached_model is not None:
                return cached_model

            try:
                whisper_model = cast(
                    Any,
                    import_module("faster_whisper"),
                ).WhisperModel
            except (AttributeError, ImportError) as exc:
                raise RuntimeError(
                    "faster-whisper is required for the configured "
                    "preprocessing transcription backend"
                ) from exc

            validate_whisper_artifact(
                settings=self._settings,
                observed_backend_version=self._backend_version,
            )
            model_kwargs: dict[str, object] = {
                "device": self._settings.device,
                "compute_type": self._settings.compute_type,
                "local_files_only": self._settings.local_files_only,
            }
            if self._settings.cache_directory is not None:
                model_kwargs["download_root"] = self._settings.cache_directory
            if self._settings.model_revision is not None:
                model_kwargs["revision"] = self._settings.model_revision

            self._model = whisper_model(
                self._settings.model_name,
                **model_kwargs,
            )
            return self._model

    def transcribe(
        self,
        audio_path: str | os.PathLike[str],
        **decode_settings: object,
    ) -> tuple[Any, Any]:
        """Transcribe through the repository-owned model."""

        segments, info = self.get_model().transcribe(
            str(audio_path),
            **decode_settings,
        )
        return segments, info

    def provenance(
        self,
        *,
        parameters_hash: str,
        source_hash: str,
        output_hash: str,
        confidence: float | None,
        warnings: tuple[str, ...] = (),
    ) -> ProducerProvenance:
        """Build the canonical provenance envelope from the pinned report."""

        report = self.report
        return ProducerProvenance(
            producer_type=ProducerType.EXTERNAL_MODEL,
            producer_name=report.backend_name,
            producer_version=report.backend_version,
            model_id=report.model_id,
            model_revision=report.model_revision,
            artifact_hash=report.artifact_hash,
            parameters_hash=parameters_hash,
            confidence=confidence,
            warnings=warnings,
            source_hash=source_hash,
            output_hash=output_hash,
        )

    def close(self) -> None:
        """Release the cached backend and its native resources."""

        with self._lock:
            model, self._model = self._model, None
        if model is None:
            return
        close = getattr(model, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> WhisperModelLoader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
