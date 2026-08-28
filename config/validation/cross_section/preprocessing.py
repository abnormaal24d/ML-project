"""Preprocessing backend and pinning rules."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from config.environment.source_selection import PRODUCTION_ENVIRONMENTS

if TYPE_CHECKING:
    from config.settings.root import Settings


def _missing_required_text_fields(
    value: Any,
    field_names: tuple[str, ...],
) -> list[str]:
    """Return names of missing or blank required text fields."""

    missing: list[str] = []
    for field_name in field_names:
        field_value = getattr(value, field_name, None)
        if field_value is None or (
            isinstance(field_value, str) and not field_value.strip()
        ):
            missing.append(field_name)
    return missing


def _validate_video_transcription_flags(settings: Settings) -> None:
    """Fail closed when video transcription intent cannot execute."""

    video = settings.collection.processors.video

    if not video.run_transcription:
        return

    if not video.generate_transcriptions:
        raise ValueError(
            "collection.processors.video.run_transcription=true requires "
            "collection.processors.video.generate_transcriptions=true"
        )

    if not video.extract_audio_track:
        raise ValueError(
            "collection.processors.video.run_transcription=true requires "
            "collection.processors.video.extract_audio_track=true"
        )


def _validate_preprocessing_configuration(
    settings: Settings,
) -> None:
    """Validate transcription alignment and production model pinning."""

    transcription = settings.preprocessing.transcription
    processors = settings.collection.processors

    transcription_requested = bool(
        processors.audio.run_transcription
        or processors.video.run_transcription
    )

    if transcription_requested != transcription.enabled:
        raise ValueError(
            "collection processor run_transcription settings must match "
            "preprocessing.transcription.enabled"
        )

    if settings.application.environment not in PRODUCTION_ENVIRONMENTS:
        return

    if transcription.enabled:
        missing = _missing_required_text_fields(
            transcription,
            (
                "model_revision",
                "model_artifact_hash",
                "backend_version",
            ),
        )

        if missing:
            raise ValueError(
                f"production transcription must pin these fields: {missing}"
            )

        if not transcription.local_files_only:
            raise ValueError(
                "production transcription must set local_files_only=true"
            )
        model_name = transcription.model_name
        if not (
            PurePosixPath(model_name).is_absolute()
            or PureWindowsPath(model_name).is_absolute()
        ):
            raise ValueError(
                "production transcription model_name must be an absolute "
                "local model directory"
            )

    ocr = settings.preprocessing.ocr
    ocr_backend = str(ocr.backend).strip().lower()

    if ocr_backend == "disabled":
        return

    missing = _missing_required_text_fields(
        ocr,
        (
            "backend_version",
            "model_id",
            "model_revision",
            "model_artifact_hash",
            "model_artifact_path",
        ),
    )

    if missing:
        raise ValueError(f"production OCR must pin these fields: {missing}")
