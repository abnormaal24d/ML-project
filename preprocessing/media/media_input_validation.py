"""Shared media input validation and readiness rules."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.collection.modality_acceptance import (
        ModalityAcceptanceSettings,
    )
    from preprocessing.preprocessing_input import PreprocessingInput


def resolve_media_path(*, item: PreprocessingInput) -> str | None:
    """Resolve the canonical typed media path."""

    return as_optional_text(item.media_path)


def resolve_path_object(*, media_path: str | None) -> Path | None:
    """Return a Path object when a non-empty media path exists."""

    if media_path is None:
        return None
    return Path(media_path)


def modality_preprocessing_limit(
    acceptance: ModalityAcceptanceSettings,
) -> int:
    """Return the strict preprocessing byte cap for a modality."""

    for value in (
        acceptance.preprocessing_max_bytes,
        acceptance.fetch_max_bytes,
    ):
        if value is not None:
            return int(value)
    raise ValueError("modality acceptance requires a byte limit")


def resolve_mime_type(
    *, item: PreprocessingInput, media_path: str | None
) -> str | None:
    """Resolve the media MIME type from the record or filename suffix."""

    declared = as_optional_text(item.mime_type)
    if declared is not None:
        return declared.lower()
    if media_path:
        guessed, _encoding = mimetypes.guess_type(media_path)
        if guessed:
            return guessed.lower()
    return None


def resolve_byte_size(
    *,
    item: PreprocessingInput,
    path: Path | None,
) -> int | None:
    """Resolve byte size from the typed field or the physical file."""

    declared_size = as_optional_int(item.byte_size)
    if declared_size is not None:
        return declared_size
    if path is not None and path.exists():
        try:
            return path.stat().st_size
        except OSError:
            return None
    return None


def validate_common_media_fields(
    *,
    item: PreprocessingInput,
    allowed_mime_types: tuple[str, ...],
    min_bytes: int,
    max_bytes: int,
) -> tuple[str | None, dict[str, object]]:
    """Validate path, MIME and byte-size fields shared by all media."""

    media_path = resolve_media_path(item=item)
    path = resolve_path_object(media_path=media_path)
    mime_type = resolve_mime_type(item=item, media_path=media_path)
    byte_size = resolve_byte_size(item=item, path=path)
    signals: dict[str, object] = {
        "media_path": media_path,
        "mime_type": mime_type,
        "byte_size": byte_size,
        "path_exists": bool(path is not None and path.exists()),
    }

    if media_path is None:
        return "missing_media_path", signals
    if mime_type is None or mime_type not in set(allowed_mime_types):
        return "unsupported_mime_type", signals
    if path is None or not path.exists():
        return "file_not_found", signals
    if byte_size is None or byte_size < min_bytes:
        return "empty_media_payload", signals
    if byte_size > max_bytes:
        return "too_large", signals

    try:
        actual_size = path.stat().st_size
    except OSError:
        return "file_not_found", signals
    signals["actual_byte_size"] = actual_size
    if byte_size is not None and actual_size < byte_size:
        return "partial_download", signals
    return None, signals


def as_optional_text(value: object) -> str | None:
    """Coerce scalar metadata into normalized non-empty text when possible."""

    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def as_optional_int(value: object) -> int | None:
    """Coerce scalar metadata into an integer when possible."""

    number = as_optional_float(value)
    if number is None:
        return None
    return int(number)


def as_optional_float(value: object) -> float | None:
    """Coerce scalar metadata into a finite float when possible."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def payload_field(
    *,
    item: PreprocessingInput,
    name: str,
) -> Any:
    """Return one canonical payload field when it contains a value."""

    value = item.payload.get(name)
    return value if _has_meaningful_payload_value(value) else None


def _has_meaningful_payload_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


@dataclass(frozen=True, slots=True)
class MediaValidationResult:
    """Validated media status with structured diagnostic signals."""

    rejection_reason: str | None = None
    signals: dict[str, object] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        """Return True when the validator found no reject reason."""

        return self.rejection_reason is None


def accepted_media_result(
    *,
    signals: dict[str, object] | None = None,
) -> MediaValidationResult:
    """Build an accepted result while keeping call sites readable."""

    return MediaValidationResult(
        rejection_reason=None,
        signals=dict(signals or {}),
    )


def rejected_media_result(
    *,
    reason: str,
    signals: dict[str, object] | None = None,
) -> MediaValidationResult:
    """Build a rejected result with normalized signal storage."""

    return MediaValidationResult(
        rejection_reason=reason,
        signals=dict(signals or {}),
    )


_METADATA_FETCH_MODES = frozenset(
    {
        "metadata_only",
        "metadata_probe",
        "partial",
        "partial_probe",
        "embed_metadata",
        "head_only_oversized",
        "partial_probe_failed_fallback_head_only",
    }
)


def metadata_fetch_mode(*, payload: dict[str, object]) -> str:
    return str(payload.get("fetch_mode") or "").strip().lower()


def is_metadata_fetch_mode(*, payload: dict[str, object]) -> bool:
    return metadata_fetch_mode(payload=payload) in _METADATA_FETCH_MODES


def has_transcript_material(
    *, payload: dict[str, object], transcript_text: str | None
) -> bool:
    if as_optional_text(transcript_text):
        return True
    segments = payload.get("transcript_segments")
    return isinstance(segments, list) and bool(segments)


def has_video_training_metadata(
    *,
    payload: dict[str, object],
    transcript_text: str | None,
    ocr_text: str | None,
) -> bool:
    keyframes = payload.get("keyframes")
    if isinstance(keyframes, list) and keyframes:
        return True
    if as_optional_text(ocr_text) or as_optional_text(
        payload.get("frame_ocr_text")
    ):
        return True
    if has_transcript_material(
        payload=payload, transcript_text=transcript_text
    ):
        return True
    return as_optional_float(payload.get("video_duration_seconds")) is not None


def has_audio_training_metadata(
    *,
    payload: dict[str, object],
    transcript_text: str | None,
) -> bool:
    if has_transcript_material(
        payload=payload, transcript_text=transcript_text
    ):
        return True
    return as_optional_float(payload.get("audio_duration_seconds")) is not None
