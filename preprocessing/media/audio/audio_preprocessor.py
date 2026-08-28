"""Audio preprocessing orchestration."""

from __future__ import annotations

import wave
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger
from preprocessing.media.audio.audio_fingerprint import (
    AudioFingerprintError,
)
from preprocessing.media.base_media_preprocessor import BaseMediaPreprocessor
from preprocessing.media.media_input_validation import (
    MediaValidationResult,
    accepted_media_result,
    as_optional_float,
    as_optional_int,
    as_optional_text,
    has_audio_training_metadata,
    is_metadata_fetch_mode,
    modality_preprocessing_limit,
    payload_field,
    rejected_media_result,
    resolve_media_path,
    resolve_path_object,
    validate_common_media_fields,
)
from preprocessing.media.privacy_inspection import (
    inspect_media_privacy,
)
from preprocessing.media.transcript_segment_normalizer import (
    normalize_segments,
    summarize_timeline,
)
from preprocessing.preprocessed_media import (
    PreprocessedAudio,
    canonical_privacy_intervals,
    canonical_time_intervals,
    canonical_transcript_segments,
)
from preprocessing.preprocessing_input import PreprocessingInput
from preprocessing.preprocessing_result import PreprocessingQuarantineRecord
from preprocessing.privacy.field_inspection import text_payload_fields
from preprocessing.privacy.inspection.inspect_audio import inspect_audio
from preprocessing.privacy.inspection.local_content_factories import (
    AudioPrivacyContentFactory,
)
from preprocessing.privacy.text_privacy import PiiDetector

if TYPE_CHECKING:
    from config.collection.modality_acceptance import (
        ModalityAcceptanceSettings,
    )
    from config.preprocessing.media_settings import AudioValidationSettings
    from preprocessing.media.ports import EmbeddedMetadataAdapter

_ALLOWED_AUDIO_MIME_TYPES: tuple[str, ...] = (
    "audio/mpeg",
    "audio/mp4",
    "audio/mp4a-latm",
    "audio/ogg",
    "audio/wav",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
    "audio/aac",
    "audio/flac",
)


class AudioPreprocessor(BaseMediaPreprocessor[PreprocessedAudio]):
    """Validate audio transcript, segments, metadata, and fingerprints."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        settings: AudioValidationSettings,
        modality_acceptance: ModalityAcceptanceSettings,
        max_duration_seconds: float,
        pii_detector: PiiDetector,
        privacy_content_factory: AudioPrivacyContentFactory,
        audio_fingerprint_calculator: Callable[[Path], str],
        embedded_metadata_adapter: EmbeddedMetadataAdapter,
        now: Callable[[], datetime],
        generate_id: Callable[[], str],
    ) -> None:
        if max_duration_seconds <= 0.0:
            raise ValueError("max_duration_seconds must be greater than zero")
        super().__init__(
            modality="audio",
            logger=logger,
            now=now,
            generate_id=generate_id,
        )
        self._settings = settings
        self._modality_acceptance = modality_acceptance
        self._max_duration_seconds = max_duration_seconds
        self._pii_detector = pii_detector
        self._privacy_content_factory = privacy_content_factory
        self._audio_fingerprint_calculator = audio_fingerprint_calculator
        self._embedded_metadata_adapter = embedded_metadata_adapter

    def _validate(self, *, item: PreprocessingInput) -> MediaValidationResult:
        if not self._settings.enabled:
            return accepted_media_result(signals={})
        reason, signals = validate_common_media_fields(
            item=item,
            allowed_mime_types=_ALLOWED_AUDIO_MIME_TYPES,
            min_bytes=self._settings.min_bytes,
            max_bytes=modality_preprocessing_limit(self._modality_acceptance),
        )
        media_path = resolve_media_path(item=item)
        path = resolve_path_object(media_path=media_path)
        duration = _duration_seconds(item=item)
        sample_rate = _sample_rate(item=item)
        channels = _channels(item=item)
        decode_failed = _decode_failed(path=path)
        has_transcript = _has_transcript(item=item)
        signals.update(
            {
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "has_transcript": has_transcript,
                "decode_checked": decode_failed is not None,
            }
        )
        if reason is not None:
            metadata_only = _accept_metadata_only_audio(
                item=item, reason=reason, signals=signals
            )
            if metadata_only is not None:
                return metadata_only
            return rejected_media_result(reason=reason, signals=signals)
        if decode_failed is True:
            metadata_only = _accept_metadata_only_audio(
                item=item, reason="decode_failed", signals=signals
            )
            if metadata_only is not None:
                return metadata_only
            return rejected_media_result(
                reason="decode_failed", signals=signals
            )
        if duration is None:
            return rejected_media_result(
                reason="duration_missing", signals=signals
            )
        if duration < self._settings.min_duration_seconds:
            return rejected_media_result(
                reason="duration_too_short", signals=signals
            )
        if duration > self._max_duration_seconds:
            return rejected_media_result(
                reason="duration_too_long", signals=signals
            )
        if sample_rate is None or sample_rate < self._settings.min_sample_rate:
            return rejected_media_result(
                reason="metadata_incomplete", signals=signals
            )
        if (
            channels is None
            or channels < self._settings.min_channels
            or channels > self._settings.max_channels
        ):
            return rejected_media_result(
                reason="metadata_incomplete", signals=signals
            )
        if self._settings.require_transcript and not has_transcript:
            return rejected_media_result(
                reason=(
                    "metadata_only_audio"
                    if _has_audio_metadata(
                        duration=duration,
                        sample_rate=sample_rate,
                        channels=channels,
                    )
                    else "missing_transcript"
                ),
                signals=signals,
            )
        return accepted_media_result(signals=signals)

    def _build_record(
        self,
        *,
        item: PreprocessingInput,
        validation: MediaValidationResult,
    ) -> PreprocessedAudio | PreprocessingQuarantineRecord:
        # Validators own resolved stream metadata; do not re-read item/payload.
        duration_seconds = as_optional_float(
            validation.signals.get("duration_seconds")
        )
        sample_rate = as_optional_int(validation.signals.get("sample_rate"))
        channels = as_optional_int(validation.signals.get("channels"))
        loudness_lufs = as_optional_float(item.payload.get("loudness_lufs"))
        transcript_text = item.transcript_text or as_optional_text(
            item.payload.get("transcript_text")
        )
        segments = normalize_segments(item.payload.get("transcript_segments"))
        fields = text_payload_fields(
            item=item,
            names=(
                "transcript_text",
                "transcript_preview",
                "page_title",
                "surrounding_text",
                "html_context",
                "author",
                "creator",
                "description",
            ),
        )
        if transcript_text:
            fields["transcript_text"] = transcript_text
        for index, segment in enumerate(segments):
            text = as_optional_text(segment.get("text"))
            if text:
                fields[f"transcript_segment:{index}"] = text
        duration_ms = int((duration_seconds or 0.0) * 1000)
        embedded_fields, metadata_artifact, metadata_rejection = (
            self._prepare_embedded_metadata(
                item=item,
                adapter=self._embedded_metadata_adapter,
            )
        )
        if metadata_rejection is not None:
            return PreprocessingQuarantineRecord.from_input(
                item=item,
                reason=metadata_rejection,
                quality_signals={},
            )
        fields.update(embedded_fields)

        inspected_path = Path(
            metadata_artifact.path
            if metadata_artifact is not None
            else item.media_path or ""
        )
        audio_fingerprint: str | None = None
        if self._settings.require_audio_fingerprint:
            try:
                audio_fingerprint = self._audio_fingerprint_calculator(
                    inspected_path
                )
            except AudioFingerprintError as exc:
                return PreprocessingQuarantineRecord.from_input(
                    item=item,
                    reason=str(exc),
                    quality_signals={
                        "audio_fingerprint_required": True,
                    },
                )

        full_decode_completed = _decode_failed(path=inspected_path) is False
        inspection_content = self._privacy_content_factory.build(
            item=item,
            media_path=inspected_path,
            metadata={},
            duration_ms=duration_ms,
            transcript_segments=segments,
            full_decode_completed=full_decode_completed,
            audio_fingerprint=audio_fingerprint,
            residual=False,
        )
        inspection = inspect_audio(
            inspection_content,
            self._pii_detector.registry,
        )
        privacy = inspect_media_privacy(
            item=item,
            object_id=self._media_id(item=item),
            detector=self._pii_detector,
            fields=fields,
            inspection=inspection,
            media_path=str(inspected_path),
            source_media_path=item.media_path,
            inspected_artifact=metadata_artifact,
            content_field_prefixes=("transcript",),
        )
        if privacy.rejection_reason is not None:
            return PreprocessingQuarantineRecord.from_input(
                item=item,
                reason=privacy.rejection_reason,
                quality_signals={
                    "privacy_status": privacy.clearance.status.value,
                    "privacy_reasons": list(privacy.clearance.reasons),
                },
            )
        transcript_text = privacy.fields.get("transcript_text")
        segments = tuple(
            {
                **segment,
                "text": privacy.fields.get(
                    f"transcript_segment:{index}",
                    str(segment.get("text") or ""),
                ),
            }
            for index, segment in enumerate(segments)
        )
        semantic_text = transcript_text or _segments_text(segments=segments)
        quality = self._quality_for_valid_item(
            item=item,
            validation=validation,
            semantic_text=semantic_text,
            has_alignment_material=bool(semantic_text),
            extra_signals={
                "segment_count": len(segments),
                "sample_rate": sample_rate,
                "channels": channels,
            },
        )
        fingerprints = self._fingerprints(
            item=item,
            primary_text=semantic_text,
        )
        if audio_fingerprint is not None:
            fingerprints["audio_chromaprint"] = audio_fingerprint
        return PreprocessedAudio(
            media_id=self._media_id(item=item),
            source_id=item.source_id,
            source_url=item.source_url,
            normalized_url=item.normalized_url,
            domain=item.domain,
            media_path=privacy.media_path,
            mime_type=item.mime_type,
            duration_seconds=duration_seconds,
            transcript_text=transcript_text,
            transcript_language=item.resolved_language()
            or as_optional_text(item.payload.get("transcript_language")),
            transcript_segments=segments,
            quality=quality,
            normalized_audio_path=privacy.media_path,
            sample_rate=sample_rate,
            channels=channels,
            loudness_lufs=loudness_lufs,
            dedupe_fingerprints=fingerprints,
            alignment_signals={
                "transcript_available": bool(transcript_text),
                "segments_available": bool(segments),
                **summarize_timeline(
                    segments=segments,
                    duration_seconds=duration_seconds,
                    payload=item.payload,
                ),
            },
            safety_status="passed",
            privacy_clearance=privacy.clearance,
            privacy_evidence={
                "analysis": privacy.analysis_evidence.to_dict(),
                "residual": (
                    privacy.residual_evidence.to_dict()
                    if privacy.residual_evidence is not None
                    else None
                ),
            },
            timed_segments=canonical_transcript_segments(segments),
            voice_activity_intervals=canonical_time_intervals(
                item.payload.get("voice_activity_intervals")
                or item.payload.get("vad_segments")
            ),
            privacy_intervals=canonical_privacy_intervals(
                item.payload.get("privacy_intervals")
            ),
            snr_db=as_optional_float(item.payload.get("snr_db")),
            speech_ratio=as_optional_float(item.payload.get("speech_ratio")),
            music_ratio=as_optional_float(item.payload.get("music_ratio")),
            overlap_ratio=as_optional_float(item.payload.get("overlap_ratio")),
        )


def _duration_seconds(*, item: PreprocessingInput) -> float | None:
    return item.duration_seconds or as_optional_float(
        payload_field(
            item=item,
            name="audio_duration_seconds",
        )
    )


def _sample_rate(*, item: PreprocessingInput) -> int | None:
    value = as_optional_float(
        payload_field(
            item=item,
            name="audio_sample_rate",
        )
    )
    return int(value) if value is not None else None


def _channels(*, item: PreprocessingInput) -> int | None:
    value = as_optional_float(
        payload_field(
            item=item,
            name="audio_channels",
        )
    )
    return int(value) if value is not None else None


def _has_transcript(*, item: PreprocessingInput) -> bool:
    if as_optional_text(item.transcript_text):
        return True
    if as_optional_text(item.payload.get("transcript_text")):
        return True
    segments = item.payload.get("transcript_segments")
    return isinstance(segments, list) and bool(segments)


def _has_audio_metadata(
    *,
    duration: float | None,
    sample_rate: int | None,
    channels: int | None,
) -> bool:
    return any(
        value is not None for value in (duration, sample_rate, channels)
    )


def _accept_metadata_only_audio(
    *,
    item: PreprocessingInput,
    reason: str,
    signals: dict[str, object],
) -> MediaValidationResult | None:
    if reason not in {
        "too_large",
        "decode_failed",
        "partial_download",
        "file_not_found",
    }:
        return None
    if not is_metadata_fetch_mode(payload=item.payload):
        return None
    if not has_audio_training_metadata(
        payload=item.payload,
        transcript_text=item.transcript_text,
    ):
        return None
    signals.update(
        {
            "metadata_only_accepted": True,
            "metadata_only_reason": reason,
        }
    )
    return accepted_media_result(signals=signals)


def _decode_failed(*, path: Path | None) -> bool | None:
    """Probe only stdlib WAV headers; never pull codec libraries into preprocessing.

    Non-WAV decode checks belong in crawler enrichment. Returning
    None means "unknown" so metadata-only acceptance can still apply.
    """

    if path is None or not path.exists():
        return None
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() <= 0
    except (wave.Error, EOFError, OSError):
        return True


def _segments_text(
    *,
    segments: tuple[dict[str, object], ...],
) -> str | None:
    text = " ".join(
        str(segment.get("text") or "") for segment in segments
    ).strip()
    return text or None
