"""Synchronous, provenance-preserving speech transcription service."""

from __future__ import annotations

import math
from collections.abc import Buffer, Callable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    SupportsFloat,
    SupportsIndex,
    cast,
)

from logger.project_logger import ProjectLogger
from preprocessing.media.speech.transcript_quality import (
    score_transcript_quality,
    training_label_rules,
)
from preprocessing.media.speech.transcription_result import (
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
)
from preprocessing.provenance import (
    ProducerProvenance,
    ProducerType,
    hash_file,
    hash_parameters,
    hash_text,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from config.preprocessing.media_settings import TranscriptionSettings
    from preprocessing.media.adapters.whisper_model_loader import (
        WhisperModelLoader,
    )

_TRANSCRIPTION_RECOVERABLE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    IndexError,
    EOFError,
)


class SpeechTranscriber:
    """Create derived ASR artifacts through the repository-owned backend."""

    def __init__(
        self,
        *,
        settings: TranscriptionSettings,
        model_repository: WhisperModelLoader | None = None,
        logger: ProjectLogger,
        audio_stream_status: Callable[[Path], str] | None = None,
    ) -> None:
        if settings.enabled and model_repository is None:
            raise ValueError(
                "enabled transcription requires a WhisperModelLoader"
            )
        self._settings = settings
        self._model_repository = model_repository
        self._logger = logger
        self._audio_stream_status = audio_stream_status

    @property
    def enabled(self) -> bool:
        return bool(self._settings.enabled)

    @property
    def max_audio_duration_seconds(self) -> float:
        return float(self._settings.max_audio_duration_seconds)

    def _transcribe_sync(
        self,
        *,
        media_path: Path,
        language: str | None = None,
    ) -> TranscriptionResult | None:
        """Return a normalized result, converting backend failures to warnings."""

        if not self.enabled:
            self._logger.info(
                "speech_transcription_disabled", media_path=str(media_path)
            )
            return None
        if not media_path.exists() or not media_path.is_file():
            self._logger.warning(
                "speech_transcription_media_missing",
                media_path=str(media_path),
            )
            return None
        if media_path.stat().st_size <= 0:
            self._logger.warning(
                "speech_transcription_media_empty",
                media_path=str(media_path),
            )
            return None
        if (
            self._audio_stream_status is not None
            and self._audio_stream_status(media_path) == "missing"
        ):
            self._logger.warning(
                "speech_transcription_skipped_no_audio_stream",
                media_path=str(media_path),
            )
            return None
        decode_settings = _transcribe_kwargs(
            settings=self._settings,
            language=language,
        )
        try:
            source_hash = hash_file(media_path)
            repository = self._require_repository()
            segments, info = repository.transcribe(
                media_path,
                **decode_settings,
            )
            result = self._normalize_result(
                media_path=media_path,
                source_hash=source_hash,
                segments=segments,
                info=info,
                decode_settings=decode_settings,
                expected_language=language or self._settings.language,
            )
        except _TRANSCRIPTION_RECOVERABLE_ERRORS as exc:
            self._logger.warning(
                "speech_transcription_failed",
                media_path=str(media_path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None

        if result is None:
            return None
        self._logger.info(
            "speech_transcription_completed",
            media_path=str(media_path),
            text_chars=len(result.text),
            segment_count=len(result.segments),
            language=result.language,
            training_label_eligible=result.training_label_eligible,
        )
        return result

    async def transcribe(
        self,
        *,
        media_path: Path,
        language: str | None = None,
    ) -> TranscriptionResult | None:
        """Run transcription off the event loop with a bounded timeout."""

        import asyncio

        from config.environment.default_values import (
            DEFAULT_MEDIA_ANALYSIS_TIMEOUT_SECONDS,
        )

        timeout = max(
            DEFAULT_MEDIA_ANALYSIS_TIMEOUT_SECONDS * 5,
            float(self.max_audio_duration_seconds or 600) * 1.5,
        )
        return await asyncio.wait_for(
            asyncio.to_thread(
                self._transcribe_sync,
                media_path=media_path,
                language=language,
            ),
            timeout=timeout,
        )

    async def transcribe_if_allowed(
        self,
        *,
        media_path: Path,
        run_transcription: bool,
        duration_seconds: float | None,
        max_duration_seconds: float,
        language: str | None = None,
    ) -> TranscriptionResult | None:
        if not self.enabled:
            return None
        if not self._should_transcribe(
            run_transcription=run_transcription,
            duration_seconds=duration_seconds,
            max_duration_seconds=self._effective_max_duration(
                max_duration_seconds=max_duration_seconds,
            ),
        ):
            return None
        return await self.transcribe(media_path=media_path, language=language)

    @staticmethod
    def _should_transcribe(
        *,
        run_transcription: bool,
        duration_seconds: float | None,
        max_duration_seconds: float,
    ) -> bool:
        if not run_transcription:
            return False
        if duration_seconds is None:
            return True
        if max_duration_seconds <= 0.0:
            return True
        return duration_seconds <= max_duration_seconds

    def _effective_max_duration(self, *, max_duration_seconds: float) -> float:
        service_max = self.max_audio_duration_seconds
        if max_duration_seconds <= 0.0:
            return service_max
        if service_max <= 0.0:
            return max_duration_seconds
        return min(max_duration_seconds, service_max)

    def _normalize_result(
        self,
        *,
        media_path: Path,
        source_hash: str,
        segments: Iterable[Any],
        info: Any,
        decode_settings: dict[str, object],
        expected_language: str | None,
    ) -> TranscriptionResult | None:
        info_value = cast("Any", info)
        language = _optional_text(getattr(info_value, "language", None))
        language_probability = coerce_float(
            getattr(info_value, "language_probability", None)
        )
        parameters_hash = hash_parameters(decode_settings)
        repository = self._require_repository()
        repository_warnings = _repository_warnings(repository)
        normalized = self._normalize_segments(
            media_path=media_path,
            source_hash=source_hash,
            segments=segments,
            language=language,
            parameters_hash=parameters_hash,
            repository_warnings=repository_warnings,
        )

        if not normalized:
            self._logger.warning(
                "speech_transcription_empty", media_path=str(media_path)
            )
            # Create a valid minimal provenance for the explicit empty case (use real hashes)
            empty_params_hash = hash_parameters({"result": "empty_transcript"})
            empty_output_hash = hash_text("")
            empty_provenance = ProducerProvenance(
                producer_type=ProducerType.EXTERNAL_TOOL,
                producer_name="empty_transcript_placeholder",
                producer_version="1.0",
                parameters_hash=empty_params_hash,
                source_hash=source_hash or hash_text("unknown"),
                output_hash=empty_output_hash,
                model_id=None,
                model_revision=None,
            )
            # return a result with empty status; do not lose the media object
            # caller must retain original media asset
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=None,
                language_probability=None,
                segments=(),
                provenance=empty_provenance,
                avg_logprob=None,
                no_speech_probability=None,
                compression_ratio=None,
                training_label_eligible=False,
                label_weight=0.0,
                reject_reason="empty_transcript",
                transcription_status="empty",
                audio_extraction_status="success",
                video_download_status="success",
            )

        joined = " ".join(segment.text for segment in normalized).strip()
        avg_logprob = _average(
            tuple(segment.avg_logprob for segment in normalized)
        )
        no_speech_probability = _average(
            tuple(segment.no_speech_probability for segment in normalized)
        )
        compression_ratio = _average(
            tuple(segment.compression_ratio for segment in normalized)
        )
        quality, reject_reason = score_transcript_quality(
            transcript_text=joined,
            transcript_segments=normalized,
            avg_logprob=avg_logprob,
            no_speech_probability=no_speech_probability,
            compression_ratio=compression_ratio,
            language_probability=language_probability,
            audio_duration=coerce_float(getattr(info_value, "duration", None)),
            language=language,
            expected_language=expected_language,
        )
        eligible, label_weight = training_label_rules(
            quality_score=quality,
            reject_reason=reject_reason,
            minimum_quality=self._settings.minimum_label_quality,
        )
        finalized_segments = tuple(
            _apply_result_eligibility(
                segment=segment, result_eligible=eligible
            )
            for segment in normalized
        )
        output_hash = hash_parameters(
            {
                "text": joined,
                "language": language,
                "segments": [
                    {
                        "text": segment.text,
                        "start_seconds": segment.start_seconds,
                        "end_seconds": segment.end_seconds,
                    }
                    for segment in finalized_segments
                ],
            }
        )
        warnings = repository_warnings + (
            (reject_reason,) if reject_reason is not None else ()
        )
        provenance = repository.provenance(
            parameters_hash=parameters_hash,
            source_hash=source_hash,
            output_hash=output_hash,
            confidence=quality,
            warnings=warnings,
        )
        return TranscriptionResult(
            text=joined,
            confidence=quality,
            language=language,
            language_probability=language_probability,
            segments=finalized_segments,
            provenance=provenance,
            avg_logprob=avg_logprob,
            no_speech_probability=no_speech_probability,
            compression_ratio=compression_ratio,
            training_label_eligible=eligible,
            label_weight=label_weight,
            reject_reason=reject_reason,
            decode_settings=dict(decode_settings),
            transcription_status="success",
            audio_extraction_status="success",
            video_download_status="success",
        )

    def _normalize_segments(
        self,
        *,
        media_path: Path,
        source_hash: str,
        segments: Iterable[Any],
        language: str | None,
        parameters_hash: str,
        repository_warnings: tuple[str, ...],
    ) -> list[TranscriptSegment]:
        normalized: list[TranscriptSegment] = []
        for segment in segments:
            segment_value = cast("Any", segment)
            text = " ".join(str(getattr(segment_value, "text", "")).split())
            if not text:
                continue
            start_seconds = coerce_float(
                getattr(segment_value, "start", None), default=0.0
            )
            end_seconds = coerce_float(
                getattr(segment_value, "end", None), default=start_seconds
            )
            if start_seconds is None or end_seconds is None:
                continue
            if start_seconds < 0.0 or end_seconds < start_seconds:
                self._logger.warning(
                    "speech_transcription_invalid_segment_timing",
                    media_path=str(media_path),
                )
                continue
            avg_logprob = coerce_float(
                getattr(segment_value, "avg_logprob", None)
            )
            no_speech_probability = coerce_float(
                getattr(segment_value, "no_speech_prob", None)
            )
            confidence = _segment_confidence(avg_logprob=avg_logprob)
            eligible = (
                confidence is None
                or confidence >= self._settings.minimum_label_quality
            ) and (
                no_speech_probability is None or no_speech_probability <= 0.8
            )
            weight = (
                confidence
                if eligible and confidence is not None
                else (1.0 if eligible else 0.0)
            )
            output_hash = hash_parameters(
                {
                    "text": text,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "language": language,
                }
            )
            provenance = self._require_repository().provenance(
                parameters_hash=parameters_hash,
                source_hash=source_hash,
                output_hash=output_hash,
                confidence=confidence,
                warnings=repository_warnings,
            )
            normalized.append(
                TranscriptSegment(
                    text=text,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    language=language,
                    avg_logprob=avg_logprob,
                    no_speech_probability=no_speech_probability,
                    confidence=confidence,
                    provenance=provenance,
                    training_label_eligible=eligible,
                    label_weight=weight,
                    compression_ratio=coerce_float(
                        getattr(segment_value, "compression_ratio", None)
                    ),
                    words=_normalize_words(segment=segment_value),
                )
            )
        return normalized

    def _require_repository(self) -> WhisperModelLoader:
        repository = self._model_repository
        if repository is None:
            raise RuntimeError("transcription repository is unavailable")
        return repository


def _segment_confidence(*, avg_logprob: float | None) -> float | None:
    if avg_logprob is None:
        return None
    return round(max(0.0, min(1.0, math.exp(avg_logprob))), 4)


def _transcribe_kwargs(
    *,
    settings: TranscriptionSettings,
    language: str | None,
) -> dict[str, object]:
    resolved_language = (language or settings.language or "").strip() or None
    kwargs: dict[str, object] = {
        "beam_size": settings.beam_size,
        "language": resolved_language,
        "vad_filter": settings.vad_filter,
        "word_timestamps": settings.word_timestamps,
        "temperature": settings.temperature,
        "condition_on_previous_text": settings.condition_on_previous_text,
    }
    if settings.vad_filter:
        kwargs["vad_parameters"] = {
            "min_silence_duration_ms": settings.vad_min_silence_duration_ms,
        }
    return kwargs


def _normalize_words(*, segment: Any) -> tuple[TranscriptWord, ...]:
    words: list[TranscriptWord] = []
    for raw_word in getattr(segment, "words", None) or ():
        text = str(getattr(raw_word, "word", "")).strip()
        start = coerce_float(getattr(raw_word, "start", None))
        end = coerce_float(getattr(raw_word, "end", None))
        if (
            not text
            or start is None
            or end is None
            or start < 0.0
            or end < start
        ):
            continue
        words.append(
            TranscriptWord(
                text=text,
                start_seconds=start,
                end_seconds=end,
                confidence=coerce_float(
                    getattr(raw_word, "probability", None)
                ),
            )
        )
    return tuple(words)


def _apply_result_eligibility(
    *,
    segment: TranscriptSegment,
    result_eligible: bool,
) -> TranscriptSegment:
    eligible = result_eligible and segment.training_label_eligible
    return TranscriptSegment(
        text=segment.text,
        start_seconds=segment.start_seconds,
        end_seconds=segment.end_seconds,
        language=segment.language,
        avg_logprob=segment.avg_logprob,
        no_speech_probability=segment.no_speech_probability,
        confidence=segment.confidence,
        provenance=segment.provenance,
        training_label_eligible=eligible,
        label_weight=segment.label_weight if eligible else 0.0,
        compression_ratio=segment.compression_ratio,
        speaker_id=segment.speaker_id,
        words=segment.words,
    )


def _repository_warnings(
    repository: WhisperModelLoader,
) -> tuple[str, ...]:
    report = repository.report
    warnings: list[str] = []
    if report.model_revision is None:
        warnings.append("unpinned_model_revision")
    if report.artifact_hash is None:
        warnings.append("unverified_model_artifact")
    if report.backend_version == "unavailable":
        warnings.append("unreported_backend_version")
    return tuple(warnings)


def _average(values: tuple[float | None, ...]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def coerce_float(
    value: object,
    *,
    default: float | None = None,
    allow_bool: bool = False,
) -> float | None:
    """Coerce backend metadata without accepting booleans as numbers."""

    if value is None:
        return default
    if isinstance(value, bool):
        return float(value) if allow_bool else default
    try:
        return float(cast(str | Buffer | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError):
        return default
