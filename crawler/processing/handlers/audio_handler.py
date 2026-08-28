"""Audio persisting processor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from config.collection.processors import AudioProcessorSettings
from config.environment.default_values import (
    DEFAULT_OPTIONAL_NUMBER_ROUND_DIGITS,
    ENRICHMENT_PREVIEW_MAX_CHARACTERS,
)
from crawler.analysis.enrichment.audio.audio_analyzer import (
    AudioAnalysisResult,
)
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.analysis.enrichment.audio.audio_analyzer import (
        AudioAnalysisResult,
        AudioAnalyzer,
    )
    from crawler.fetching.results.result import FetchResult
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )
    from crawler.storage.datasets.writing.dataset_writer import DatasetWriter

_TRANSCRIPTION_BLOCKED_FETCH_MODES = frozenset(
    (
        "metadata_only",
        "metadata_probe",
        "head_only_oversized",
        "partial_probe_failed_fallback_head_only",
    )
)


class AudioHandler(
    PersistingProcessor[AudioProcessorSettings, AudioAnalysisResult]
):
    """Persisting processor for audio fetch results."""

    def __init__(
        self,
        *,
        settings: AudioProcessorSettings,
        dataset_writer: DatasetWriter,
        logger: ProjectLogger,
        failure_handler: ProcessorFailureHandler,
        analyzer: AudioAnalyzer | None = None,
    ) -> None:
        super().__init__(
            settings=settings,
            dataset_writer=dataset_writer,
            logger=logger,
            failure_handler=failure_handler,
        )
        self._settings: AudioProcessorSettings = settings
        if analyzer is None:
            raise ValueError("AudioHandler requires an injected AudioAnalyzer")
        self._analyzer = analyzer

    async def prepare_analysis(
        self,
        *,
        result: FetchResult,
    ) -> AudioAnalysisResult:
        """Analyze the fetched audio result."""
        return await self._analyzer.analyze(
            result=result,
        )

    async def validate_result(
        self,
        *,
        result: FetchResult,
        analysis: AudioAnalysisResult | None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Validate analyzed audio quality before persistence."""

        if analysis is None:
            raise ValueError("Audio analysis is required for validation")

        accepted, reject_reason, quality_fields = self._evaluate_quality(
            analysis=analysis,
            payload_size=result.body_size,
        )

        transcript_segment_count = len(analysis.transcript_segments)
        transcript_char_count = len(analysis.transcript_text or "")
        transcript_available = bool(analysis.transcript_text)
        metadata_extracted = bool(analysis.metadata)
        host = urlparse(result.final_url).hostname

        quality_fields = {
            **quality_fields,
            "transcript_available": transcript_available,
            "transcript_language": analysis.transcript_language,
            "transcript_segment_count": transcript_segment_count,
        }

        payload = result.payload
        payload_mb = round(max(0, int(result.body_size)) / 1_000_000.0, 1)
        payload_truncated = bool(payload.truncated) if payload else False
        payload_fetch_mode = (
            str(payload.fetch_mode) if payload else "metadata_only"
        )
        duration_seconds = analysis.duration_seconds
        duration_seconds_log = (
            round(duration_seconds, DEFAULT_OPTIONAL_NUMBER_ROUND_DIGITS)
            if duration_seconds is not None
            else None
        )
        transcription_requested = self._should_run_transcription(
            result=result,
        )
        candidate_parts = [
            f"audio | host={host or 'unknown'}",
            f"payload_mb={payload_mb}",
            f"metadata={str(metadata_extracted).lower()}",
            f"transcript={str(transcription_requested).lower()}",
        ]
        if payload_truncated:
            candidate_parts.append("probe=partial")
        candidate_message = " ".join(candidate_parts)

        log_fields = {
            "url": result.final_url,
            "host": host,
            "mime_type": result.mime_type,
            "transcription_requested": transcription_requested,
            "metadata_requested": bool(self._settings.extract_metadata),
            "transcript_available": transcript_available,
            "transcript_language": analysis.transcript_language,
            "transcript_segment_count": transcript_segment_count,
            "transcript_char_count": transcript_char_count,
            "metadata_extracted": metadata_extracted,
            "duration_seconds": duration_seconds_log,
            "sample_rate": analysis.sample_rate,
            "channels": analysis.channels,
            "bitrate": analysis.bitrate,
            "payload_bytes": result.body_size,
            "payload_mb": payload_mb,
            "payload_truncated": payload_truncated,
            "payload_fetch_mode": payload_fetch_mode,
            "payload_observed_bytes": (
                payload.observed_bytes if payload else result.body_size
            ),
            "payload_complete": bool(payload.is_complete_payload)
            if payload
            else False,
            "source_content_length": (
                payload.source_content_length if payload else None
            ),
        }

        if accepted:
            self._logger.info(
                "audio_candidate_analyzed",
                message=candidate_message,
                **log_fields,
                transcript_source=analysis.transcript_source,
                transcript_confidence=analysis.transcript_confidence,
            )
        else:
            self._logger.warning(
                "audio_candidate_rejected",
                **log_fields,
                reject_reason=reject_reason or "quality_rejected",
            )

        return accepted, reject_reason, quality_fields

    async def build_enrichment(
        self,
        *,
        result: FetchResult,
        analysis: AudioAnalysisResult | None,
    ) -> Any:
        """Build persisted enrichment fields for the analyzed audio."""

        if analysis is None:
            raise ValueError("Audio analysis is required for enrichment")

        return self._build_audio_enrichment_fields(
            analysis=analysis,
            run_transcription=self._should_run_transcription(result=result),
        )

    def _should_run_transcription(self, *, result: FetchResult) -> bool:
        if not bool(self._settings.run_transcription):
            return False

        payload = result.payload
        if payload is None:
            return False

        fetch_mode = str(payload.fetch_mode or "").strip().lower()
        if fetch_mode in _TRANSCRIPTION_BLOCKED_FETCH_MODES:
            return False

        if not bool(payload.is_complete_payload):
            return False

        max_bytes = self._settings.max_transcription_bytes
        if max_bytes <= 0:
            return False
        return int(result.body_size or 0) <= max_bytes

    def _evaluate_quality(
        self,
        *,
        analysis: AudioAnalysisResult,
        payload_size: int,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        fields: dict[str, Any] = {
            "payload_bytes": payload_size,
            "quality_modality": "audio",
        }
        if payload_size < self._settings.min_bytes:
            fields["quality_score"] = 0.0
            return False, "audio_too_small", fields

        dur = analysis.duration_seconds
        sr = analysis.sample_rate
        ch = analysis.channels

        fields["audio_duration_seconds"] = dur
        fields["metadata_extracted"] = bool(analysis.metadata)
        fields["audio_metadata_status"] = analysis.metadata_status
        fields["sample_rate"] = sr
        fields["channels"] = ch

        if self._settings.require_metadata_for_acceptance and dur is None:
            fields["quality_score"] = 0.0
            return False, "audio_metadata_missing", fields

        if dur is not None and dur < self._settings.min_duration_seconds:
            fields["quality_score"] = 0.2
            return False, "audio_too_short", fields

        max_duration_seconds = self._optional_float(
            self._settings.max_duration_seconds
        )
        if (
            dur is not None
            and max_duration_seconds is not None
            and dur > max_duration_seconds > 0.0
        ):
            fields["quality_score"] = 0.1
            return False, "audio_too_long", fields

        if sr is not None and sr < self._settings.min_sample_rate:
            fields["quality_score"] = 0.15
            return False, "audio_sample_rate_too_low", fields

        if ch is not None and ch > self._settings.max_channels:
            fields["quality_score"] = 0.15
            return False, "audio_channel_count_unsupported", fields

        quality_score = 0.5
        if dur:
            quality_score += 0.1
        if sr and sr >= 16000:
            quality_score += 0.1
        if analysis.transcript_confidence:
            quality_score += 0.15 * min(1.0, analysis.transcript_confidence)
        fields["quality_score"] = max(0.0, min(1.0, quality_score))
        return True, None, fields

    def _build_audio_enrichment_fields(
        self,
        *,
        analysis: AudioAnalysisResult,
        run_transcription: bool,
    ) -> dict[str, object]:
        payload: dict[str, object] = {}

        if bool(self._settings.extract_metadata):
            removed_generic_fields = {
                "duration_seconds",
                "sample_rate",
                "channels",
                "bitrate",
            }
            payload.update(
                {
                    key: value
                    for key, value in analysis.metadata.items()
                    if value is not None and key not in removed_generic_fields
                }
            )
            payload.update(
                {
                    key: value
                    for key, value in {
                        "audio_duration_seconds": analysis.duration_seconds,
                        "audio_sample_rate": analysis.sample_rate,
                        "audio_channels": analysis.channels,
                        "audio_bitrate": analysis.bitrate,
                    }.items()
                    if value is not None
                }
            )

        if run_transcription:
            payload["transcription_status"] = analysis.transcription_status
            payload["transcript_available"] = bool(
                analysis.transcript_text and analysis.transcript_text.strip()
            )
            if analysis.transcription_provenance is not None:
                payload["transcription_provenance"] = dict(
                    analysis.transcription_provenance
                )
            if analysis.transcript_text and analysis.transcript_text.strip():
                payload["transcript_text"] = analysis.transcript_text
                payload["transcript_preview"] = analysis.transcript_text[
                    :ENRICHMENT_PREVIEW_MAX_CHARACTERS
                ]
                payload["transcript_confidence"] = (
                    analysis.transcript_confidence
                )
                payload["transcript_source"] = analysis.transcript_source
                payload["transcript_language"] = analysis.transcript_language
                payload["transcript_segments"] = list(
                    analysis.transcript_segments
                )
                payload["transcript_quality_score"] = (
                    analysis.transcript_confidence
                )

        payload.update(
            self._audio_speaker_enrichment_fields(analysis=analysis)
        )
        payload.update(self._audio_event_enrichment_fields(analysis=analysis))
        payload.update(
            self._audio_prosody_enrichment_fields(analysis=analysis)
        )

        payload["audio_pipeline_metrics"] = {
            "decode_success": bool(analysis.metadata),
            "transcription_requested": run_transcription,
            "transcription_success": (
                payload.get("transcription_status") == "success"
            ),
            "diarization_success": bool(
                payload.get("speaker_diarization_available")
            ),
            "generation_targets": 0,
            "rejection_reasons": [],
            "duration_seconds": payload.get("audio_duration_seconds"),
            "quality_score": None,
        }
        return payload

    @staticmethod
    def _audio_speaker_enrichment_fields(
        *,
        analysis: AudioAnalysisResult,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "speaker_segments": list(analysis.speaker_segments),
            "speaker_diarization_available": bool(analysis.speaker_segments),
            "speaker_diarization_status": (
                analysis.speaker_diarization_status
            ),
            "speaker_count": analysis.speaker_count,
            "overlapping_speech": analysis.overlapping_speech,
        }
        if analysis.speaker_diarization_model_name:
            payload["speaker_diarization_model_name"] = (
                analysis.speaker_diarization_model_name
            )
        if analysis.speaker_diarization_model_version:
            payload["speaker_diarization_model_version"] = (
                analysis.speaker_diarization_model_version
            )
        return payload

    @staticmethod
    def _audio_event_enrichment_fields(
        *,
        analysis: AudioAnalysisResult,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "sound_events": list(analysis.sound_events),
            "audio_event_analysis_status": (
                analysis.audio_event_analysis_status
            ),
            "audio_event_analysis_reasons": list(
                analysis.audio_event_analysis_reasons
            ),
        }
        for field_name, value in (
            ("background_noise_label", analysis.background_noise_label),
            ("acoustic_scene_label", analysis.acoustic_scene_label),
            ("sound_label", analysis.sound_label),
            ("audio_event_confidence", analysis.audio_event_confidence),
            ("audio_event_model_name", analysis.audio_event_model_name),
        ):
            if value is not None:
                payload[field_name] = value
        return payload

    @staticmethod
    def _audio_prosody_enrichment_fields(
        *,
        analysis: AudioAnalysisResult,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "audio_emotion_analysis_status": (
                analysis.audio_emotion_analysis_status
            ),
            "audio_emotion_analysis_reasons": list(
                analysis.audio_emotion_analysis_reasons
            ),
        }
        for field_name, value in (
            ("prosody", analysis.prosody),
            ("emotion_label", analysis.emotion_label),
            ("emotion_confidence", analysis.emotion_confidence),
            ("arousal", analysis.arousal),
            ("valence", analysis.valence),
            ("dominance", analysis.dominance),
            ("audio_emotion_model_name", analysis.audio_emotion_model_name),
        ):
            if value is not None:
                payload[field_name] = value
        return payload

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (float, int)):
            return float(value)
        try:
            return float(str(value).strip())
        except ValueError:
            return None
