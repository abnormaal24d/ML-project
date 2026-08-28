"""Video persisting processor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from config.collection.processors import VideoProcessorSettings
from config.environment.default_values import (
    DEFAULT_OPTIONAL_NUMBER_ROUND_DIGITS,
)
from crawler.analysis.enrichment.video.video_analysis_result import (
    VideoAnalysisResult,
)
from crawler.analysis.enrichment.video.video_enrichment_payload import (
    build_video_enrichment_payload,
)
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.analysis.enrichment.video.video_analyzer import VideoAnalyzer
    from crawler.fetching.results.result import FetchResult
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )
    from crawler.storage.datasets.writing.dataset_writer import DatasetWriter

_HEAD_ONLY_METADATA_STATUSES = frozenset(
    {
        "embed_metadata",
        "head_only_oversized",
        "partial_probe_failed_fallback_head_only",
    }
)


class VideoHandler(
    PersistingProcessor[VideoProcessorSettings, VideoAnalysisResult]
):
    """Persisting processor for video fetch results."""

    def __init__(
        self,
        *,
        settings: VideoProcessorSettings,
        dataset_writer: DatasetWriter,
        logger: ProjectLogger,
        failure_handler: ProcessorFailureHandler,
        analyzer: VideoAnalyzer,
    ) -> None:
        super().__init__(
            settings=settings,
            dataset_writer=dataset_writer,
            logger=logger,
            failure_handler=failure_handler,
        )
        self._settings: VideoProcessorSettings = settings
        self._analyzer = analyzer

    async def prepare_analysis(
        self,
        *,
        result: FetchResult,
    ) -> VideoAnalysisResult:
        """Analyze the fetched video result."""
        return await self._analyzer.analyze(
            result=result,
        )

    async def validate_result(
        self,
        *,
        result: FetchResult,
        analysis: VideoAnalysisResult | None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Validate analyzed video quality before persistence."""

        if analysis is None:
            raise ValueError("Video analysis is required for validation")

        accepted, reject_reason, quality_fields = self._evaluate_quality(
            analysis=analysis,
            payload_size=self._resolve_payload_size(result=result),
        )
        frame_ocr_requested = self._should_run_frame_ocr(result=result)
        quality_fields = {
            **quality_fields,
            "transcript_available": bool(analysis.transcript_text),
            "transcript_language": analysis.transcript_language,
            "transcript_segment_count": len(analysis.transcript_segments),
            "keyframe_count": len(analysis.keyframes),
            "frame_ocr_requested": frame_ocr_requested,
            "frame_ocr_available": bool(analysis.frame_ocr_text),
        }
        payload = result.payload
        payload_mb = round(max(0, int(result.body_size)) / 1_000_000.0, 1)
        payload_truncated = bool(payload.truncated) if payload else False
        payload_fetch_mode = (
            str(payload.fetch_mode) if payload else "metadata_only"
        )
        payload_observed_bytes = (
            payload.observed_bytes if payload else result.body_size
        )
        payload_complete = (
            bool(payload.is_complete_payload) if payload else False
        )
        source_content_length = (
            payload.source_content_length if payload else None
        )
        transcript_char_count = len(analysis.transcript_text or "")
        frame_ocr_char_count = len(analysis.frame_ocr_text or "")
        host = urlparse(result.final_url).hostname
        transcription_requested = self._should_run_transcription(result=result)
        keyframes_requested = (
            bool(self._settings.keyframes.enabled)
            and self._has_complete_media_payload(result=result)
            and self._is_within_analysis_limit(
                result=result,
                limit_bytes=self._settings.max_full_analysis_bytes,
            )
            and self._is_within_analysis_limit(
                result=result,
                limit_bytes=self._settings.max_frame_analysis_bytes,
            )
        )
        duration_seconds_log = (
            round(
                analysis.duration_seconds,
                DEFAULT_OPTIONAL_NUMBER_ROUND_DIGITS,
            )
            if analysis.duration_seconds is not None
            else None
        )
        fps_log = (
            round(analysis.fps, DEFAULT_OPTIONAL_NUMBER_ROUND_DIGITS)
            if analysis.fps is not None
            else None
        )
        candidate_parts = [
            f"video | host={host or 'unknown'}",
            f"payload_mb={payload_mb}",
        ]
        if analysis.metadata_status:
            candidate_parts.append(
                f"metadata_status={analysis.metadata_status}"
            )
        else:
            candidate_parts.append(
                f"metadata={str(bool(analysis.metadata)).lower()}"
            )
        candidate_parts.append(
            f"transcript={str(transcription_requested).lower()}"
        )
        candidate_parts.append(f"ocr={str(frame_ocr_requested).lower()}")
        if payload_truncated:
            candidate_parts.append("probe=partial")
        candidate_message = " ".join(candidate_parts)

        payload_log_fields = {
            "payload_bytes": result.body_size,
            "payload_mb": payload_mb,
            "payload_truncated": payload_truncated,
            "payload_fetch_mode": payload_fetch_mode,
            "payload_observed_bytes": payload_observed_bytes,
            "payload_complete": payload_complete,
            "source_content_length": source_content_length,
        }
        if accepted:
            self._logger.info(
                "video_candidate_analyzed",
                message=candidate_message,
                url=result.final_url,
                host=host,
                mime_type=result.mime_type,
                **payload_log_fields,
                transcription_requested=transcription_requested,
                frame_ocr_requested=frame_ocr_requested,
                keyframes_requested=keyframes_requested,
                metadata_requested=self._settings.extract_metadata,
                transcript_available=bool(analysis.transcript_text),
                transcript_language=analysis.transcript_language,
                transcript_segment_count=len(analysis.transcript_segments),
                transcript_char_count=transcript_char_count,
                transcript_source=analysis.transcript_source,
                transcript_confidence=analysis.transcript_confidence,
                duration_seconds=duration_seconds_log,
                width=analysis.width,
                height=analysis.height,
                fps=fps_log,
                frame_count=analysis.frame_count,
                keyframe_count=len(analysis.keyframes),
                frame_ocr_available=bool(analysis.frame_ocr_text),
                frame_ocr_char_count=frame_ocr_char_count,
                metadata_extracted=bool(analysis.metadata),
                metadata_status=analysis.metadata_status,
            )
        else:
            self._logger.warning(
                "video_candidate_rejected",
                url=result.final_url,
                host=host,
                mime_type=result.mime_type,
                **payload_log_fields,
                reject_reason=reject_reason or "quality_rejected",
                transcription_requested=transcription_requested,
                frame_ocr_requested=frame_ocr_requested,
                keyframes_requested=keyframes_requested,
                metadata_requested=self._settings.extract_metadata,
                transcript_available=bool(analysis.transcript_text),
                transcript_language=analysis.transcript_language,
                transcript_segment_count=len(analysis.transcript_segments),
                transcript_char_count=transcript_char_count,
                metadata_extracted=bool(analysis.metadata),
                metadata_status=analysis.metadata_status,
                duration_seconds=duration_seconds_log,
                width=analysis.width,
                height=analysis.height,
                fps=fps_log,
                frame_count=analysis.frame_count,
                keyframe_count=len(analysis.keyframes),
                frame_ocr_available=bool(analysis.frame_ocr_text),
                frame_ocr_char_count=frame_ocr_char_count,
            )
        return accepted, reject_reason, quality_fields

    async def build_enrichment(
        self,
        *,
        result: FetchResult,
        analysis: VideoAnalysisResult | None,
    ) -> dict[str, object]:
        """Build persisted enrichment fields for the analyzed video."""

        if analysis is None:
            raise ValueError("Video analysis is required for enrichment")

        return build_video_enrichment_payload(
            analysis=analysis,
            result=result,
            settings=self._settings,
            extract_metadata=self._settings.extract_metadata,
            run_transcription=self._should_run_transcription(result=result),
        )

    @staticmethod
    def _resolve_payload_size(*, result: FetchResult) -> int:
        """Return the persisted payload size for the fetch result."""
        payload = result.payload
        if payload is None:
            return 0

        fetch_mode = str(payload.fetch_mode or "").strip().lower()
        if fetch_mode in {
            "head_only_oversized",
            "partial_probe_failed_fallback_head_only",
        }:
            return int(payload.observed_bytes or 0)

        return payload.byte_size

    def _should_run_transcription(self, *, result: FetchResult) -> bool:
        return (
            bool(self._settings.run_transcription)
            and self._settings.generate_transcriptions
            and self._settings.extract_audio_track
            and self._has_complete_media_payload(result=result)
            and self._is_within_analysis_limit(
                result=result,
                limit_bytes=self._settings.max_full_analysis_bytes,
            )
            and self._is_within_analysis_limit(
                result=result,
                limit_bytes=self._settings.max_transcription_bytes,
            )
        )

    def _should_run_frame_ocr(self, *, result: FetchResult) -> bool:
        return (
            bool(self._settings.run_ocr)
            and self._has_complete_media_payload(result=result)
            and self._is_within_analysis_limit(
                result=result,
                limit_bytes=self._settings.max_full_analysis_bytes,
            )
            and self._is_within_analysis_limit(
                result=result,
                limit_bytes=self._settings.max_frame_analysis_bytes,
            )
        )

    def _is_within_analysis_limit(
        self,
        *,
        result: FetchResult,
        limit_bytes: int,
    ) -> bool:
        if limit_bytes <= 0:
            return False

        return self._resolve_payload_size(result=result) <= limit_bytes

    @staticmethod
    def _has_complete_media_payload(*, result: FetchResult) -> bool:
        payload = result.payload
        if payload is None:
            return False

        fetch_mode = str(payload.fetch_mode or "").strip().lower()
        if fetch_mode in {
            "embed_metadata",
            "metadata_only",
            "metadata_probe",
            "head_only_oversized",
            "partial_probe_failed_fallback_head_only",
        }:
            return False

        return bool(payload.is_complete_payload)

    def _evaluate_quality(
        self,
        *,
        analysis: VideoAnalysisResult,
        payload_size: int,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        fields: dict[str, Any] = {
            "payload_bytes": payload_size,
            "quality_modality": "video",
        }
        fields["metadata_extracted"] = bool(analysis.metadata)
        fields["metadata_status"] = analysis.metadata_status
        fields["video_duration_seconds"] = analysis.duration_seconds
        fields["video_width"] = analysis.width
        fields["video_height"] = analysis.height
        if analysis.metadata_status == "analysis_timeout":
            fields["quality_score"] = 0.0
            return False, "video_analysis_timeout", fields
        if analysis.metadata_status in _HEAD_ONLY_METADATA_STATUSES:
            fields["quality_score"] = 0.5
            fields["metadata_only_record"] = True
            return True, None, fields
        if payload_size < self._settings.min_bytes:
            fields["quality_score"] = 0.0
            return False, "video_too_small", fields
        if (
            self._settings.require_metadata_for_acceptance
            and not analysis.metadata
        ):
            fields["quality_score"] = 0.0
            return False, "video_metadata_missing", fields
        if (
            analysis.duration_seconds is not None
            and self._settings.min_duration_seconds > 0.0
            and analysis.duration_seconds < self._settings.min_duration_seconds
        ):
            fields["quality_score"] = 0.2
            return False, "video_too_short", fields
        if (
            analysis.width is not None
            and analysis.width < self._settings.min_width
        ):
            fields["quality_score"] = 0.2
            return False, "video_width_too_small", fields
        if (
            analysis.height is not None
            and analysis.height < self._settings.min_height
        ):
            fields["quality_score"] = 0.2
            return False, "video_height_too_small", fields

        if (
            analysis.duration_seconds is not None
            and analysis.duration_seconds > self._settings.max_duration_seconds
        ):
            fields["quality_score"] = 0.1
            return False, "video_too_long", fields
        if analysis.fps is not None and analysis.fps < self._settings.min_fps:
            fields["quality_score"] = 0.2
            return False, "video_fps_too_low", fields
        if analysis.width and analysis.height:
            aspect_ratio = max(analysis.width, analysis.height) / max(
                1, min(analysis.width, analysis.height)
            )
            if aspect_ratio > self._settings.max_aspect_ratio:
                fields["quality_score"] = 0.15
                return False, "video_extreme_aspect_ratio", fields
        if (
            analysis.transcript_confidence is not None
            and analysis.transcript_confidence < 0.4
        ):
            fields["video_transcript_low_confidence"] = True
        fields["quality_score"] = 0.85 if analysis.duration_seconds else 0.55
        return True, None, fields
