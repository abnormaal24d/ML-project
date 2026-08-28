"""Semantic video-analysis status labels for handlers and curation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from config.collection.processors import VideoProcessorSettings
    from crawler.analysis.enrichment.video.video_analysis_result import (
        VideoAnalysisResult,
    )
    from crawler.fetching.results.result import FetchResult

SEMANTIC_VIDEO_READY = "semantic_analysis_ready"
SEMANTIC_VIDEO_SKIPPED_FETCH_MODE = "semantic_analysis_skipped_fetch_mode"
SEMANTIC_VIDEO_SKIPPED_SIZE = "semantic_analysis_skipped_size"
SEMANTIC_VIDEO_MISSING_TEXT_OR_KEYFRAMES = "missing_keyframes_or_text"

_METADATA_FETCH_MODES = frozenset(
    {
        "metadata_only",
        "metadata_probe",
        "head_only_oversized",
        "partial_probe_failed_fallback_head_only",
    }
)


@dataclass(frozen=True, slots=True)
class VideoSemanticStatus:
    """Semantic-readiness status stored with persisted video enrichment."""

    status: str
    reason: str | None

    @property
    def ready(self) -> bool:
        return self.status == SEMANTIC_VIDEO_READY


def resolve_video_semantic_status(
    *,
    result: FetchResult,
    analysis: VideoAnalysisResult,
    settings: VideoProcessorSettings,
) -> VideoSemanticStatus:
    """Explain whether a video can produce semantic training pairs.

    settings (VideoProcessorSettings or similar): expected to have
    max_full_analysis_bytes, max_frame_analysis_bytes etc. for limit checks.
    """

    if _has_text_or_keyframes(analysis=analysis):
        return VideoSemanticStatus(
            status=SEMANTIC_VIDEO_READY,
            reason=None,
        )

    payload = result.payload
    fetch_mode = _normalize(payload.fetch_mode) if payload is not None else ""
    if fetch_mode in _METADATA_FETCH_MODES or not (
        payload is not None and payload.is_complete_payload
    ):
        _logger.debug(
            "video_semantic_skipped_fetch_mode",
            extra={"fetch_mode": fetch_mode},
        )
        return VideoSemanticStatus(
            status="semantic_analysis_skipped",
            reason=SEMANTIC_VIDEO_SKIPPED_FETCH_MODE,
        )

    payload_size = 0
    if payload is not None:
        payload_size = (
            payload.observed_bytes
            if payload.observed_bytes is not None
            else payload.byte_size
        )
    if _exceeds_any_semantic_limit(
        payload_size=payload_size,
        settings=settings,
    ):
        _logger.debug(
            "video_semantic_skipped_size",
            extra={"payload_size": payload_size},
        )
        return VideoSemanticStatus(
            status="semantic_analysis_skipped",
            reason=SEMANTIC_VIDEO_SKIPPED_SIZE,
        )

    return VideoSemanticStatus(
        status="semantic_analysis_incomplete",
        reason=SEMANTIC_VIDEO_MISSING_TEXT_OR_KEYFRAMES,
    )


def persisted_video_reject_reason(
    *,
    enrichment: Mapping[str, object],
    fallback: str = SEMANTIC_VIDEO_MISSING_TEXT_OR_KEYFRAMES,
) -> str:
    """Return the semantic reject reason stored by the video handler."""

    reason = _normalize(enrichment.get("semantic_video_analysis_reason"))
    if reason:
        return reason
    status = _normalize(enrichment.get("semantic_video_analysis_status"))
    if status in {"semantic_analysis_skipped", "semantic_analysis_incomplete"}:
        return fallback
    return fallback


def _has_text_or_keyframes(*, analysis: VideoAnalysisResult) -> bool:
    transcript_text = (
        analysis.transcription.text
        if analysis.transcription is not None
        else None
    )
    frame_ocr_text = (
        analysis.frame_ocr.text if analysis.frame_ocr is not None else None
    )
    return bool(
        _normalize(transcript_text)
        or _normalize(frame_ocr_text)
        or analysis.keyframes
    )


def _exceeds_any_semantic_limit(
    *, payload_size: int, settings: VideoProcessorSettings
) -> bool:
    for limit_name, limit_bytes in (
        ("max_full_analysis_bytes", settings.max_full_analysis_bytes),
        ("max_frame_analysis_bytes", settings.max_frame_analysis_bytes),
        ("max_transcription_bytes", settings.max_transcription_bytes),
    ):
        if payload_size > limit_bytes > 0:
            _logger.debug(
                "video_semantic_limit_exceeded",
                extra={
                    "limit_name": limit_name,
                    "limit": limit_bytes,
                    "payload_size": payload_size,
                },
            )
            return True
    return False


def _normalize(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split()).lower()
