"""Resolve effective processor flags for curated snapshot builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crawler.curation.snapshots.dataset_assembly.curated_quality_filter import (
    processor_flag,
)


@dataclass(frozen=True, slots=True)
class CuratedProcessorProfile:
    configured_image_ocr: bool
    configured_audio_transcription: bool
    configured_video_transcription: bool
    configured_document_ocr: bool
    effective_image_ocr: bool
    effective_audio_transcription: bool
    effective_video_transcription: bool
    effective_document_ocr: bool


def resolve_curated_processor_profile(
    *,
    processors_payload: dict[str, Any],
    raw_kind_counts: dict[str, int],
) -> CuratedProcessorProfile:
    configured_image_ocr = processor_flag(
        processors_payload, "image", "run_ocr", True
    )
    configured_audio_transcription = processor_flag(
        processors_payload, "audio", "run_transcription", True
    )
    configured_video_transcription = processor_flag(
        processors_payload, "video", "run_transcription", True
    )
    configured_document_ocr = processor_flag(
        processors_payload, "document", "run_ocr", True
    )
    return CuratedProcessorProfile(
        configured_image_ocr=configured_image_ocr,
        configured_audio_transcription=configured_audio_transcription,
        configured_video_transcription=configured_video_transcription,
        configured_document_ocr=configured_document_ocr,
        effective_image_ocr=(
            configured_image_ocr and raw_kind_counts.get("image", 0) > 0
        ),
        effective_audio_transcription=(
            configured_audio_transcription
            and raw_kind_counts.get("audio", 0) > 0
        ),
        effective_video_transcription=(
            configured_video_transcription
            and raw_kind_counts.get("video", 0) > 0
        ),
        effective_document_ocr=(
            configured_document_ocr and raw_kind_counts.get("document", 0) > 0
        ),
    )
