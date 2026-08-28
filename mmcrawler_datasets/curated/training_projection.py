"""Training-owned projections of canonical curated timed-media records."""

from __future__ import annotations

from dataclasses import dataclass

from mmcrawler_datasets.curated.evidence import AssetContextRecord
from mmcrawler_datasets.curated.timed_media import (
    CuratedAudioRecord,
    CuratedVideoRecord,
    TranscriptSegment,
)
from preprocessing.privacy.clearance import PrivacyClearance


@dataclass(frozen=True, slots=True)
class TrainingTimedMediaInput:
    """Consumer-specific fields required to assemble training samples."""

    media_id: str
    media_path: str
    transcript_text: str | None
    source_url: str
    allow_training: bool | None
    parent_document_id: str | None
    modality: str
    domain: str
    page_title: str | None
    transcript_preview: str | None
    surrounding_text: str | None
    html_context: str | None
    media_mime_type: str | None
    transcript_segments: tuple[TranscriptSegment, ...]
    license: str | None
    license_url: str | None
    governance_note: str | None
    robots_status: str | None
    terms_source: str | None
    usage_rules: str | None
    asset_fetch_mode: str | None
    is_complete_payload: bool
    near_duplicate_cluster_id: str | None
    media_fingerprint: str | None
    privacy_clearance: PrivacyClearance
    asset_context: AssetContextRecord
    language: str | None
    context_score: float | None
    quality_score: float
    trainable: bool
    curated_rejection_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "media_id": self.media_id,
            "media_path": self.media_path,
            "media_mime_type": self.media_mime_type,
            "transcript_text": self.transcript_text,
            "transcript_preview": self.transcript_preview,
            "transcript_segments": [
                segment.model_dump(mode="json")
                for segment in self.transcript_segments
            ],
            "page_title": self.page_title,
            "surrounding_text": self.surrounding_text,
            "html_context": self.html_context,
            "asset_fetch_mode": self.asset_fetch_mode,
            "is_complete_payload": self.is_complete_payload,
            "language": self.language,
            "context_score": self.context_score,
            "quality_score": self.quality_score,
            "trainable": self.trainable,
            "curated_rejection_reason": self.curated_rejection_reason,
            "asset_context": self.asset_context.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TrainingAudioInput(TrainingTimedMediaInput):
    normalized_audio_path: str
    target_audio_path: str
    audio_duration_seconds: float | None
    audio_chromaprint: str | None


@dataclass(frozen=True, slots=True)
class TrainingVideoInput(TrainingTimedMediaInput):
    normalized_video_path: str
    target_video_path: str
    video_duration_seconds: float | None
    video_width: int | None
    video_height: int | None
    frame_ocr_text: str | None
    frame_ocr_preview: str | None
    keyframes: tuple[dict[str, object], ...]
    video_keyframe_phashes: tuple[str, ...] | None

    def to_dict(self) -> dict[str, object]:
        payload = TrainingTimedMediaInput.to_dict(self)
        payload.update(
            {
                "frame_ocr_text": self.frame_ocr_text,
                "frame_ocr_preview": self.frame_ocr_preview,
                "keyframes": [dict(frame) for frame in self.keyframes],
            }
        )
        return payload


def project_audio_record(record: CuratedAudioRecord) -> TrainingAudioInput:
    """Project persisted audio state into the training-owned input model."""

    return TrainingAudioInput(
        media_id=record.media_id,
        media_path=record.media_path,
        transcript_text=record.transcript_text,
        source_url=record.source_url,
        allow_training=record.allow_training,
        parent_document_id=record.parent_document_id,
        modality=record.modality,
        domain=record.domain,
        page_title=record.page_title,
        transcript_preview=record.transcript_preview,
        surrounding_text=record.surrounding_text,
        html_context=record.html_context,
        media_mime_type=record.media_mime_type,
        transcript_segments=record.transcript_segments,
        license=record.license,
        license_url=record.license_url,
        governance_note=record.governance_note,
        robots_status=record.robots_status,
        terms_source=record.terms_source,
        usage_rules=record.usage_rules,
        asset_fetch_mode=record.asset_fetch_mode,
        is_complete_payload=record.is_complete_payload,
        near_duplicate_cluster_id=record.near_duplicate_cluster_id,
        media_fingerprint=record.media_fingerprint,
        privacy_clearance=PrivacyClearance.from_dict(
            record.privacy_clearance.to_dict()
        ),
        asset_context=record.asset_context,
        language=record.language,
        context_score=record.context_score,
        quality_score=record.quality_score,
        trainable=record.trainable,
        curated_rejection_reason=record.curated_rejection_reason,
        normalized_audio_path=record.normalized_audio_path,
        target_audio_path=record.target_audio_path,
        audio_duration_seconds=record.audio_duration_seconds,
        audio_chromaprint=record.audio_chromaprint,
    )


def project_video_record(record: CuratedVideoRecord) -> TrainingVideoInput:
    """Project persisted video state into the training-owned input model."""

    return TrainingVideoInput(
        media_id=record.media_id,
        media_path=record.media_path,
        transcript_text=record.transcript_text,
        source_url=record.source_url,
        allow_training=record.allow_training,
        parent_document_id=record.parent_document_id,
        modality=record.modality,
        domain=record.domain,
        page_title=record.page_title,
        transcript_preview=record.transcript_preview,
        surrounding_text=record.surrounding_text,
        html_context=record.html_context,
        media_mime_type=record.media_mime_type,
        transcript_segments=record.transcript_segments,
        license=record.license,
        license_url=record.license_url,
        governance_note=record.governance_note,
        robots_status=record.robots_status,
        terms_source=record.terms_source,
        usage_rules=record.usage_rules,
        asset_fetch_mode=record.asset_fetch_mode,
        is_complete_payload=record.is_complete_payload,
        near_duplicate_cluster_id=record.near_duplicate_cluster_id,
        media_fingerprint=record.media_fingerprint,
        privacy_clearance=PrivacyClearance.from_dict(
            record.privacy_clearance.to_dict()
        ),
        asset_context=record.asset_context,
        language=record.language,
        context_score=record.context_score,
        quality_score=record.quality_score,
        trainable=record.trainable,
        curated_rejection_reason=record.curated_rejection_reason,
        normalized_video_path=record.normalized_video_path,
        target_video_path=record.target_video_path,
        video_duration_seconds=record.video_duration_seconds,
        video_width=record.video_width,
        video_height=record.video_height,
        frame_ocr_text=record.frame_ocr_text,
        frame_ocr_preview=record.frame_ocr_preview,
        keyframes=record.keyframes,
        video_keyframe_phashes=record.video_keyframe_phashes,
    )


TimedMediaTrainingInput = TrainingAudioInput | TrainingVideoInput

__all__ = [
    "TimedMediaTrainingInput",
    "TrainingAudioInput",
    "TrainingTimedMediaInput",
    "TrainingVideoInput",
    "project_audio_record",
    "project_video_record",
]
