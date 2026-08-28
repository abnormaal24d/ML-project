"""Canonical strict wire contracts for curated audio and video records."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Literal, TypeAlias

from pydantic import field_validator, model_validator

from mmcrawler_datasets.curated.evidence import (
    AssetContextRecord,
    PrivacyClearanceRecord,
)
from mmcrawler_datasets.curated.strict import (
    FiniteFloat,
    NonEmptyText,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Score,
    Sha256Text,
    StrictContractModel,
    relative_path,
)


class TranscriptSegment(StrictContractModel):
    """One normalized transcript segment in canonical second units."""

    text: NonEmptyText
    source: NonEmptyText
    start_seconds: NonNegativeFloat | None
    end_seconds: NonNegativeFloat | None
    confidence: Score | None

    @model_validator(mode="after")
    def _validate_range(self) -> TranscriptSegment:
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("segment end precedes segment start")
        return self

    @classmethod
    def from_preprocessed(
        cls, value: Mapping[str, object]
    ) -> TranscriptSegment:
        """Normalize one trusted preprocessing segment to the wire schema."""

        known = {
            "text",
            "source",
            "start_seconds",
            "end_seconds",
            "confidence",
            "start_ms",
            "end_ms",
        }
        unknown = set(value).difference(known)
        if unknown:
            raise ValueError(
                f"transcript segment contains unknown fields: {sorted(unknown)}"
            )
        start_seconds = _segment_seconds(
            seconds=value.get("start_seconds"),
            milliseconds=value.get("start_ms"),
            field_name="start",
        )
        end_seconds = _segment_seconds(
            seconds=value.get("end_seconds"),
            milliseconds=value.get("end_ms"),
            field_name="end",
        )
        return cls.model_validate(
            {
                "text": value.get("text"),
                "source": value.get("source") or "transcript_segment",
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "confidence": value.get("confidence"),
            }
        )


class CuratedTimedMediaRecord(StrictContractModel):
    """Shared canonical persisted fields for curated audio and video."""

    schema_version: Literal["3.0"]
    snapshot_id: NonEmptyText
    media_id: NonEmptyText
    object_id: NonEmptyText
    source_run_id: NonEmptyText
    source_url: NonEmptyText
    media_path: NonEmptyText
    media_mime_type: NonEmptyText | None
    domain: NonEmptyText
    language: NonEmptyText | None
    parent_document_id: NonEmptyText | None
    page_title: NonEmptyText | None
    surrounding_text: NonEmptyText | None
    html_context: NonEmptyText | None
    transcript_text: NonEmptyText | None
    transcript_preview: NonEmptyText | None
    transcript_language: NonEmptyText | None
    transcript_segments: tuple[TranscriptSegment, ...]
    context_score: Score | None
    quality_score: Score
    fetch_mode: NonEmptyText | None
    asset_fetch_mode: NonEmptyText | None
    is_complete_payload: bool
    observed_bytes: NonNegativeInt
    source_content_length: NonNegativeInt | None
    source_content_type: NonEmptyText | None
    fetch_duration_seconds: NonNegativeFloat | None
    payload_sha256: Sha256Text
    media_fingerprint: NonEmptyText | None
    near_duplicate_cluster_id: NonEmptyText | None
    allow_training: bool | None
    license: NonEmptyText | None
    license_url: NonEmptyText | None
    governance_note: NonEmptyText | None
    robots_status: NonEmptyText | None
    terms_source: NonEmptyText | None
    usage_rules: NonEmptyText | None
    privacy_clearance: PrivacyClearanceRecord
    safety_status: NonEmptyText
    asset_context: AssetContextRecord
    trainable: bool
    curated_media_status: Literal["trainable", "metadata_only"]
    curated_rejection_reason: NonEmptyText | None

    @field_validator(
        "media_path",
        "normalized_audio_path",
        "target_audio_path",
        "normalized_video_path",
        "target_video_path",
        check_fields=False,
    )
    @classmethod
    def _validate_media_path(cls, value: str) -> str:
        return relative_path(value)

    @field_validator("transcript_segments", mode="before")
    @classmethod
    def _validate_segments(cls, value: object) -> object:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        raise TypeError("transcript_segments must be an array")

    @model_validator(mode="after")
    def _validate_governance(self) -> CuratedTimedMediaRecord:
        expected_status = "trainable" if self.trainable else "metadata_only"
        if self.curated_media_status != expected_status:
            raise ValueError("trainable and curated_media_status disagree")
        if self.language != self.transcript_language:
            raise ValueError("language and transcript_language disagree")
        if self.asset_context.safety_status != self.safety_status:
            raise ValueError("asset_context and safety_status disagree")
        if self.trainable:
            if self.allow_training is not True:
                raise ValueError(
                    "trainable media requires allow_training=true"
                )
            if self.license is None:
                raise ValueError("trainable media requires a license")
            if not self.privacy_clearance.permits_training:
                raise ValueError("trainable media requires privacy clearance")
            if self.curated_rejection_reason is not None:
                raise ValueError(
                    "trainable media cannot have a rejection reason"
                )
        elif self.curated_rejection_reason is None:
            raise ValueError("metadata-only media requires a rejection reason")
        return self


class CuratedAudioRecord(CuratedTimedMediaRecord):
    """Canonical persisted curated audio record."""

    modality: Literal["audio"]
    normalized_audio_path: NonEmptyText
    target_audio_path: NonEmptyText
    audio_duration_seconds: NonNegativeFloat | None
    audio_sample_rate: PositiveInt | None
    audio_channels: PositiveInt | None
    audio_loudness_lufs: FiniteFloat | None
    audio_chromaprint: NonEmptyText | None


class CuratedVideoRecord(CuratedTimedMediaRecord):
    """Canonical persisted curated video record."""

    modality: Literal["video"]
    normalized_video_path: NonEmptyText
    target_video_path: NonEmptyText
    video_duration_seconds: NonNegativeFloat | None
    video_width: PositiveInt | None
    video_height: PositiveInt | None
    frame_ocr_text: NonEmptyText | None
    frame_ocr_preview: NonEmptyText | None
    keyframes: tuple[dict[str, object], ...]
    video_keyframe_phashes: tuple[NonEmptyText, ...] | None

    @field_validator("keyframes", mode="before")
    @classmethod
    def _validate_keyframes(cls, value: object) -> object:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        raise TypeError("keyframes must be an array")

    @field_validator("video_keyframe_phashes", mode="before")
    @classmethod
    def _validate_keyframe_hashes(cls, value: object) -> object:
        if value is None or isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        raise TypeError("video_keyframe_phashes must be an array or null")

    @model_validator(mode="after")
    def _require_keyframe_clearance(self) -> CuratedVideoRecord:
        if not self.keyframes:
            return self
        approved_keyframes = {
            item.object_id
            for item in self.privacy_clearance.approved_objects
            if item.role == "keyframe"
        }
        for index, keyframe in enumerate(self.keyframes):
            object_id = str(
                keyframe.get("object_id") or keyframe.get("frame_id") or ""
            ).strip()
            if not object_id or object_id not in approved_keyframes:
                raise ValueError(
                    "persisted keyframes require object-level privacy clearance"
                    f" (keyframe index {index})"
                )
        return self


TimedMediaRecord: TypeAlias = CuratedAudioRecord | CuratedVideoRecord


def _segment_seconds(
    *,
    seconds: object,
    milliseconds: object,
    field_name: str,
) -> float | None:
    if seconds is not None and milliseconds is not None:
        raise ValueError(f"segment {field_name} has conflicting units")
    if seconds is not None:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise TypeError(f"segment {field_name}_seconds must be numeric")
        result = float(seconds)
    elif milliseconds is not None:
        if isinstance(milliseconds, bool) or not isinstance(
            milliseconds,
            (int, float),
        ):
            raise TypeError(f"segment {field_name}_ms must be numeric")
        result = float(milliseconds) / 1000.0
    else:
        return None
    if not math.isfinite(result) or result < 0:
        raise ValueError(
            f"segment {field_name} must be finite and non-negative"
        )
    return result


def timed_media_contract_sha256(
    record_type: type[CuratedAudioRecord] | type[CuratedVideoRecord],
) -> str:
    """Return a deterministic digest of one generated JSON schema."""

    schema = record_type.model_json_schema()
    canonical = json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


CURATED_AUDIO_CONTRACT_SHA256 = timed_media_contract_sha256(CuratedAudioRecord)
CURATED_VIDEO_CONTRACT_SHA256 = timed_media_contract_sha256(CuratedVideoRecord)


__all__ = [
    "CURATED_AUDIO_CONTRACT_SHA256",
    "CURATED_VIDEO_CONTRACT_SHA256",
    "CuratedAudioRecord",
    "CuratedTimedMediaRecord",
    "CuratedVideoRecord",
    "TimedMediaRecord",
    "TranscriptSegment",
    "timed_media_contract_sha256",
]
