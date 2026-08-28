"""Create canonical curated audio and video records."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, overload

from crawler.curation.media.cleared_media_context import (
    ClearedMediaContext,
    project_relative_media_path,
    resolve_cleared_media,
    safe_asset_context,
)
from mmcrawler_datasets.curated.timed_media import (
    CuratedAudioRecord,
    CuratedVideoRecord,
    TranscriptSegment,
)
from preprocessing.preprocessed_media import (
    PreprocessedAudio,
    PreprocessedVideo,
)
from preprocessing.privacy.public_provenance import public_source_url
from schemas.versions import CURATED_DATASET_SCHEMA_VERSION

TimedMediaRecord = CuratedAudioRecord | CuratedVideoRecord


@overload
def build_privacy_cleared_timed_media(
    *,
    snapshot_id: str,
    schema_version: str,
    modality: Literal["audio"],
    raw_entries: Iterable[Any],
    documents: Iterable[Any],
    preprocessed: tuple[PreprocessedAudio | PreprocessedVideo, ...],
    project_root: Path,
) -> tuple[CuratedAudioRecord, ...]: ...


@overload
def build_privacy_cleared_timed_media(
    *,
    snapshot_id: str,
    schema_version: str,
    modality: Literal["video"],
    raw_entries: Iterable[Any],
    documents: Iterable[Any],
    preprocessed: tuple[PreprocessedAudio | PreprocessedVideo, ...],
    project_root: Path,
) -> tuple[CuratedVideoRecord, ...]: ...


def build_privacy_cleared_timed_media(
    *,
    snapshot_id: str,
    schema_version: str,
    modality: str,
    raw_entries: Iterable[Any],
    documents: Iterable[Any],
    preprocessed: tuple[PreprocessedAudio | PreprocessedVideo, ...],
    project_root: Path,
) -> tuple[TimedMediaRecord, ...]:
    """Build records validated by the canonical persisted contract."""

    if modality not in {"audio", "video"}:
        raise ValueError(f"unsupported timed media modality: {modality!r}")
    if schema_version != CURATED_DATASET_SCHEMA_VERSION:
        raise ValueError(
            "timed-media schema version is contract-owned: "
            f"expected={CURATED_DATASET_SCHEMA_VERSION!r}, "
            f"observed={schema_version!r}"
        )

    records: list[TimedMediaRecord] = []
    for item, context in resolve_cleared_media(
        raw_entries=raw_entries,
        documents=documents,
        preprocessed=preprocessed,
    ):
        if modality == "audio" and isinstance(item, PreprocessedAudio):
            records.append(
                _build_audio_record(
                    snapshot_id=snapshot_id,
                    item=item,
                    context=context,
                    project_root=project_root,
                )
            )
        elif modality == "video" and isinstance(item, PreprocessedVideo):
            records.append(
                _build_video_record(
                    snapshot_id=snapshot_id,
                    item=item,
                    context=context,
                    project_root=project_root,
                )
            )
    return tuple(records)


def _common_fields(
    *,
    snapshot_id: str,
    item: PreprocessedAudio | PreprocessedVideo,
    context: ClearedMediaContext,
    project_root: Path,
) -> dict[str, object]:
    record = context.entry.record
    clearance = context.clearance
    media_path = project_relative_media_path(
        media_path=item.media_path,
        project_root=project_root,
    )
    asset_context = safe_asset_context(
        context=context,
        safety_status=item.safety_status,
    )
    return {
        "schema_version": CURATED_DATASET_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "media_id": item.media_id,
        "object_id": record.object_id,
        "source_run_id": record.run_id,
        "source_url": clearance.approved_text("source_url")
        or public_source_url(item.source_url),
        "media_path": media_path,
        "media_mime_type": item.mime_type,
        "domain": item.domain,
        "language": item.transcript_language,
        "parent_document_id": (
            context.parent_document.document_id
            if context.parent_document is not None
            else None
        ),
        "page_title": clearance.approved_text("page_title")
        or clearance.approved_text("title"),
        "surrounding_text": clearance.approved_text("surrounding_text"),
        "html_context": clearance.approved_text("html_context"),
        "transcript_text": clearance.approved_text("transcript_text"),
        "transcript_preview": clearance.approved_text("transcript_preview"),
        "transcript_language": item.transcript_language,
        "transcript_segments": _transcript_segments(item.transcript_segments),
        "context_score": _explicit_context_score(item),
        "quality_score": item.quality.score,
        "fetch_mode": record.fetch_mode,
        "asset_fetch_mode": record.asset_fetch_mode or record.fetch_mode,
        "is_complete_payload": record.is_complete_payload,
        "observed_bytes": record.observed_bytes,
        "source_content_length": record.source_content_length,
        "source_content_type": record.source_content_type,
        "fetch_duration_seconds": record.fetch_duration_seconds,
        "payload_sha256": record.payload_sha256,
        "media_fingerprint": item.dedupe_fingerprints.get("media_fingerprint"),
        "near_duplicate_cluster_id": None,
        "allow_training": context.allow_training,
        "license": context.license,
        "license_url": context.license_url,
        "governance_note": context.governance_note,
        "robots_status": context.robots_status,
        "terms_source": context.terms_source,
        "usage_rules": context.usage_rules,
        "privacy_clearance": clearance.to_dict(),
        "safety_status": item.safety_status,
        "asset_context": asset_context,
        "trainable": context.trainable,
        "curated_media_status": (
            "trainable" if context.trainable else "metadata_only"
        ),
        "curated_rejection_reason": (
            None if context.trainable else "privacy_or_license_denied"
        ),
    }


def _explicit_context_score(
    item: PreprocessedAudio | PreprocessedVideo,
) -> float | None:
    """Return a real context score, never a duplicate quality score."""

    for key in ("context_score", "alignment_score"):
        if key not in item.alignment_signals:
            continue
        value = item.alignment_signals[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be numeric")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{key} must be finite and between zero and one")
        return score
    return None


def _build_audio_record(
    *,
    snapshot_id: str,
    item: PreprocessedAudio,
    context: ClearedMediaContext,
    project_root: Path,
) -> CuratedAudioRecord:
    fields = _common_fields(
        snapshot_id=snapshot_id,
        item=item,
        context=context,
        project_root=project_root,
    )
    media_path = str(fields["media_path"])
    return CuratedAudioRecord.model_validate(
        {
            **fields,
            "modality": "audio",
            "normalized_audio_path": media_path,
            "target_audio_path": media_path,
            "audio_duration_seconds": item.duration_seconds,
            "audio_sample_rate": item.sample_rate,
            "audio_channels": item.channels,
            "audio_loudness_lufs": item.loudness_lufs,
            "audio_chromaprint": item.dedupe_fingerprints.get(
                "audio_chromaprint"
            ),
        }
    )


def _build_video_record(
    *,
    snapshot_id: str,
    item: PreprocessedVideo,
    context: ClearedMediaContext,
    project_root: Path,
) -> CuratedVideoRecord:
    fields = _common_fields(
        snapshot_id=snapshot_id,
        item=item,
        context=context,
        project_root=project_root,
    )
    media_path = str(fields["media_path"])
    clearance = context.clearance
    return CuratedVideoRecord.model_validate(
        {
            **fields,
            "modality": "video",
            "normalized_video_path": media_path,
            "target_video_path": media_path,
            "video_duration_seconds": item.duration_seconds,
            "video_width": item.width,
            "video_height": item.height,
            "frame_ocr_text": clearance.approved_text("frame_ocr_text"),
            "frame_ocr_preview": clearance.approved_text("frame_ocr_preview"),
            # Keyframes require their own byte-bound privacy clearance.
            "keyframes": (),
            "video_keyframe_phashes": (
                tuple(
                    value
                    for key, value in sorted(item.dedupe_fingerprints.items())
                    if key.startswith("video_keyframe_phash:")
                )
                or None
            ),
        }
    )


def _transcript_segments(
    values: tuple[dict[str, object], ...],
) -> tuple[TranscriptSegment, ...]:
    segments: list[TranscriptSegment] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise TypeError(f"transcript segment {index} must be an object")
        try:
            segments.append(TranscriptSegment.from_preprocessed(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid transcript segment at index {index}"
            ) from exc
    return tuple(segments)


__all__ = ["build_privacy_cleared_timed_media"]
