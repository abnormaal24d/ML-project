"""Row-derived coverage for curated audio and video assembly."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Literal

from mmcrawler_datasets.curated.timed_media import (
    CuratedVideoRecord,
    TimedMediaRecord,
)

_ENRICHMENT_FIELDS_BY_MODALITY = {
    "audio": (
        "audio_duration_seconds",
        "audio_sample_rate",
        "audio_channels",
    ),
    "video": (
        "video_duration_seconds",
        "video_width",
        "video_height",
        "video_fps",
    ),
}


def build_timed_media_coverage(
    *,
    rows: tuple[TimedMediaRecord, ...],
    raw_found: int,
    enrichment_fields: tuple[str, ...],
) -> dict[str, object]:
    """Build shared coverage solely from authoritative curated rows."""

    analyzed = sum(
        1
        for row in rows
        if _has_analysis(row=row, enrichment_fields=enrichment_fields)
    )
    trainable = sum(1 for row in rows if row.trainable)
    metadata_only_by_reason = Counter(
        row.curated_rejection_reason or "unspecified"
        for row in rows
        if not row.trainable
    )

    return {
        "raw_found": raw_found,
        "analyzed": analyzed,
        "persisted_enrichment_available": analyzed,
        "curated_accepted": len(rows),
        "trainable": trainable,
        "metadata_only": len(rows) - trainable,
        "metadata_only_by_reason": dict(metadata_only_by_reason),
        "rejected_by_reason": {},
        "accepted_without_parent_document": sum(
            1 for row in rows if not row.parent_document_id
        ),
    }


def audio_coverage(
    rows: tuple[TimedMediaRecord, ...],
) -> dict[str, int]:
    """Return audio-specific observations without redoing trainability."""

    return {
        "accepted_with_transcript": sum(
            1 for row in rows if _has_transcript(row=row)
        ),
        "accepted_with_transcript_text": sum(
            1 for row in rows if row.transcript_text
        ),
        "accepted_with_transcript_preview": sum(
            1 for row in rows if row.transcript_preview
        ),
        "accepted_with_external_transcript": sum(
            1 for row in rows if _has_external_transcript(row=row)
        ),
        "complete_audio_payload": sum(
            1 for row in rows if row.is_complete_payload
        ),
    }


def video_coverage(
    rows: tuple[TimedMediaRecord, ...],
) -> dict[str, int]:
    """Return video-specific observations without redoing trainability."""

    return {
        "accepted_with_transcript": sum(
            1 for row in rows if row.transcript_text
        ),
        "accepted_with_external_transcript": sum(
            1 for row in rows if _has_external_transcript(row=row)
        ),
        "accepted_with_keyframes": sum(
            1
            for row in rows
            if isinstance(row, CuratedVideoRecord) and row.keyframes
        ),
    }


def refresh_timed_media_coverage(
    *,
    previous: Mapping[str, object],
    rows: tuple[TimedMediaRecord, ...],
    modality: Literal["audio", "video"],
    dropped_as_duplicate: int,
) -> dict[str, object]:
    """Refresh canonical coverage after curated-row deduplication."""

    enrichment_fields = _ENRICHMENT_FIELDS_BY_MODALITY.get(modality)
    if enrichment_fields is None:
        raise ValueError(f"unsupported timed-media modality: {modality!r}")

    raw_found_value = previous.get("raw_found")
    raw_found = (
        raw_found_value
        if isinstance(raw_found_value, int)
        and not isinstance(raw_found_value, bool)
        else len(rows)
    )
    refreshed = build_timed_media_coverage(
        rows=rows,
        raw_found=raw_found,
        enrichment_fields=enrichment_fields,
    )
    refreshed.update(
        audio_coverage(rows) if modality == "audio" else video_coverage(rows)
    )
    refreshed["rejected_by_reason"] = _refreshed_rejections(
        previous=previous,
        dropped_as_duplicate=dropped_as_duplicate,
    )
    return refreshed


def _has_analysis(
    *,
    row: TimedMediaRecord,
    enrichment_fields: tuple[str, ...],
) -> bool:
    if row.transcript_text:
        return True
    if row.transcript_preview:
        return True
    if row.transcript_segments:
        return True
    if isinstance(row, CuratedVideoRecord):
        if row.keyframes or row.frame_ocr_text or row.frame_ocr_preview:
            return True
        return any(
            field in enrichment_fields and value is not None
            for field, value in (
                ("video_duration_seconds", row.video_duration_seconds),
                ("video_width", row.video_width),
                ("video_height", row.video_height),
            )
        )
    return any(
        field in enrichment_fields and value is not None
        for field, value in (
            ("audio_duration_seconds", row.audio_duration_seconds),
            ("audio_sample_rate", row.audio_sample_rate),
            ("audio_channels", row.audio_channels),
        )
    )


def _has_transcript(*, row: TimedMediaRecord) -> bool:
    if row.transcript_text or row.transcript_preview:
        return True
    return bool(row.transcript_segments)


def _has_external_transcript(*, row: TimedMediaRecord) -> bool:
    return any(
        segment.source == "external_transcript_document"
        for segment in row.transcript_segments
    )


def _refreshed_rejections(
    *,
    previous: Mapping[str, object],
    dropped_as_duplicate: int,
) -> dict[str, int]:
    rejected: dict[str, int] = {}
    previous_rejected = previous.get("rejected_by_reason")
    if isinstance(previous_rejected, Mapping):
        for reason, count in previous_rejected.items():
            if str(reason) == "dropped_as_duplicate":
                continue
            if (
                isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
            ):
                rejected[str(reason)] = count

    if dropped_as_duplicate > 0:
        rejected["dropped_as_duplicate"] = int(dropped_as_duplicate)
    return rejected
