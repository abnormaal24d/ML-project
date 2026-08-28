"""Construction helpers for document and media training samples."""

from __future__ import annotations

from pathlib import Path

from mmcrawler_datasets.training_samples.artifact_path import (
    _resolve_artifact_path,
)
from mmcrawler_datasets.training_samples.common import (
    _validated_object_sha256,
    as_opt_float,
    as_opt_str,
    as_opt_str_normalized,
    guess_mime_type,
    normalize_training_text,
)
from mmcrawler_datasets.training_samples.models import (
    TrainingObject,
    TrainingTextSpan,
    _build_object,
)
from preprocessing.privacy.clearance import ApprovedObjectRole


def _primary_lineage_digest(
    *,
    row: dict[str, object],
    object_id: str,
) -> str | None:
    clearance = row.get("privacy_clearance")
    if not isinstance(clearance, dict):
        return None
    approved_objects = clearance.get("approved_objects")
    if not isinstance(approved_objects, list):
        return None
    for approved in approved_objects:
        if not isinstance(approved, dict):
            continue
        if (
            approved.get("object_id") == object_id
            and approved.get("role") == ApprovedObjectRole.PRIMARY_MEDIA.value
        ):
            value = approved.get("derived_from_digest")
            return value if isinstance(value, str) and value else None
    return None


def _build_document_text_spans(
    *,
    text: str,
) -> tuple[tuple[TrainingTextSpan, ...], int | None]:
    raw_pages = [
        segment.strip() for segment in text.split("\f") if segment.strip()
    ]

    if len(raw_pages) > 1:
        spans: list[TrainingTextSpan] = []
        search_start = 0
        for index, page in enumerate(raw_pages, start=1):
            start = text.find(page, search_start)
            if start < 0:
                start = search_start
            excerpt = page[:1200]
            end = start + len(excerpt)
            search_start = start + len(page)
            spans.append(
                TrainingTextSpan(
                    text=excerpt,
                    source="document_page_text",
                    page_number=index,
                    text_start=start,
                    text_end=end,
                    relation_type="page_contains_text",
                    evidence_source="native_document_text",
                )
            )
        return tuple(spans), len(spans)

    return (
        (
            TrainingTextSpan(
                text=text,
                source="document_text",
                page_number=1,
                text_start=0,
                text_end=len(text),
                relation_type="page_contains_text",
                evidence_source="native_document_text",
            ),
        ),
        1,
    )


def _build_media_objects(
    *,
    row: dict[str, object],
    object_path: str | None,
    media_id: str,
    project_root: Path,
    curated_snapshot_directory: Path | None = None,
) -> tuple[TrainingObject, ...]:
    objects: list[TrainingObject] = []

    primary_path = _resolve_artifact_path(
        raw_path=object_path,
        project_root=project_root,
        curated_snapshot_directory=curated_snapshot_directory,
    )
    primary = _build_object(
        object_id=media_id,
        object_path=primary_path,
        object_mime_type=as_opt_str_normalized(row.get("media_mime_type")),
        role=ApprovedObjectRole.PRIMARY_MEDIA,
        derived_from_sha256=_primary_lineage_digest(
            row=row, object_id=media_id
        ),
    )
    if primary is not None:
        objects.append(primary)
    primary_digest = primary.object_sha256 if primary is not None else None

    keyframes = row.get("keyframes")
    if isinstance(keyframes, list):
        for index, frame in enumerate(keyframes, start=1):
            if not isinstance(frame, dict):
                continue
            timestamp_seconds = as_opt_float(frame.get("timestamp_seconds"))
            frame_path = _resolve_artifact_path(
                raw_path=frame.get("frame_path"),
                project_root=project_root,
                curated_snapshot_directory=curated_snapshot_directory,
            )
            if frame_path is None:
                continue
            objects.append(
                TrainingObject(
                    object_id=f"{media_id}:frame:{index}",
                    object_path=frame_path,
                    object_sha256=_validated_object_sha256(frame_path),
                    object_mime_type=(
                        as_opt_str_normalized(frame.get("frame_mime_type"))
                        or "image/jpeg"
                    ),
                    role=ApprovedObjectRole.KEYFRAME,
                    derived_from_sha256=primary_digest,
                    frame_timestamp_seconds=timestamp_seconds,
                )
            )

    for index, proxy in enumerate(_iter_visual_proxies(row=row), start=1):
        proxy_path = _resolve_artifact_path(
            raw_path=proxy.get("media_path"),
            project_root=project_root,
            curated_snapshot_directory=curated_snapshot_directory,
        )
        if proxy_path is None:
            continue
        objects.append(
            TrainingObject(
                object_id=f"{media_id}:visual_proxy:{index}",
                object_path=proxy_path,
                object_sha256=_validated_object_sha256(proxy_path),
                object_mime_type=(
                    as_opt_str_normalized(proxy.get("mime_type"))
                    or as_opt_str_normalized(proxy.get("frame_mime_type"))
                    or guess_mime_type(proxy_path.relative_path)
                    or "image/jpeg"
                ),
                role=ApprovedObjectRole.VISUAL_PROXY,
                derived_from_sha256=primary_digest,
                confidence=as_opt_float(proxy.get("confidence")),
            )
        )

    return tuple(objects)


def _build_media_text_spans(
    *,
    row: dict[str, object],
    fallback_text: str,
    fallback_source: str | None,
) -> tuple[TrainingTextSpan, ...]:
    raw_segments = row.get("transcript_segments")
    spans: list[TrainingTextSpan] = []

    if isinstance(raw_segments, list):
        spans.extend(_normalized_training_spans(raw_segments))

    if spans:
        return tuple(spans)

    return (
        TrainingTextSpan(
            text=fallback_text,
            source=fallback_source or "paired_text",
        ),
    )


def _normalized_training_spans(
    raw_segments: list[object],
) -> tuple[TrainingTextSpan, ...]:
    normalized: list[TrainingTextSpan] = []
    seen: set[tuple[str, float | None, float | None]] = set()
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        text = normalize_training_text(segment.get("text"))
        if text is None:
            continue
        start = as_opt_float(segment.get("start_seconds"))
        end = as_opt_float(segment.get("end_seconds"))
        if start is not None and start < 0:
            continue
        if end is not None and end < 0:
            continue
        if start is not None and end is not None and end < start:
            continue
        confidence = as_opt_float(segment.get("confidence"))
        if confidence is not None and confidence < 0.01:
            continue
        key = (text.casefold(), start, end)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            TrainingTextSpan(
                text=text,
                source=as_opt_str_normalized(segment.get("source"))
                or "transcript_segment",
                start_seconds=start,
                end_seconds=end,
                confidence=confidence,
            )
        )
    return tuple(
        sorted(
            normalized,
            key=lambda item: (
                item.start_seconds is None,
                item.start_seconds or 0.0,
                item.end_seconds or 0.0,
            ),
        )
    )


def _iter_visual_proxies(
    *,
    row: dict[str, object],
) -> tuple[dict[str, object], ...]:
    raw_proxies = row.get("visual_proxies")
    if not isinstance(raw_proxies, list):
        return ()

    proxies: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in raw_proxies:
        if not isinstance(item, dict):
            continue
        proxy = dict(item)
        path = as_opt_str(proxy.get("media_path"))
        if path is None or path in seen:
            continue
        seen.add(path)
        proxies.append(proxy)
    return tuple(proxies)
