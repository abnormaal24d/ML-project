"""Apply deduplication and curated quality coverage reporting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crawler.curation.media.context.timed_media_coverage import (
    refresh_timed_media_coverage,
)
from crawler.curation.media.image.caption_validation import (
    has_caption_garbage,
    is_boilerplate_caption,
)
from crawler.curation.snapshots.dataset_assembly.curated_assembly_types import (
    CuratedValidationReport,
    TimedMediaRowDeduper,
)
from crawler.curation.snapshots.dataset_assembly.curated_sample_pairer import (
    CuratedSampleBundle,
)
from mmcrawler_datasets.curated.document import (
    ChunkRecord,
    CuratedDocumentRecord,
)
from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.curated.timed_media import (
    CuratedAudioRecord,
    CuratedVideoRecord,
)


@dataclass(frozen=True, slots=True)
class CuratedFilteredBundle:
    """Curated records after deduplication and validation."""

    documents: tuple[CuratedDocumentRecord, ...]
    chunks: tuple[ChunkRecord, ...]
    images: tuple[CuratedImageRecord, ...]
    audio_rows: tuple[CuratedAudioRecord, ...]
    video_rows: tuple[CuratedVideoRecord, ...]
    sync_rows: tuple[dict[str, Any], ...]
    dedupe_stats: dict[str, int]
    image_coverage: dict[str, object]
    audio_coverage: dict[str, object]
    video_coverage: dict[str, object]
    validation_payload: dict[str, object]
    validation_valid: bool


def validate_curated_snapshot(
    *,
    documents: tuple[CuratedDocumentRecord, ...],
    chunks: tuple[ChunkRecord, ...],
    images: tuple[CuratedImageRecord, ...],
    audio_rows: tuple[CuratedAudioRecord, ...],
    video_rows: tuple[CuratedVideoRecord, ...],
    sync_rows: tuple[dict[str, Any], ...],
) -> CuratedValidationReport:
    """Validate curated entities, lineage, and multimodal coverage."""

    errors: list[str] = []
    document_ids = [document.document_id for document in documents]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    image_ids = [image.image_id for image in images]
    audio_ids = [row.media_id for row in audio_rows]
    video_ids = [row.media_id for row in video_rows]
    document_id_set = set(document_ids)
    media_id_set = set(audio_ids) | set(video_ids)
    image_id_set = set(image_ids)

    errors.extend(duplicate_errors(kind="document_id", values=document_ids))
    errors.extend(duplicate_errors(kind="chunk_id", values=chunk_ids))
    errors.extend(duplicate_errors(kind="image_id", values=image_ids))
    errors.extend(duplicate_errors(kind="audio_media_id", values=audio_ids))
    errors.extend(duplicate_errors(kind="video_media_id", values=video_ids))

    errors.extend(
        f"orphan_chunk:{chunk.chunk_id}:{chunk.document_id}"
        for chunk in chunks
        if chunk.document_id not in document_id_set
    )
    errors.extend(
        f"orphan_image_parent:{image.image_id}:{image.parent_document_id}"
        for image in images
        if image.parent_document_id
        and image.parent_document_id not in document_id_set
    )
    errors.extend(
        f"missing_image_path:{image.image_id}"
        for image in images
        if not image.media_path
    )

    for modality, rows in (("audio", audio_rows), ("video", video_rows)):
        errors.extend(
            f"orphan_media_parent:{modality}:"
            f"{row.media_id}:{parent_document_id}"
            for row in rows
            if (parent_document_id := row.parent_document_id)
            and parent_document_id not in document_id_set
        )
        errors.extend(
            f"missing_media_path:{modality}:{row.media_id}"
            for row in rows
            if not row.media_path
        )

    errors.extend(
        f"orphan_alignment_document:{str(row.get('object_id') or '')}:"
        f"{document_id}"
        for row in sync_rows
        if (document_id := str(row.get("text_document_id") or ""))
        and document_id not in document_id_set
    )
    errors.extend(
        f"orphan_alignment_object:image:{str(row.get('object_id') or '')}"
        for row in sync_rows
        if str(row.get("object_modality") or "") == "image"
        and str(row.get("object_id") or "") not in image_id_set
    )
    errors.extend(
        "orphan_alignment_object:"
        f"{str(row.get('object_modality') or '')}:"
        f"{str(row.get('object_id') or '')}"
        for row in sync_rows
        if str(row.get("object_modality") or "") in {"audio", "video"}
        and str(row.get("object_id") or "") not in media_id_set
    )

    payload = {
        "valid": not errors,
        "errors": errors,
        "counts": {
            "documents": len(documents),
            "chunks": len(chunks),
            "images": len(images),
            "audio_rows": len(audio_rows),
            "video_rows": len(video_rows),
            "alignments": len(sync_rows),
        },
        "coverage": {
            "audio": summarize_media_validation_coverage(audio_rows),
            "video": summarize_media_validation_coverage(video_rows),
            "images": build_validation_image_coverage(images=images),
        },
    }
    return CuratedValidationReport(
        valid=not errors,
        errors=tuple(errors),
        payload=payload,
    )


def apply_curated_quality_filters(
    *,
    bundle: CuratedSampleBundle,
    raw_entries: tuple[Any, ...],
    document_deduper: Callable[
        ...,
        tuple[tuple[CuratedDocumentRecord, ...], dict[str, Any]],
    ],
    image_deduper: Callable[..., tuple[CuratedImageRecord, ...]],
    media_row_deduper: TimedMediaRowDeduper,
    sync_row_deduper: Callable[..., tuple[dict[str, Any], ...]],
    snapshot_validator: Callable[..., CuratedValidationReport],
) -> CuratedFilteredBundle:
    documents = bundle.documents
    preprocessed_documents_by_id = bundle.preprocessed_documents_by_id
    dedupe_stats: dict[str, int] = {
        "documents_before_dedup": len(documents),
    }
    documents, preprocessed_documents_by_id = document_deduper(
        documents=documents,
        preprocessed_documents_by_id=preprocessed_documents_by_id,
    )
    dedupe_stats["documents_after_dedup"] = len(documents)

    chunks = bundle.chunks
    images = bundle.images
    dedupe_stats["images_before_dedup"] = len(images)
    images = image_deduper(images=images)
    dedupe_stats["images_after_dedup"] = len(images)
    image_coverage = build_image_coverage(
        images=images,
        raw_entries=raw_entries,
        dropped_as_duplicate=max(
            0,
            dedupe_stats["images_before_dedup"]
            - dedupe_stats["images_after_dedup"],
        ),
    )

    audio_rows = bundle.audio_rows
    dedupe_stats["audio_before_dedup"] = len(audio_rows)
    audio_rows = media_row_deduper(rows=audio_rows)
    dedupe_stats["audio_after_dedup"] = len(audio_rows)
    audio_duplicates = max(
        0,
        dedupe_stats["audio_before_dedup"] - dedupe_stats["audio_after_dedup"],
    )
    audio_coverage = refresh_timed_media_coverage(
        previous=bundle.audio_coverage,
        rows=audio_rows,
        modality="audio",
        dropped_as_duplicate=audio_duplicates,
    )

    video_rows = bundle.video_rows
    dedupe_stats["video_before_dedup"] = len(video_rows)
    video_rows = media_row_deduper(rows=video_rows)
    dedupe_stats["video_after_dedup"] = len(video_rows)
    video_duplicates = max(
        0,
        dedupe_stats["video_before_dedup"] - dedupe_stats["video_after_dedup"],
    )
    video_coverage = refresh_timed_media_coverage(
        previous=bundle.video_coverage,
        rows=video_rows,
        modality="video",
        dropped_as_duplicate=video_duplicates,
    )

    sync_rows = bundle.sync_rows
    dedupe_stats["alignments_before_dedup"] = len(sync_rows)
    sync_rows = sync_row_deduper(rows=sync_rows)
    dedupe_stats["alignments_after_dedup"] = len(sync_rows)

    validation = snapshot_validator(
        documents=documents,
        chunks=chunks,
        images=images,
        audio_rows=audio_rows,
        video_rows=video_rows,
        sync_rows=sync_rows,
    )
    validation_payload = get_validation_payload(validation)
    validation_valid = is_validation_valid(validation)

    return CuratedFilteredBundle(
        documents=documents,
        chunks=chunks,
        images=images,
        audio_rows=audio_rows,
        video_rows=video_rows,
        sync_rows=sync_rows,
        dedupe_stats=dedupe_stats,
        image_coverage=image_coverage,
        audio_coverage=audio_coverage,
        video_coverage=video_coverage,
        validation_payload=validation_payload,
        validation_valid=validation_valid,
    )


def is_validation_valid(validation: CuratedValidationReport) -> bool:
    return bool(validation.valid)


def get_validation_payload(
    validation: CuratedValidationReport,
) -> dict[str, object]:
    return validation.payload


def processor_flag(
    processors_payload: object,
    processor_name: str,
    flag_name: str,
    default: bool,
) -> bool:
    if not isinstance(processors_payload, dict):
        return default

    processor_payload = processors_payload.get(processor_name)
    if not isinstance(processor_payload, dict):
        return default
    if flag_name not in processor_payload:
        return default

    value = processor_payload[flag_name]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return False


def _caption_validation_coverage(
    images: tuple[CuratedImageRecord, ...],
) -> dict[str, int]:
    return {
        "with_html_garbage_caption": sum(
            has_caption_garbage(image.caption_text) for image in images
        ),
        "with_boilerplate_caption": sum(
            is_boilerplate_caption(image.caption_text) for image in images
        ),
    }


def build_image_coverage(
    *,
    images: tuple[CuratedImageRecord, ...],
    raw_entries: tuple[Any, ...],
    dropped_as_duplicate: int,
) -> dict[str, object]:
    """Summarize image coverage in the curated snapshot manifest."""

    raw_found = sum(1 for entry in raw_entries if entry.record.kind == "image")
    accepted_with_ocr = sum(
        1 for image in images if bool(image.ocr_text or image.ocr_preview)
    )
    accepted_with_caption = sum(
        1 for image in images if bool(image.caption_text)
    )
    trainable_caption_sources = {"figcaption", "alt", "ocr"}
    with_trainable_caption = sum(
        1
        for image in images
        if (
            bool(image.caption_text)
            and image.caption_quality_score >= 0.35
            and image.caption_source in trainable_caption_sources
        )
    )
    caption_quality_scores = [
        float(image.caption_quality_score)
        for image in images
        if image.caption_text
    ]
    rejected_by_reason = {
        "dropped_as_duplicate": dropped_as_duplicate,
        "missing_ocr_or_caption": max(
            0,
            raw_found - len(images) - dropped_as_duplicate,
        ),
    }
    return {
        "raw_found": raw_found,
        "curated_accepted": len(images),
        "accepted_with_ocr": accepted_with_ocr,
        "accepted_with_caption": accepted_with_caption,
        "with_caption_text": accepted_with_caption,
        "with_high_quality_caption": sum(
            1
            for image in images
            if image.caption_text and image.caption_quality_score >= 0.35
        ),
        "with_trainable_caption": with_trainable_caption,
        "with_alt_caption": caption_source_count(images=images, source="alt"),
        "with_figcaption": caption_source_count(
            images=images,
            source="figcaption",
        ),
        "with_ocr_caption": caption_source_count(images=images, source="ocr"),
        "with_surrounding_caption": caption_source_count(
            images=images,
            source="surrounding",
        ),
        "with_page_title_caption": caption_source_count(
            images=images,
            source="page_title",
        ),
        **_caption_validation_coverage(images),
        "average_caption_quality_score": average(caption_quality_scores),
        "rejected_by_reason": rejected_by_reason,
    }


def duplicate_errors(*, kind: str, values: list[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return [
        f"duplicate_{kind}:{value}:{count}"
        for value, count in counts.items()
        if count > 1
    ]


def summarize_media_validation_coverage(
    rows: tuple[CuratedAudioRecord | CuratedVideoRecord, ...],
) -> dict[str, int]:
    return {
        "total": len(rows),
        "with_parent_document": sum(
            1 for row in rows if row.parent_document_id
        ),
        "with_page_title": sum(1 for row in rows if row.page_title),
        "with_surrounding_text": sum(
            1 for row in rows if row.surrounding_text
        ),
        "with_html_context": sum(1 for row in rows if row.html_context),
        "with_transcript_preview": sum(
            1 for row in rows if row.transcript_preview
        ),
        "with_transcript_text": sum(1 for row in rows if row.transcript_text),
        "with_transcript_segments": sum(
            1 for row in rows if row.transcript_segments
        ),
    }


def build_validation_image_coverage(
    *,
    images: tuple[CuratedImageRecord, ...],
) -> dict[str, object]:
    caption_quality_scores = [
        float(image.caption_quality_score)
        for image in images
        if image.caption_text
    ]

    return {
        "with_parent_document": sum(
            1 for image in images if image.parent_document_id
        ),
        "with_caption_text": sum(1 for image in images if image.caption_text),
        "with_high_quality_caption": sum(
            1
            for image in images
            if image.caption_text and image.caption_quality_score >= 0.35
        ),
        "with_trainable_caption": sum(
            1
            for image in images
            if image.caption_text
            and image.caption_quality_score >= 0.35
            and image.caption_source in {"figcaption", "alt", "ocr"}
        ),
        "with_alt_caption": caption_source_count(images=images, source="alt"),
        "with_figcaption": caption_source_count(
            images=images,
            source="figcaption",
        ),
        "with_ocr_caption": caption_source_count(images=images, source="ocr"),
        "with_surrounding_caption": caption_source_count(
            images=images,
            source="surrounding",
        ),
        "with_page_title_caption": caption_source_count(
            images=images,
            source="page_title",
        ),
        **_caption_validation_coverage(images),
        "average_caption_quality_score": average(caption_quality_scores),
        "with_ocr_preview": sum(1 for image in images if image.ocr_preview),
        "with_dimensions": sum(
            1
            for image in images
            if image.image_width is not None and image.image_height is not None
        ),
        "with_average_hash": sum(
            1 for image in images if image.image_average_hash
        ),
    }


def caption_source_count(
    *,
    images: tuple[CuratedImageRecord, ...],
    source: str,
) -> int:
    return sum(1 for image in images if image.caption_source == source)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
