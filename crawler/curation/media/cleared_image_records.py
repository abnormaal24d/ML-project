"""Create curated images from privacy-cleared preprocessing outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from crawler.curation.media.cleared_media_context import (
    project_relative_media_path,
    resolve_cleared_media,
    safe_asset_context,
)
from mmcrawler_datasets.curated.evidence import PrivacyClearanceRecord
from mmcrawler_datasets.curated.image import CuratedImageRecord
from preprocessing.preprocessed_media import PreprocessedImage
from preprocessing.privacy.clearance import PrivacyClearance
from preprocessing.privacy.public_provenance import public_source_url


def build_privacy_cleared_images(
    *,
    snapshot_id: str,
    schema_version: str,
    raw_entries: Iterable[Any],
    documents: Iterable[Any],
    preprocessed: tuple[PreprocessedImage, ...],
    project_root: Path,
) -> tuple[CuratedImageRecord, ...]:
    """Build image rows without consulting raw captions, OCR, or HTML."""

    rows: list[CuratedImageRecord] = []
    for item, context in resolve_cleared_media(
        raw_entries=raw_entries,
        documents=documents,
        preprocessed=preprocessed,
    ):
        if not isinstance(item, PreprocessedImage):
            continue
        record = context.entry.record
        clearance = context.clearance
        media_path = project_relative_media_path(
            media_path=item.media_path,
            project_root=project_root,
        )
        caption_text, caption_source = _caption(
            clearance=clearance,
            item=item,
        )
        rows.append(
            CuratedImageRecord(
                schema_version=schema_version,
                snapshot_id=snapshot_id,
                image_id=item.media_id,
                object_id=record.object_id,
                source_run_id=record.run_id,
                media_path=media_path,
                image_mime_type=item.mime_type,
                source_url=clearance.approved_text("source_url")
                or public_source_url(item.source_url),
                parent_document_id=(
                    context.parent_document.document_id
                    if context.parent_document is not None
                    else None
                ),
                page_title=clearance.approved_text("page_title")
                or clearance.approved_text("title"),
                alt_text=clearance.approved_text("alt_text"),
                figcaption=clearance.approved_text("figcaption"),
                surrounding_text=clearance.approved_text("surrounding_text"),
                caption_text=caption_text,
                caption_source=caption_source,
                caption_quality_score=_caption_quality(item),
                context_score=float(item.quality.score),
                ocr_preview=clearance.approved_text("ocr_preview"),
                image_width=item.width,
                image_height=item.height,
                image_format=_media_format(item.mime_type),
                image_average_hash=item.dedupe_fingerprints.get("image_ahash"),
                split=None,
                allow_training=context.allow_training,
                license=context.license,
                license_url=context.license_url,
                governance_note=context.governance_note,
                robots_status=context.robots_status,
                terms_source=context.terms_source,
                usage_rules=context.usage_rules,
                ocr_text=clearance.approved_text("ocr_text"),
                ocr_confidence=item.ocr_confidence,
                ocr_language=item.ocr_language,
                ocr_quality_score=item.ocr_quality_score,
                image_quality_score=float(item.quality.score),
                image_aspect_ratio=(
                    float(item.width) / float(item.height)
                    if item.width and item.height
                    else None
                ),
                image_payload_bytes=(
                    int(record.observed_bytes or record.byte_size or 0) or None
                ),
                image_difference_hash=item.dedupe_fingerprints.get(
                    "image_dhash"
                ),
                image_phash=item.dedupe_fingerprints.get("image_phash"),
                normalized_media_path=media_path,
                trainable=context.trainable,
                curated_media_status=(
                    "trainable" if context.trainable else "metadata_only"
                ),
                curated_rejection_reason=(
                    None if context.trainable else "privacy_or_license_denied"
                ),
                privacy_clearance=PrivacyClearanceRecord.model_validate(
                    context.clearance.to_dict()
                ),
                asset_context=safe_asset_context(
                    context=context,
                    safety_status=item.safety_status,
                ),
            )
        )
    return tuple(rows)


def _caption(
    *,
    clearance: PrivacyClearance,
    item: PreprocessedImage,
) -> tuple[str | None, str | None]:
    preferred = str(item.alignment_signals.get("caption_source") or "").strip()
    for field_name, source in (
        ("caption_text", preferred or "caption"),
        ("figcaption", "figcaption"),
        ("alt_text", "alt"),
        ("ocr_text", "ocr"),
        ("surrounding_text", "surrounding"),
        ("page_title", "page_title"),
    ):
        value = clearance.approved_text(field_name)
        if value:
            return value, source
    return None, None


def _caption_quality(item: PreprocessedImage) -> float:
    value = item.alignment_signals.get("caption_quality_score")
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, str, bytes, bytearray),
    ):
        return float(item.quality.score)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return float(item.quality.score)


def _media_format(mime_type: str | None) -> str | None:
    if not mime_type or "/" not in mime_type:
        return None
    return mime_type.rsplit("/", 1)[-1].casefold()


__all__ = ["build_privacy_cleared_images"]
