"""Stable split-group keys for related curated records."""

from __future__ import annotations

from pathlib import Path
from typing import TypeGuard

from mmcrawler_datasets.assembly.text_pairing import (
    DocumentTextRejection,
    TimedMedia,
    read_document_text,
    select_image_text,
    select_media_text,
)
from mmcrawler_datasets.curated.document import CuratedDocumentRecord
from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.snapshots.curated import CuratedSnapshot


def document_group(document: CuratedDocumentRecord) -> str:
    """Return the stable split key for one source-document family."""

    return (
        document.near_duplicate_cluster_id
        or document.exact_duplicate_key
        or document.document_id
    )


def image_group(
    *,
    image: CuratedImageRecord,
    document: CuratedDocumentRecord | None,
) -> str:
    """Return an image split key, preferring parent-document lineage."""

    if document is not None:
        return document_group(document)
    return (
        image.parent_document_id
        or image.image_phash
        or image.image_average_hash
        or image.image_id
    )


def media_group(
    *,
    record: TimedMedia,
    document: CuratedDocumentRecord | None,
) -> str:
    """Return an audio/video split key with a deterministic fallback."""

    if document is not None:
        return document_group(document)
    return (
        record.parent_document_id
        or record.media_fingerprint
        or record.media_id
    )


def collect_group_keys(
    *,
    snapshot: CuratedSnapshot,
    require_allow_training: bool,
    snapshot_directory: Path,
    document_text_cache: dict[str, str | None],
) -> tuple[str, ...]:
    """Collect every viable family before deterministic split assignment."""

    documents = snapshot.documents_by_id
    keys: set[str] = set()
    for chunk in snapshot.chunks:
        document = documents.get(chunk.document_id)
        if not _document_allowed(document, require_allow_training):
            continue
        if chunk.text.strip():
            keys.add(document_group(document))
    for document in snapshot.documents:
        if not _document_allowed(document, require_allow_training):
            continue
        clearance = document.privacy_clearance
        if clearance is None:
            continue
        try:
            text = read_document_text(
                snapshot_directory=snapshot_directory,
                text_path=document.text_path,
                privacy_clearance=clearance,
                cache=document_text_cache,
            )
        except DocumentTextRejection:
            continue
        if text:
            keys.add(document_group(document))
    for image in snapshot.images:
        if require_allow_training and image.allow_training is not True:
            continue
        if select_image_text(image) is None:
            continue
        document = (
            documents.get(image.parent_document_id)
            if image.parent_document_id
            else None
        )
        keys.add(image_group(image=image, document=document))
    timed_media: list[TimedMedia] = [
        *snapshot.audio,
        *snapshot.video,
    ]
    for record in timed_media:
        if require_allow_training and record.allow_training is not True:
            continue
        if select_media_text(record) is None:
            continue
        document = (
            documents.get(record.parent_document_id)
            if record.parent_document_id
            else None
        )
        keys.add(media_group(record=record, document=document))
    return tuple(keys)


def _document_allowed(
    document: CuratedDocumentRecord | None,
    require_allow_training: bool,
) -> TypeGuard[CuratedDocumentRecord]:
    if (
        document is None
        or document.quality_bucket == "reject"
        or document.privacy_clearance is None
        or not document.privacy_clearance.permits_training
    ):
        return False
    return not require_allow_training or document.allow_training is True


__all__ = [
    "collect_group_keys",
    "document_group",
    "image_group",
    "media_group",
]
