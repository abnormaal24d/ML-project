"""Persist curated snapshot artifacts and manifest metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mmcrawler_datasets.curated.timed_media import (
    CURATED_AUDIO_CONTRACT_SHA256,
    CURATED_VIDEO_CONTRACT_SHA256,
    CuratedAudioRecord,
    CuratedVideoRecord,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from crawler.curation.publishing.dataset_export.curated_dataset_writer import (
        CuratedDatasetWriter,
    )
    from mmcrawler_datasets.curated.document import (
        ChunkRecord,
        CuratedDocumentRecord,
    )
    from mmcrawler_datasets.curated.image import CuratedImageRecord


@dataclass(frozen=True, slots=True)
class CuratedDatasetWriteResult:
    """Paths to curated snapshot artifacts."""

    documents_path: Path
    chunks_path: Path
    images_path: Path
    audio_path: Path
    video_path: Path
    sync_path: Path


def write_curated_dataset(
    *,
    dataset_writer: CuratedDatasetWriter,
    documents: tuple[CuratedDocumentRecord, ...],
    chunks: tuple[ChunkRecord, ...],
    images: tuple[CuratedImageRecord, ...],
    audio_rows: tuple[CuratedAudioRecord, ...],
    video_rows: tuple[CuratedVideoRecord, ...],
    sync_rows: tuple[dict[str, Any], ...],
    snapshot_manifest_writer: Callable[..., None],
    manifest_path: Path,
    snapshot_id: str,
    schema_version: str,
    source_run_ids: tuple[str, ...],
    dedupe_stats: dict[str, int],
    image_coverage: dict[str, object],
    audio_coverage: dict[str, object],
    video_coverage: dict[str, object],
    validation_payload: dict[str, object],
    content_fingerprint: str,
) -> CuratedDatasetWriteResult:
    documents_path = dataset_writer.write_documents(documents=documents)
    chunks_path = dataset_writer.write_chunks(chunks=chunks)
    images_path = dataset_writer.write_images(images=images)
    audio_path = dataset_writer.write_audio(records=audio_rows)
    video_path = dataset_writer.write_video(records=video_rows)
    sync_path = dataset_writer.write_cross_modal_alignments(rows=sync_rows)

    snapshot_manifest_writer(
        path=manifest_path,
        snapshot_id=snapshot_id,
        schema_version=schema_version,
        documents=len(documents),
        chunks=len(chunks),
        images=len(images),
        audio=len(audio_rows),
        video=len(video_rows),
        alignments=len(sync_rows),
        documents_path=documents_path,
        chunks_path=chunks_path,
        images_path=images_path,
        audio_path=audio_path,
        video_path=video_path,
        alignments_path=sync_path,
        source_run_ids=source_run_ids,
        dedupe_stats=dedupe_stats,
        image_coverage=image_coverage,
        audio_coverage=audio_coverage,
        video_coverage=video_coverage,
        validation_payload=validation_payload,
        content_fingerprint=content_fingerprint,
        curated_audio_contract_sha256=CURATED_AUDIO_CONTRACT_SHA256,
        curated_video_contract_sha256=CURATED_VIDEO_CONTRACT_SHA256,
    )

    return CuratedDatasetWriteResult(
        documents_path=documents_path,
        chunks_path=chunks_path,
        images_path=images_path,
        audio_path=audio_path,
        video_path=video_path,
        sync_path=sync_path,
    )
