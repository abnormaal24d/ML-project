"""Assemble curated documents, chunks, and multimodal training pairs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from crawler.curation.media.cleared_image_records import (
    build_privacy_cleared_images,
)
from crawler.curation.media.cleared_timed_media_records import (
    build_privacy_cleared_timed_media,
)
from crawler.curation.media.context.timed_media_coverage import (
    audio_coverage,
    build_timed_media_coverage,
    video_coverage,
)
from crawler.storage.datasets.extraction.page_extraction_artifact import (
    PageExtractionArtifactError,
)
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.curated.document import (
    ChunkRecord,
    CuratedDocumentRecord,
)
from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.curated.timed_media import (
    CuratedAudioRecord,
    CuratedVideoRecord,
)

if TYPE_CHECKING:
    from pathlib import Path

    from crawler.curation.ingest.schema.entry import RawManifestEntry
    from crawler.curation.snapshots.dataset_assembly.curated_assembly_types import (
        PreprocessingInputBuilder,
    )
    from preprocessing.multimodal_preprocessor import MultimodalPreprocessor


def _build_curated_chunks(
    *,
    snapshot_id: str,
    schema_version: str,
    preprocessed_chunks: tuple[Any, ...],
) -> tuple[ChunkRecord, ...]:
    return tuple(
        _build_chunk_record(
            snapshot_id=snapshot_id,
            schema_version=schema_version,
            chunk=chunk,
        )
        for chunk in preprocessed_chunks
    )


def _build_chunk_record(
    *,
    snapshot_id: str,
    schema_version: str,
    chunk: Any,
) -> ChunkRecord:
    return ChunkRecord(
        schema_version=schema_version,
        snapshot_id=snapshot_id,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        token_count_estimate=chunk.token_count_estimate,
        text=chunk.text,
        language=chunk.language,
        title=chunk.title,
        section_path=chunk.section_path,
        quality_score=chunk.quality_score,
        exact_duplicate_key=chunk.exact_duplicate_key or "",
        near_duplicate_cluster_id=chunk.near_duplicate_cluster_id,
        split=chunk.split,
    )


@dataclass(frozen=True, slots=True)
class CuratedSampleBundle:
    """Intermediate curated records before quality filtering."""

    documents: tuple[CuratedDocumentRecord, ...]
    preprocessed_documents_by_id: dict[str, Any]
    chunks: tuple[ChunkRecord, ...]
    images: tuple[CuratedImageRecord, ...]
    audio_rows: tuple[CuratedAudioRecord, ...]
    video_rows: tuple[CuratedVideoRecord, ...]
    sync_rows: tuple[dict[str, Any], ...]
    audio_coverage: dict[str, object]
    video_coverage: dict[str, object]


async def build_curated_samples(
    *,
    snapshot_id: str,
    snapshot_directory: Path,
    project_root: Path,
    schema_version: str,
    raw_entries: tuple[RawManifestEntry, ...],
    document_curator_factory: Callable[..., Any],
    preprocessing_input_builder: PreprocessingInputBuilder,
    preprocessing_phase_runner: MultimodalPreprocessor,
    logger: ProjectLogger,
    chunker: Any,
    sync_row_assembler: Callable[..., tuple[dict[str, Any], ...]],
    documents: tuple[Any, ...] | None = None,
    preprocessed_documents_by_id: dict[str, Any] | None = None,
) -> CuratedSampleBundle:
    """Build curated sample rows for one snapshot.

    Structural multimodal path:
      canonical preprocessing-input builder
        → await MultimodalPreprocessor.process
        → documents → CuratedDocumentAssembler.assemble
        → privacy-cleared media outputs become authoritative curated rows
        → raw entries contribute only lineage and governance identifiers
    """

    document_assembler = document_curator_factory(
        snapshot_directory=snapshot_directory,
    )
    preprocessed_images: tuple[Any, ...] = ()
    preprocessed_audio: tuple[Any, ...] = ()
    preprocessed_video: tuple[Any, ...] = ()

    if documents is None or preprocessed_documents_by_id is None:
        entries_by_source_id = {
            entry.record.fetch_record_id: entry for entry in raw_entries
        }
        try:
            preprocessing_inputs = preprocessing_input_builder(
                raw_entries=raw_entries,
            )
        except PageExtractionArtifactError as exc:
            logger.error(
                "page_extraction_load_failed",
                reason=type(exc).__name__,
                detail=str(exc),
            )
            raise

        preprocessing_result = await preprocessing_phase_runner.process(
            inputs=preprocessing_inputs
        )
        documents, preprocessed_documents_by_id = document_assembler.assemble(
            snapshot_id=snapshot_id,
            documents=preprocessing_result.documents,
            entries_by_source_id=entries_by_source_id,
        )
        preprocessed_images = preprocessing_result.images
        preprocessed_audio = preprocessing_result.audio
        preprocessed_video = preprocessing_result.video
        logger.info(
            "multimodal_preprocessing_completed_for_curation",
            snapshot_id=snapshot_id,
            input_count_by_modality=preprocessing_result.diagnostics.get(
                "input_count_by_modality"
            ),
            output_count_by_type=preprocessing_result.diagnostics.get(
                "output_count_by_type"
            ),
            preprocessed_documents=len(preprocessing_result.documents),
            preprocessed_images=len(preprocessing_result.images),
            preprocessed_audio=len(preprocessing_result.audio),
            preprocessed_video=len(preprocessing_result.video),
            skipped_sources=len(preprocessing_result.skipped_sources),
            quarantine_records=len(preprocessing_result.quarantine_records),
            # Pair assemblers still own curated training-pair rows from raw
            # crawl context (parent pages, captions, trainability gates).
            curated_media_pair_source="privacy_cleared_preprocessing",
            multimodal_media_items_available={
                "image": len(preprocessing_result.images),
                "audio": len(preprocessing_result.audio),
                "video": len(preprocessing_result.video),
            },
        )

    if documents is None or preprocessed_documents_by_id is None:
        raise RuntimeError("preprocessed document inputs were not resolved")

    preprocessed_chunks = chunker.build(
        documents=tuple(
            preprocessed_documents_by_id[document.document_id]
            for document in documents
            if document.document_id in preprocessed_documents_by_id
        )
    )
    chunks = _build_curated_chunks(
        snapshot_id=snapshot_id,
        schema_version=schema_version,
        preprocessed_chunks=preprocessed_chunks,
    )

    # Media rows are built only from privacy-cleared preprocessing outputs.
    images = build_privacy_cleared_images(
        snapshot_id=snapshot_id,
        schema_version=schema_version,
        raw_entries=raw_entries,
        documents=documents,
        preprocessed=preprocessed_images,
        project_root=project_root,
    )
    audio_rows = build_privacy_cleared_timed_media(
        snapshot_id=snapshot_id,
        schema_version=schema_version,
        modality="audio",
        raw_entries=raw_entries,
        documents=documents,
        preprocessed=preprocessed_audio,
        project_root=project_root,
    )
    video_rows = build_privacy_cleared_timed_media(
        snapshot_id=snapshot_id,
        schema_version=schema_version,
        modality="video",
        raw_entries=raw_entries,
        documents=documents,
        preprocessed=preprocessed_video,
        project_root=project_root,
    )
    sync_rows = sync_row_assembler(
        snapshot_id=snapshot_id,
        images=images,
        audio_rows=audio_rows,
        video_rows=video_rows,
        schema_version=schema_version,
        documents=documents,
        chunks=chunks,
        preprocessed_documents_by_id=preprocessed_documents_by_id,
    )

    return CuratedSampleBundle(
        documents=documents,
        preprocessed_documents_by_id=preprocessed_documents_by_id,
        chunks=chunks,
        images=images,
        audio_rows=audio_rows,
        video_rows=video_rows,
        sync_rows=sync_rows,
        audio_coverage={
            **build_timed_media_coverage(
                rows=audio_rows,
                raw_found=sum(
                    1 for entry in raw_entries if entry.record.kind == "audio"
                ),
                enrichment_fields=(
                    "audio_duration_seconds",
                    "audio_sample_rate",
                    "audio_channels",
                ),
            ),
            **audio_coverage(audio_rows),
        },
        video_coverage={
            **build_timed_media_coverage(
                rows=video_rows,
                raw_found=sum(
                    1 for entry in raw_entries if entry.record.kind == "video"
                ),
                enrichment_fields=(
                    "video_duration_seconds",
                    "video_width",
                    "video_height",
                ),
            ),
            **video_coverage(video_rows),
        },
    )
