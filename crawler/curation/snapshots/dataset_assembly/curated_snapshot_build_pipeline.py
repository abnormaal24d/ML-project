"""Sample pairing, filtering, and dataset write steps for curated snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.curation.snapshots.dataset_assembly.curated_assembly_types import (
    CuratedAssemblyResult,
    CuratedDatasetAssemblerDependencies,
)
from crawler.curation.snapshots.dataset_assembly.curated_dataset_manifest_writer import (
    write_curated_dataset,
)
from crawler.curation.snapshots.dataset_assembly.curated_quality_filter import (
    CuratedFilteredBundle,
    apply_curated_quality_filters,
)
from crawler.curation.snapshots.dataset_assembly.curated_sample_pairer import (
    build_curated_samples,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from pathlib import Path

    from config.settings.datasets import DatasetPathSettings
    from crawler.curation.snapshots.dataset_assembly.curated_record_loader import (
        CuratedRawRecordSet,
    )


def validation_error_count(validation_payload: dict[str, object]) -> int:
    validation_errors = validation_payload.get("errors", ())
    if isinstance(
        validation_errors,
        (list, tuple, set, dict, str, bytes),
    ):
        return len(validation_errors)
    return 0


async def build_filtered_curated_bundle(
    *,
    dependencies: CuratedDatasetAssemblerDependencies,
    snapshot_id: str,
    snapshot_directory: Path,
    project_root: Path,
    schema_version: str,
    record_set: CuratedRawRecordSet,
) -> CuratedFilteredBundle:
    sample_bundle = await build_curated_samples(
        snapshot_id=snapshot_id,
        snapshot_directory=snapshot_directory,
        project_root=project_root,
        schema_version=schema_version,
        raw_entries=record_set.raw_entries,
        document_curator_factory=dependencies.document_curator_factory,
        preprocessing_input_builder=(dependencies.preprocessing_input_builder),
        preprocessing_phase_runner=dependencies.preprocessing_phase_runner,
        logger=dependencies.logger,
        chunker=dependencies.chunker,
        sync_row_assembler=dependencies.sync_row_assembler,
    )
    return apply_curated_quality_filters(
        bundle=sample_bundle,
        raw_entries=record_set.raw_entries,
        document_deduper=dependencies.document_deduper,
        image_deduper=dependencies.image_deduper,
        media_row_deduper=dependencies.media_row_deduper,
        sync_row_deduper=dependencies.sync_row_deduper,
        snapshot_validator=dependencies.snapshot_validator,
    )


def write_curated_snapshot(
    *,
    logger: ProjectLogger,
    dependencies: CuratedDatasetAssemblerDependencies,
    dataset_paths: DatasetPathSettings,
    schema_version: str,
    snapshot_id: str,
    snapshot_directory: Path,
    record_set: CuratedRawRecordSet,
    filtered_bundle: CuratedFilteredBundle,
) -> CuratedAssemblyResult:
    if not filtered_bundle.validation_valid:
        logger.error(
            "curated_snapshot_validation_failed",
            snapshot_id=snapshot_id,
            snapshot_directory=snapshot_directory.as_posix(),
            validation_payload=filtered_bundle.validation_payload,
        )
        raise RuntimeError(
            "curated snapshot validation failed; inspect validation payload"
        )

    dataset_writer = dependencies.dataset_writer_factory(
        snapshot_directory=snapshot_directory,
    )
    write_curated_dataset(
        dataset_writer=dataset_writer,
        documents=filtered_bundle.documents,
        chunks=filtered_bundle.chunks,
        images=filtered_bundle.images,
        audio_rows=filtered_bundle.audio_rows,
        video_rows=filtered_bundle.video_rows,
        sync_rows=filtered_bundle.sync_rows,
        snapshot_manifest_writer=dependencies.snapshot_manifest_writer,
        manifest_path=snapshot_directory
        / dataset_paths.snapshot_manifest_filename,
        snapshot_id=snapshot_id,
        schema_version=schema_version,
        source_run_ids=record_set.source_run_ids,
        dedupe_stats=filtered_bundle.dedupe_stats,
        image_coverage=filtered_bundle.image_coverage,
        audio_coverage=filtered_bundle.audio_coverage,
        video_coverage=filtered_bundle.video_coverage,
        validation_payload=filtered_bundle.validation_payload,
        content_fingerprint=record_set.content_fingerprint,
    )

    return CuratedAssemblyResult(
        snapshot_id=snapshot_id,
        snapshot_directory=snapshot_directory,
        documents=len(filtered_bundle.documents),
        chunks=len(filtered_bundle.chunks),
        images=len(filtered_bundle.images),
        audio=len(filtered_bundle.audio_rows),
        video=len(filtered_bundle.video_rows),
        alignments=len(filtered_bundle.sync_rows),
        source_run_ids=record_set.source_run_ids,
    )
