"""Concrete curated curation collaborators for the snapshot graph."""

from __future__ import annotations

from pathlib import Path

from config.settings.datasets import (
    CuratedDatasetWriterSettings,
    CuratedDocumentAssemblerSettings,
    DatasetPathSettings,
    NearDeduperSettings,
    RawManifestReaderSettings,
)
from crawler.curation.documents.assembler import CuratedDocumentAssembler
from crawler.curation.publishing.curated_artifact_writer import (
    CuratedArtifactWriter,
)
from crawler.curation.publishing.dataset_export.curated_dataset_writer import (
    CuratedDatasetWriter,
)
from crawler.curation.publishing.dataset_export.jsonl_writer import JsonlWriter
from crawler.curation.snapshots.dataset_assembly import (
    curated_assembly_types,
)
from crawler.governance.domains.domain_governance_registry import (
    DomainGovernanceRegistry,
)
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.similarity.text_deduplication import (
    NearTextDeduplicator,
)
from shared.runtime_primitives import Clock


def document_curator_factory(
    *,
    document_curator_settings: CuratedDocumentAssemblerSettings,
    near_deduper_settings: NearDeduperSettings,
    artifacts_directory: str,
    source_registry: DomainGovernanceRegistry,
    logger: ProjectLogger,
    clock: Clock,
) -> curated_assembly_types.DocumentCuratorFactory:
    def build(
        *,
        snapshot_directory: Path,
    ) -> CuratedDocumentAssembler:
        return _document_curator(
            snapshot_directory=snapshot_directory,
            document_curator_settings=document_curator_settings,
            near_deduper_settings=near_deduper_settings,
            artifacts_directory=artifacts_directory,
            source_registry=source_registry,
            logger=logger,
            clock=clock,
        )

    return build


def dataset_writer_factory(
    *,
    writer_settings: CuratedDatasetWriterSettings,
    dataset_paths: DatasetPathSettings,
) -> curated_assembly_types.DatasetWriterFactory:
    def build(
        *,
        snapshot_directory: Path,
    ) -> CuratedDatasetWriter:
        return _dataset_writer(
            snapshot_directory=snapshot_directory,
            writer_settings=writer_settings,
            dataset_paths=dataset_paths,
        )

    return build


def raw_manifest_reader_settings(
    *,
    configured: RawManifestReaderSettings,
    raw_run_selection_mode: str | None,
    selected_run_ids: tuple[str, ...] | None,
) -> RawManifestReaderSettings:
    return configured.model_copy(
        update={
            "run_selection_mode": (
                raw_run_selection_mode
                if raw_run_selection_mode is not None
                else configured.run_selection_mode
            ),
            "selected_run_ids": (
                tuple(selected_run_ids)
                if selected_run_ids is not None
                else configured.selected_run_ids
            ),
        }
    )


def _document_curator(
    *,
    snapshot_directory: Path,
    document_curator_settings: CuratedDocumentAssemblerSettings,
    near_deduper_settings: NearDeduperSettings,
    artifacts_directory: str,
    source_registry: DomainGovernanceRegistry,
    logger: ProjectLogger,
    clock: Clock,
) -> CuratedDocumentAssembler:
    artifact_writer = CuratedArtifactWriter(
        snapshot_directory=snapshot_directory,
        artifacts_directory=artifacts_directory,
    )
    return CuratedDocumentAssembler(
        settings=document_curator_settings,
        source_domain_registry=source_registry,
        artifact_writer=artifact_writer,
        logger=logger,
        clock=clock,
        near_deduper=NearTextDeduplicator(
            threshold=near_deduper_settings.threshold,
            shingle_width=near_deduper_settings.shingle_width,
            candidate_bands=near_deduper_settings.candidate_bands,
            use_buckets=near_deduper_settings.use_buckets,
        ),
    )


def _dataset_writer(
    *,
    snapshot_directory: Path,
    writer_settings: CuratedDatasetWriterSettings,
    dataset_paths: DatasetPathSettings,
) -> CuratedDatasetWriter:
    return CuratedDatasetWriter(
        settings=writer_settings,
        dataset_paths=dataset_paths,
        snapshot_directory=snapshot_directory,
        jsonl_writer=JsonlWriter(),
    )
