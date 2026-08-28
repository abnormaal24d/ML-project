"""Compose the concrete curated snapshot object graph."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from config.settings.root import Settings
from crawler.curation.ingest.curation_input_loader import CurationInputLoader
from crawler.curation.preprocessing_input_builder import (
    build_preprocessing_inputs,
)
from crawler.curation.snapshots.alignment_rows import CuratedSnapshotRows
from crawler.curation.snapshots.dataset_assembly import (
    curated_assembly_types,
    curated_dataset_assembler,
)
from crawler.curation.snapshots.dataset_assembly.curated_quality_filter import (
    validate_curated_snapshot,
)
from crawler.curation.snapshots.dataset_assembly.curated_record_loader import (
    CURATED_INPUT_KINDS,
)
from crawler.curation.snapshots.dataset_assembly.curated_snapshot_fingerprint import (
    build_snapshot_fingerprint_payload,
)
from crawler.curation.snapshots.deduplication import CuratedSnapshotDeduplicator
from crawler.curation.snapshots.manifest import CuratedSnapshotManifest
from crawler.governance.domains.domain_governance_registry import (
    DomainGovernanceRegistry,
)
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.storage.datasets.run_layout.dataset_path_layout import snapshot_directory
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.assembly.text_chunk_splitter import TextChunkSplitter
from mmcrawler_datasets.schema import SplitAssigner
from orchestration.composition.curated_snapshot_services import (
    dataset_writer_factory,
    document_curator_factory,
    raw_manifest_reader_settings,
)
from orchestration.composition.preprocessing_dependencies import (
    build_multimodal_preprocessor,
)
from shared.runtime_primitives import Clock, IdGenerator

if TYPE_CHECKING:
    from config.settings.datasets import (
        DatasetPathSettings,
        RawManifestReaderSettings,
        SplitAssignerSettings,
    )
    from orchestration.workflow.curated_snapshot_runtime import CuratedSnapshotRuntime
    from preprocessing.multimodal_preprocessor import MultimodalPreprocessor


class _ConcreteCuratedSnapshotAssemblerFactory:
    """Build curated snapshot assemblers from one composed static graph.

    Stores exact dependencies only. The dynamic selection inputs
    (``raw_run_selection_mode`` / ``selected_run_ids``) are per-call.
    """

    def __init__(
        self,
        *,
        config: curated_assembly_types.CuratedDatasetAssemblerConfig,
        logger: ProjectLogger,
        dataset_paths: "DatasetPathSettings",
        project_root: Path,
        configured_raw_manifest_reader: "RawManifestReaderSettings",
        minimum_modality_counts: Mapping[str, int],
        snapshot_id_factory: partial[str],
        document_curator: curated_assembly_types.DocumentCuratorFactory,
        preprocessing_input_builder: curated_assembly_types.PreprocessingInputBuilder,
        preprocessing_phase_runner: "MultimodalPreprocessor",
        writer_factory: curated_assembly_types.DatasetWriterFactory,
        chunker: TextChunkSplitter,
    ) -> None:
        self._config = config
        self._logger = logger
        self._dataset_paths = dataset_paths
        self._project_root = project_root
        self._configured_raw_manifest_reader = configured_raw_manifest_reader
        self._minimum_modality_counts = dict(minimum_modality_counts)
        self._snapshot_id_factory = snapshot_id_factory
        self._document_curator = document_curator
        self._preprocessing_input_builder = preprocessing_input_builder
        self._preprocessing_phase_runner = preprocessing_phase_runner
        self._writer_factory = writer_factory
        self._chunker = chunker

    def __call__(
        self,
        *,
        raw_run_selection_mode: str | None = None,
        selected_run_ids: tuple[str, ...] | None = None,
    ) -> curated_dataset_assembler.CuratedDatasetAssembler:
        reader_settings = raw_manifest_reader_settings(
            configured=self._configured_raw_manifest_reader,
            raw_run_selection_mode=raw_run_selection_mode,
            selected_run_ids=selected_run_ids,
        )
        return curated_dataset_assembler.CuratedDatasetAssembler(
            config=self._config,
            dependencies=curated_assembly_types.CuratedDatasetAssemblerDependencies(
                logger=self._logger,
                raw_manifest_reader=CurationInputLoader(
                    settings=reader_settings,
                    project_root=self._project_root,
                    logger=self._logger,
                    dataset_paths=self._dataset_paths,
                    minimum_modality_counts=dict(self._minimum_modality_counts),
                ),
                snapshot_id_factory=self._snapshot_id_factory,
                snapshot_directory_resolver=snapshot_directory,
                document_curator_factory=self._document_curator,
                preprocessing_input_builder=self._preprocessing_input_builder,
                preprocessing_phase_runner=self._preprocessing_phase_runner,
                dataset_writer_factory=self._writer_factory,
                chunker=self._chunker,
                document_deduper=CuratedSnapshotDeduplicator.documents,
                image_deduper=CuratedSnapshotDeduplicator.images,
                media_row_deduper=CuratedSnapshotDeduplicator.media_rows,
                sync_row_assembler=CuratedSnapshotRows.build_sync,
                sync_row_deduper=CuratedSnapshotDeduplicator.sync_rows,
                snapshot_validator=validate_curated_snapshot,
                snapshot_manifest_writer=CuratedSnapshotManifest.write,
            ),
        )


def build_curated_snapshot_assembler_factory(
    *,
    settings: Settings,
    logger: ProjectLogger,
    clock: Clock,
    id_generator: IdGenerator,
    host_normalizer: HostNormalizer,
) -> _ConcreteCuratedSnapshotAssemblerFactory:
    """Build the curated assembler factory with every static dependency."""

    curation_settings = settings.datasets.curation
    split_assigner = _split_assigner(
        split_settings=settings.datasets.splits.curation
    )
    source_registry = DomainGovernanceRegistry(
        entries=settings.sources.active.governance,
        host_normalizer=host_normalizer,
        logger=logger,
    )
    multimodal_preprocessor = build_multimodal_preprocessor(
        settings=settings,
        logger=logger,
        clock=clock,
        id_generator=id_generator,
    )
    chunk_splitter = TextChunkSplitter(
        settings=settings.datasets.curation.document_chunker,
        assign_split=split_assigner.assign,
        logger=logger,
    )
    config = curated_assembly_types.CuratedDatasetAssemblerConfig(
        settings=curation_settings.builder,
        dataset_paths=settings.datasets.paths,
        project_root=settings.paths.root,
        relevant_kinds=CURATED_INPUT_KINDS,
        snapshot_fingerprint_payload=build_snapshot_fingerprint_payload(
            source_profile=settings.sources.active,
            preprocessing=settings.preprocessing,
            curation=settings.datasets.curation,
            processors=settings.collection.processors,
        ),
    )
    return _ConcreteCuratedSnapshotAssemblerFactory(
        config=config,
        logger=logger,
        dataset_paths=settings.datasets.paths,
        project_root=settings.paths.root,
        configured_raw_manifest_reader=settings.datasets.raw.manifest_reader,
        minimum_modality_counts={
            "page": settings.crawl_output_gate.minimum_records.page,
            "image": settings.crawl_output_gate.minimum_records.image,
            "document": settings.crawl_output_gate.minimum_records.document,
            "audio": settings.crawl_output_gate.minimum_records.audio,
            "video": settings.crawl_output_gate.minimum_records.video,
        },
        snapshot_id_factory=partial(
            _snapshot_id,
            clock=clock,
            id_generator=id_generator,
        ),
        document_curator=document_curator_factory(
            document_curator_settings=curation_settings.document_assembler,
            near_deduper_settings=curation_settings.near_deduper,
            artifacts_directory=settings.datasets.paths.artifacts_directory,
            source_registry=source_registry,
            logger=logger,
            clock=clock,
        ),
        preprocessing_input_builder=partial(
            build_preprocessing_inputs,
            max_input_bytes=settings.preprocessing.input_validation.max_input_bytes,
        ),
        preprocessing_phase_runner=multimodal_preprocessor,
        writer_factory=dataset_writer_factory(
            writer_settings=curation_settings.writer,
            dataset_paths=settings.datasets.paths,
        ),
        chunker=chunk_splitter,
    )


def build_curated_snapshot_runtime(
    *,
    settings: Settings,
    logger: ProjectLogger,
    clock: Clock,
    id_generator: IdGenerator,
) -> CuratedSnapshotRuntime:
    """Wire the workflow curated-snapshot runtime for one settings tree."""

    from orchestration.workflow.curated_snapshot_runtime import CuratedSnapshotRuntime

    processors = settings.collection.processors
    profile_log_fields = {
        "dataset_subdirectory": settings.datasets.paths.output_subdirectory,
        "image_ocr": processors.image.run_ocr,
        "audio_transcription": processors.audio.run_transcription,
        "video_transcription": processors.video.run_transcription,
        "document_ocr": processors.document.run_ocr,
        "require_allow_training": (
            settings.datasets.curation.document_assembler.require_allow_training
        ),
    }
    return CuratedSnapshotRuntime(
        logger=logger,
        assembler_factory=build_curated_snapshot_assembler_factory(
            settings=settings,
            logger=logger,
            clock=clock,
            id_generator=id_generator,
        ),
        profile_log_fields=profile_log_fields,
    )


def _snapshot_id(*, clock: Clock, id_generator: IdGenerator) -> str:
    timestamp = clock.now().strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{id_generator.generate()}"


def _split_assigner(*, split_settings: "SplitAssignerSettings") -> SplitAssigner:
    return SplitAssigner(
        train_ratio=split_settings.train_ratio,
        val_ratio=split_settings.val_ratio,
        test_ratio=split_settings.test_ratio,
    )
