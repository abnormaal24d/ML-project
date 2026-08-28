"""Dataset composition: datachecker, dataset writer, and path resolution.

Owns the canonical DataChecker used by workflow dispatch, the crawler-run
dataset writer, and dataset path resolution.
"""

from __future__ import annotations

import shutil
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from config.collection.training_input_gate import TrainingInputMode
from config.path_resolution.project_paths import ProjectPaths
from config.path_resolution.workflow_artifact_paths import ArtifactPathRegistry
from config.settings.datasets import DatasetPathSettings
from config.settings.fingerprint_sections import (
    SettingsPayloads,
    build_settings_payloads,
)
from config.settings.root import Settings
from crawler.coverage.gaps import CoverageGapAnalyzer
from crawler.coverage.state import CoverageState
from crawler.curation.ingest.curation_input_loader import CurationInputLoader
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.governance.domains.domain_governance_registry import (
    DomainGovernanceRegistry,
)
from crawler.governance.processing_activity import ProcessingActivityRegistry
from crawler.storage.datasets.manifests.dataset_manifest_writer import (
    DatasetManifestWriter,
)
from crawler.storage.datasets.records.dataset_record import (
    DatasetRecordCreator,
)
from crawler.storage.datasets.records.record_index import (
    DatasetRecordIndex,
)
from crawler.storage.datasets.run_layout.dataset_path_layout import (
    build_run_directory,
)
from crawler.storage.datasets.sync_index.sync_index_compactor import (
    SyncIndexCompactor,
)
from crawler.storage.datasets.sync_index.sync_index_paths import SyncIndexPaths
from crawler.storage.datasets.sync_index.sync_index_reader import (
    SyncIndexReader,
)
from crawler.storage.datasets.sync_index.sync_index_updater import (
    SyncIndexUpdater,
)
from crawler.storage.datasets.writing.dataset_error_writer import (
    DatasetErrorWriter,
)
from crawler.storage.datasets.writing.dataset_run_finalizer import (
    DatasetRunFinalizer,
)
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from crawler.storage.datasets.writing.raw_payload_writer import (
    RawPayloadWriter,
)
from datachecker.data_checker import DataChecker, WorkflowFingerprints
from datachecker.fingerprints import (
    DatasetFingerprintCalculator,
    FileFingerprintCalculator,
    SettingsFingerprintCalculator,
    SourceFingerprintCalculator,
)
from datachecker.inventory.curated_snapshot_inventory import (
    CuratedInventoryReader,
)
from datachecker.inventory.raw_run_inventory import RawInventoryReader
from datachecker.inventory.training_snapshot_inventory import (
    TrainingInventoryReader,
    TrainingInventoryReaderConfig,
)
from datachecker.validation.augmentation_artifact_validator import (
    AugmentationArtifactValidator,
)
from datachecker.validation.crawl_artifact_validator import (
    CrawlArtifactValidator,
)
from datachecker.validation.preprocessing_artifact_validator import (
    PreprocessingArtifactValidator,
)
from datachecker.validation.training_artifact_validator import (
    TrainingArtifactValidator,
)
from datachecker.workflow_decision import (
    WorkflowAction,
)
from logger.factory import ProjectLoggerFactory
from orchestration.errors import (
    ApplicationWiringError,
    BootstrapError,
)
from orchestration.resource_shutdown import ResourceShutdownManager
from shared.runtime_primitives import Clock, IdGenerator

if TYPE_CHECKING:
    from crawler.fetching.response.cache import ConditionalRepresentationCache
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from datachecker.manifests.crawl_state_manifest_writer import (
        CrawlStateManifestWriter,
    )
    from orchestration.bootstrap.run_context import RunContext


def build_data_checker(
    *,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
) -> DataChecker:
    """Build the full datachecker graph with explicit DI."""

    checker_settings = settings.collection.datachecker
    dataset_paths = resolve_dataset_paths(
        paths=settings.datasets.paths,
        project_root=settings.paths.root,
    )
    artifact_path_registry = ArtifactPathRegistry(
        settings=checker_settings,
        dataset_paths=dataset_paths,
    )
    settings_fingerprint_calculator = SettingsFingerprintCalculator()
    file_fingerprint_calculator = FileFingerprintCalculator()
    dataset_fingerprint_calculator = DatasetFingerprintCalculator(
        file_fingerprint_calculator=file_fingerprint_calculator,
    )
    payloads = build_settings_payloads(
        settings=settings,
        checker_settings=checker_settings,
    )
    fingerprints = _workflow_fingerprints(
        settings=settings,
        payloads=payloads,
        settings_fingerprint_calculator=settings_fingerprint_calculator,
    )
    raw_inventory_reader = RawInventoryReader(
        settings=checker_settings,
        artifact_path_registry=artifact_path_registry,
        dataset_fingerprint_calculator=dataset_fingerprint_calculator,
        raw_schema_version=settings.datasets.schemas.raw_schema_version,
    )
    curated_inventory_reader = CuratedInventoryReader(
        settings=checker_settings,
        artifact_path_registry=artifact_path_registry,
        dataset_fingerprint_calculator=dataset_fingerprint_calculator,
    )
    training_inventory_reader = TrainingInventoryReader(
        settings=checker_settings,
        artifact_path_registry=artifact_path_registry,
        dataset_fingerprint_calculator=dataset_fingerprint_calculator,
        config=TrainingInventoryReaderConfig(
            splits_directory=dataset_paths.training_splits_directory,
            train_filename=dataset_paths.training_train_filename,
            val_filename=dataset_paths.training_val_filename,
            test_filename=dataset_paths.training_test_filename,
        ),
    )
    ordered_actions, optional_actions = _workflow_actions(
        training_input_mode=checker_settings.training_input_mode,
    )
    coverage_gap_analyzer = CoverageGapAnalyzer(settings=settings.coverage)

    project_root = settings.paths.root
    # Config validation ensures project_root is configured; see config/validation/cross_section/composition.py

    raw_manifest_loader = CurationInputLoader(
        settings=settings.datasets.raw.manifest_reader,
        dataset_paths=dataset_paths,
        project_root=project_root,
        logger=logger_factory.get_logger_for(CurationInputLoader),
        minimum_modality_counts={
            "page": settings.crawl_output_gate.minimum_records.page,
            "image": settings.crawl_output_gate.minimum_records.image,
            "document": settings.crawl_output_gate.minimum_records.document,
            "audio": settings.crawl_output_gate.minimum_records.audio,
            "video": settings.crawl_output_gate.minimum_records.video,
        },
    )

    return DataChecker(
        raw_inventory_reader=raw_inventory_reader,
        curated_inventory_reader=curated_inventory_reader,
        training_inventory_reader=training_inventory_reader,
        crawl_validator=CrawlArtifactValidator(
            minimum_output_files=checker_settings.min_crawl_output_files,
            minimum_modality_counts={
                "page": settings.crawl_output_gate.minimum_records.page,
                "image": settings.crawl_output_gate.minimum_records.image,
                "document": settings.crawl_output_gate.minimum_records.document,
                "audio": settings.crawl_output_gate.minimum_records.audio,
                "video": settings.crawl_output_gate.minimum_records.video,
            },
            minimum_raw_objects_total=(
                settings.crawl_output_gate.min_raw_objects_total
            ),
            minimum_successful_requests_total=(
                settings.crawl_output_gate.min_successful_requests_total
            ),
            minimum_quality_score=(
                settings.crawl_output_gate.min_quality_score
            ),
            selected_coverage_provider=(
                raw_manifest_loader.selected_modality_counts
            ),
            selected_evidence_provider=(
                raw_manifest_loader.selected_crawl_evidence
            ),
        ),
        preprocessing_validator=PreprocessingArtifactValidator(
            minimum_documents=checker_settings.min_preprocessed_documents,
            minimum_chunks=checker_settings.min_preprocessed_chunks,
            minimum_images=checker_settings.min_preprocessed_images,
            minimum_audio=checker_settings.min_preprocessed_audio,
            minimum_video=checker_settings.min_preprocessed_video,
            minimum_cross_modal_alignments=(
                checker_settings.min_cross_modal_alignments
            ),
            minimum_transcript_coverage=(
                checker_settings.min_transcript_coverage
            ),
            minimum_ocr_coverage=checker_settings.min_ocr_coverage,
            minimum_keyframe_coverage=checker_settings.min_keyframe_coverage,
        ),
        augmentation_validator=AugmentationArtifactValidator(),
        training_validator=TrainingArtifactValidator(
            minimum_samples=(
                settings.datasets.training.dataset_validator.min_total_samples
            ),
            minimum_modality_counts={
                "text": (
                    settings.datasets.training.dataset_validator.min_text_samples
                ),
                "image": (
                    settings.datasets.training.dataset_validator.min_image_samples
                ),
                "document": (
                    settings.datasets.training.dataset_validator.min_document_samples
                ),
                "audio": (
                    settings.datasets.training.dataset_validator.min_audio_samples
                ),
                "video": (
                    settings.datasets.training.dataset_validator.min_video_samples
                ),
            },
            minimum_task_counts=(
                settings.datasets.training.dataset_validator.effective_min_task_samples()
            ),
            require_autonomous_multimodal_readiness=(
                settings.datasets.training.dataset_validator.require_autonomous_multimodal_readiness
            ),
        ),
        artifact_path_registry=artifact_path_registry,
        file_fingerprint_calculator=file_fingerprint_calculator,
        fingerprints=fingerprints,
        coverage_gaps_resolver=coverage_gap_analyzer.gaps_from_validation_errors,
        augmentation_enabled=settings.augmentation.enabled,
        training_input_mode=checker_settings.training_input_mode,
        ordered_actions=ordered_actions,
        optional_actions=optional_actions,
        require_seed_urls=settings.sources.active.require_seed_urls,
        seed_url_count=len(settings.sources.active.seed_urls),
        logger=logger_factory.get_logger_for(DataChecker),
        monotonic_seconds=time.monotonic,
    )


def _workflow_fingerprints(
    *,
    settings: Settings,
    payloads: SettingsPayloads,
    settings_fingerprint_calculator: SettingsFingerprintCalculator,
) -> WorkflowFingerprints:
    """Calculate every workflow comparison hash from its owning subconfig."""

    source_fingerprint_calculator = SourceFingerprintCalculator(
        settings_fingerprint_calculator=settings_fingerprint_calculator,
    )
    return WorkflowFingerprints(
        source_registry=source_fingerprint_calculator.calculate(
            seed_urls=tuple(settings.sources.active.seed_urls),
            source_profile=settings.sources.active,
        ),
        crawl=settings_fingerprint_calculator.calculate(
            payload=payloads.crawl,
        ),
        preprocessing=settings_fingerprint_calculator.calculate(
            payload=payloads.preprocessing,
        ),
        normalization=settings_fingerprint_calculator.calculate(
            payload=payloads.normalization,
        ),
        deduplication=settings_fingerprint_calculator.calculate(
            payload=payloads.deduplication,
        ),
        splitting=settings_fingerprint_calculator.calculate(
            payload=payloads.splitting,
        ),
        validation=settings_fingerprint_calculator.calculate(
            payload=payloads.validation,
        ),
        augmentation=settings_fingerprint_calculator.calculate(
            payload=payloads.augmentation,
        ),
        augmentation_strategy=settings_fingerprint_calculator.calculate(
            payload=payloads.augmentation_strategy,
        ),
        training=settings_fingerprint_calculator.calculate(
            payload=payloads.training,
        ),
        model=settings_fingerprint_calculator.calculate(
            payload=payloads.model,
        ),
    )


def _workflow_actions(
    *,
    training_input_mode: TrainingInputMode,
) -> tuple[tuple[WorkflowAction, ...], tuple[WorkflowAction, ...]]:
    """Return the exact configured phase order and optional phases."""

    if training_input_mode is TrainingInputMode.PREPROCESSED_ONLY:
        return (
            (
                WorkflowAction.CRAWL,
                WorkflowAction.PREPROCESS,
                WorkflowAction.TRAIN,
            ),
            (),
        )
    if training_input_mode is TrainingInputMode.AUGMENTED_WHEN_AVAILABLE:
        return (
            (
                WorkflowAction.CRAWL,
                WorkflowAction.PREPROCESS,
                WorkflowAction.AUGMENT,
                WorkflowAction.TRAIN,
            ),
            (WorkflowAction.AUGMENT,),
        )
    if training_input_mode is TrainingInputMode.AUGMENTED_REQUIRED:
        return (
            (
                WorkflowAction.CRAWL,
                WorkflowAction.PREPROCESS,
                WorkflowAction.AUGMENT,
                WorkflowAction.TRAIN,
            ),
            (),
        )
    raise ValueError(
        f"unsupported training input mode: {training_input_mode!r}"
    )


def resolve_dataset_paths(
    *,
    paths: DatasetPathSettings,
    project_root: Path,
) -> DatasetPathSettings:
    """Resolve dataset paths against the project root."""

    path_resolver = ProjectPaths(project_root=project_root)
    return paths.model_copy(
        update={
            "workflow_artifacts_directory": path_resolver.resolve(
                paths.workflow_artifacts_directory,
            ),
            "raw_output_directory": path_resolver.resolve(
                paths.raw_output_directory,
            ),
            "curated_output_directory": path_resolver.resolve(
                paths.curated_output_directory,
            ),
            "training_output_directory": path_resolver.resolve(
                paths.training_output_directory,
            ),
            "augmented_training_output_directory": path_resolver.resolve(
                paths.augmented_training_output_directory,
            ),
            "training_checkpoint_directory": path_resolver.resolve(
                paths.training_checkpoint_directory,
            ),
        }
    )


def build_dataset_writer(
    *,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    coverage_tracker: CoverageState,
    url_normalizer: UrlNormalizer,
    shutdown_manager: ResourceShutdownManager,
    clock: Clock,
    id_generator: IdGenerator,
    host_normalizer: HostNormalizer,
    crawl_attempt_id: str | None = None,
    crawl_state_manifest_writer: CrawlStateManifestWriter | None = None,
    run_context: RunContext | None = None,
    processing_activity_registry: ProcessingActivityRegistry | None = None,
    processing_activity_id: str | None = None,
    conditional_representation_cache: ConditionalRepresentationCache,
) -> DatasetWriter:
    """Build the dataset writer graph and register its shutdown hook.

    This encapsulates the run directory, payload/manifest/sync writers, the
    record assemblers, finalizer, error writer and the top level DatasetWriter.

    P0.1: if a crawl attempt and its manifest writer are provided,
    immediately link the created raw run directory and summary to the
    attempt state.
    """

    try:
        with ExitStack() as construction:
            run_started_at = clock.now()
            dataset_run_id = (
                f"{run_started_at.strftime('%Y%m%dT%H%M%SZ')}-"
                f"{id_generator.generate()}"
            )
            dataset_started_at = run_started_at.isoformat()

            run_directory = build_run_directory(
                project_root=settings.paths.root,
                base_output_directory=(
                    settings.datasets.paths.raw_output_directory
                ),
                configured_subdirectory=(
                    settings.datasets.paths.output_subdirectory
                ),
                run_id=dataset_run_id,
            )
            construction.callback(
                shutil.rmtree, run_directory, ignore_errors=True
            )
            manifest_path = (
                run_directory / settings.datasets.paths.manifest_filename
            )

            sync_paths = SyncIndexPaths.from_settings(
                run_directory=run_directory,
                dataset_paths=settings.datasets.paths,
            )
            sync_paths.ensure_directories()
            run_summary_path = sync_paths.summary_path
            raw_run_id = dataset_run_id

            record_index = DatasetRecordIndex()
            payload_writer = RawPayloadWriter(
                settings=settings.datasets.raw.writer,
                dataset_paths=settings.datasets.paths,
                run_directory=run_directory,
            )
            manifest_writer = DatasetManifestWriter(
                settings=settings.datasets.raw.writer,
                manifest_path=manifest_path,
            )
            construction.callback(manifest_writer.close)

            sync_reader = SyncIndexReader(
                paths=sync_paths,
                dataset_paths=settings.datasets.paths,
            )
            construction.callback(
                sync_reader.close,
                fsync_enabled=settings.datasets.raw.writer.manifest_fsync_enabled,
            )

            sync_updater = SyncIndexUpdater(
                settings=settings.datasets.raw.writer,
                reader=sync_reader,
                run_directory=run_directory,
            )
            sync_compactor = SyncIndexCompactor(
                settings=settings.datasets.raw.writer,
                paths=sync_paths,
                reader=sync_reader,
                updater=sync_updater,
                manifest_writer=manifest_writer,
                record_index=record_index,
                started_at=dataset_started_at,
                run_identity={
                    "attempt_id": crawl_attempt_id,
                    "raw_run_id": raw_run_id,
                    "raw_run_directory": str(run_directory),
                    "run_summary_path": str(run_summary_path),
                    "generation_id": (
                        None
                        if run_context is None
                        else run_context.generation_id
                    ),
                    "crawl_session_id": (
                        None
                        if run_context is None
                        else run_context.crawl_session_id
                    ),
                    "workflow_id": (
                        None
                        if run_context is None
                        else run_context.workflow_id
                    ),
                },
            )

            governance_registry = DomainGovernanceRegistry(
                entries=settings.sources.active.governance,
                host_normalizer=host_normalizer,
                logger=logger_factory.get_logger_for(DomainGovernanceRegistry),
            )

            # Warn if a registry is available but no activity id was provided
            if (
                processing_activity_registry is not None
                and not processing_activity_id
            ):
                logger_factory.get_logger_for(DatasetWriter).warning(
                    "processing_activity_registry provided but processing_activity_id is missing; "
                    "this will fail-closed for training permissions"
                )

            record_creator = DatasetRecordCreator(
                settings=settings.datasets.raw.writer,
                run_id=dataset_run_id,
                now=clock.now,
                governance_registry=governance_registry,
                processing_activity_registry=processing_activity_registry,
                processing_activity_id=processing_activity_id,
            )

            dataset_logger = logger_factory.get_logger_for(DatasetWriter)
            run_finalizer = DatasetRunFinalizer(
                logger=dataset_logger,
                now=clock.now,
                run_id=dataset_run_id,
                run_directory=run_directory,
                manifest_path=manifest_path,
                manifest_writer=manifest_writer,
                sync_compactor=sync_compactor,
            )
            error_writer = DatasetErrorWriter(
                run_id=dataset_run_id,
                normalize_url=url_normalizer.normalize,
                now=clock.now,
                sync_updater=sync_updater,
                sync_compactor=sync_compactor,
                manifest_writer=manifest_writer,
            )

            dataset_writer = DatasetWriter(
                settings=settings.datasets.raw.writer,
                run_id=dataset_run_id,
                run_directory=run_directory,
                manifest_path=manifest_path,
                logger=dataset_logger,
                url_normalizer=url_normalizer,
                record_index=record_index,
                payload_writer=payload_writer,
                manifest_writer=manifest_writer,
                sync_updater=sync_updater,
                sync_compactor=sync_compactor,
                record_creator=record_creator,
                run_finalizer=run_finalizer,
                error_writer=error_writer,
                coverage_tracker=coverage_tracker,
                conditional_representation_cache=conditional_representation_cache,
            )
            shutdown_manager.add_step(
                name="dataset_writer",
                close=dataset_writer.aclose,
            )

            # P0.1: link attempt <-> raw run immediately upon creation of run dir
            if crawl_state_manifest_writer is not None:
                if crawl_attempt_id is None or run_context is None:
                    raise ApplicationWiringError(
                        "crawl state linking requires attempt and runtime identity",
                        stage="composition",
                        component="dataset_writer",
                    )
                crawl_state_manifest_writer.link_raw_run_to_attempt(
                    attempt_id=crawl_attempt_id,
                    raw_run_id=raw_run_id,
                    raw_run_directory=run_directory,
                    run_summary_path=run_summary_path,
                    generation_id=run_context.generation_id,
                    crawl_session_id=run_context.crawl_session_id,
                    workflow_id=run_context.workflow_id,
                )

            construction.pop_all()
            return dataset_writer
    except BootstrapError:
        raise
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ApplicationWiringError(
            str(exc),
            stage="composition",
            component="dataset_writer",
            cause=exc,
        ) from exc


__all__ = [
    "build_data_checker",
    "build_dataset_writer",
    "resolve_dataset_paths",
]
