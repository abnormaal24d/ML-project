"""Build concrete workflow manifest writers for workflow runs."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

from config.path_resolution.workflow_artifact_paths import ArtifactPathRegistry
from config.settings.fingerprint_sections import (
    SettingsPayloads,
    build_settings_payloads,
)
from config.settings.root import Settings
from datachecker.fingerprints import (
    DatasetFingerprintCalculator,
    FileFingerprintCalculator,
    ProjectFingerprintCalculator,
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
from datachecker.manifests.artifact_manifest import RunArtifactIdentity
from datachecker.manifests.augmentation_manifest_writer import (
    AugmentationManifestWriter,
)
from datachecker.manifests.crawl_attempt_artifacts import CrawlAttemptArtifacts
from datachecker.manifests.crawl_manifest import CrawlManifest
from datachecker.manifests.crawl_manifest_writer import CrawlManifestWriter
from datachecker.manifests.crawl_state_manifest import CrawlStateManifest
from datachecker.manifests.crawl_state_manifest_writer import (
    CrawlStateManifestWriter,
)
from datachecker.manifests.crawl_state_reference_resolver import (
    CrawlStateReferenceResolver,
)
from datachecker.manifests.manifest_file_writer import (
    GenerateId,
    ManifestFileWriter,
    Now,
)
from datachecker.manifests.preprocessing_manifest_writer import (
    PreprocessingManifestWriter,
)
from datachecker.manifests.training_artifact_manifest_writer import (
    TrainingArtifactManifestWriter,
)
from logger.factory import ProjectLoggerFactory
from orchestration.bootstrap.run_context import RunContext
from orchestration.composition.runtime.dataset import (
    resolve_dataset_paths,
)
from shared.runtime_primitives import Clock, IdGenerator


class CrawlPromotionCommitter:
    """Commit one raw crawl attempt to canonical workflow state."""

    def __init__(
        self,
        *,
        crawl_writer: CrawlManifestWriter,
        crawl_state_manifest_writer: CrawlStateManifestWriter,
    ) -> None:
        self._crawl_writer = crawl_writer
        self._crawl_state_manifest_writer = crawl_state_manifest_writer

    def commit(
        self,
        *,
        state: CrawlStateManifest,
        attempt: CrawlAttemptArtifacts,
    ) -> CrawlManifest:
        if (
            not state.attempt_id
            or not state.raw_run_id
            or not state.crawl_session_id
        ):
            raise RuntimeError(
                "crawl promotion requires complete crawl identity"
            )

        if not attempt.has_complete_files_on_disk():
            raise RuntimeError(
                "crawl promotion requires complete raw artifacts"
            )

        raw_run_directory = attempt.raw_run_directory
        run_summary_path = attempt.run_summary_path

        if raw_run_directory is None or run_summary_path is None:
            raise RuntimeError(
                "complete crawl artifacts unexpectedly lack paths"
            )

        if (
            state.raw_run_directory != raw_run_directory
            or state.run_summary_path != run_summary_path
        ):
            raise RuntimeError(
                "crawl state and resolved attempt reference different "
                "raw artifacts"
            )

        self._crawl_state_manifest_writer.write_crawl_finalization_started(
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
        )

        crawl_manifest = self._crawl_writer.write_crawl_manifest(
            raw_run_directory=raw_run_directory,
            run_summary_path=run_summary_path,
            attempt_id=state.attempt_id,
            raw_run_id=state.raw_run_id,
            crawl_session_id=state.crawl_session_id,
        )

        if not self._crawl_writer.crawl_manifest_path().is_file():
            raise RuntimeError(
                "crawl promotion did not persist its canonical manifest"
            )

        self._crawl_state_manifest_writer.write_crawl_state_succeeded(
            crawl_manifest=crawl_manifest,
        )

        return crawl_manifest

    def resume_existing(
        self,
        *,
        state: CrawlStateManifest,
    ) -> CrawlManifest | None:
        manifest = self._crawl_state_manifest_writer.existing_crawl_manifest()
        if manifest is None:
            return None
        if manifest.identity_fields() != state.identity_fields():
            return None
        if (
            state.raw_run_directory != manifest.raw_run_directory
            or state.run_summary_path != manifest.run_summary_path
        ):
            return None
        if not self._crawl_writer.crawl_manifest_path().is_file():
            return None
        self._crawl_state_manifest_writer.write_crawl_state_succeeded(
            crawl_manifest=manifest,
        )
        return manifest


@dataclass(frozen=True, slots=True)
class WorkflowManifestWriters:
    """Concrete manifest writers used by workflow phases."""

    crawl: CrawlManifestWriter
    crawl_state_manifest_writer: CrawlStateManifestWriter
    crawl_promotion: CrawlPromotionCommitter
    preprocessing: PreprocessingManifestWriter
    augmentation: AugmentationManifestWriter
    training: TrainingArtifactManifestWriter


def build_workflow_manifest_writers(
    *,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    clock: Clock,
    id_generator: IdGenerator,
    run_context: RunContext,
) -> WorkflowManifestWriters:
    """
    Build manifest writers with shared readers and fingerprint calculators.
    """

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
    source_fingerprint_calculator = SourceFingerprintCalculator(
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
    payloads = build_settings_payloads(
        settings=settings,
        checker_settings=checker_settings,
    )
    artifact_identity = _build_artifact_identity(
        settings=settings,
        payloads=payloads,
        run_context=run_context,
        settings_fingerprint_calculator=settings_fingerprint_calculator,
    )
    now: Now = clock.now
    generate_id: GenerateId = id_generator.generate
    file_writer = _build_manifest_file_writer(
        settings=settings,
        now=now,
        generate_id=generate_id,
    )
    return _assemble_manifest_writers(
        settings=settings,
        logger_factory=logger_factory,
        artifact_path_registry=artifact_path_registry,
        raw_inventory_reader=raw_inventory_reader,
        curated_inventory_reader=curated_inventory_reader,
        training_inventory_reader=training_inventory_reader,
        settings_fingerprint_calculator=settings_fingerprint_calculator,
        source_fingerprint_calculator=source_fingerprint_calculator,
        file_fingerprint_calculator=file_fingerprint_calculator,
        payloads=payloads,
        artifact_identity=artifact_identity,
        file_writer=file_writer,
        now=now,
        generate_id=generate_id,
    )


def _assemble_manifest_writers(
    *,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    artifact_path_registry: ArtifactPathRegistry,
    raw_inventory_reader: RawInventoryReader,
    curated_inventory_reader: CuratedInventoryReader,
    training_inventory_reader: TrainingInventoryReader,
    settings_fingerprint_calculator: SettingsFingerprintCalculator,
    source_fingerprint_calculator: SourceFingerprintCalculator,
    file_fingerprint_calculator: FileFingerprintCalculator,
    payloads: SettingsPayloads,
    artifact_identity: RunArtifactIdentity,
    file_writer: ManifestFileWriter,
    now: Now,
    generate_id: GenerateId,
) -> WorkflowManifestWriters:
    """Construct the five phase-specific writers from shared services."""

    crawl = CrawlManifestWriter(
        artifact_path_registry=artifact_path_registry,
        raw_inventory_reader=raw_inventory_reader,
        settings_fingerprint_calculator=settings_fingerprint_calculator,
        source_fingerprint_calculator=source_fingerprint_calculator,
        file_fingerprint_calculator=file_fingerprint_calculator,
        crawl_settings_payload=dict(payloads.crawl),
        seed_urls=tuple(settings.sources.active.seed_urls),
        source_profile=settings.sources.active,
        logger=logger_factory.get_logger_for(CrawlManifestWriter),
        project_root=settings.paths.root,
        file_writer=file_writer,
        artifact_identity=artifact_identity,
        crawl_output_gate=settings.crawl_output_gate,
        now=now,
    )
    reference_resolver = CrawlStateReferenceResolver(
        artifact_path_registry=artifact_path_registry,
        logger=logger_factory.get_logger_for(CrawlStateReferenceResolver),
        project_root=settings.paths.root,
    )
    crawl_state_manifest_writer = CrawlStateManifestWriter(
        artifact_path_registry=artifact_path_registry,
        reference_resolver=reference_resolver,
        logger=logger_factory.get_logger_for(CrawlStateManifestWriter),
        project_root=settings.paths.root,
        file_writer=file_writer,
        artifact_identity=artifact_identity,
        now=now,
        generate_id=generate_id,
    )

    return WorkflowManifestWriters(
        crawl=crawl,
        crawl_state_manifest_writer=crawl_state_manifest_writer,
        crawl_promotion=CrawlPromotionCommitter(
            crawl_writer=crawl,
            crawl_state_manifest_writer=crawl_state_manifest_writer,
        ),
        preprocessing=PreprocessingManifestWriter(
            artifact_path_registry=artifact_path_registry,
            raw_inventory_reader=raw_inventory_reader,
            curated_inventory_reader=curated_inventory_reader,
            training_inventory_reader=training_inventory_reader,
            settings_fingerprint_calculator=settings_fingerprint_calculator,
            file_fingerprint_calculator=file_fingerprint_calculator,
            preprocessing_settings_payload=dict(payloads.preprocessing),
            normalization_settings_payload=dict(payloads.normalization),
            deduplication_settings_payload=dict(payloads.deduplication),
            splitting_settings_payload=dict(payloads.splitting),
            validation_settings_payload=dict(payloads.validation),
            logger=logger_factory.get_logger_for(PreprocessingManifestWriter),
            project_root=settings.paths.root,
            file_writer=file_writer,
            artifact_identity=artifact_identity,
            now=now,
        ),
        augmentation=AugmentationManifestWriter(
            artifact_path_registry=artifact_path_registry,
            curated_inventory_reader=curated_inventory_reader,
            training_inventory_reader=training_inventory_reader,
            settings_fingerprint_calculator=settings_fingerprint_calculator,
            augmentation_settings_payload=dict(payloads.augmentation),
            augmentation_strategy_payload=dict(payloads.augmentation_strategy),
            logger=logger_factory.get_logger_for(AugmentationManifestWriter),
            project_root=settings.paths.root,
            file_writer=file_writer,
            artifact_identity=artifact_identity,
            now=now,
        ),
        training=TrainingArtifactManifestWriter(
            artifact_path_registry=artifact_path_registry,
            settings_fingerprint_calculator=settings_fingerprint_calculator,
            training_settings_payload=dict(payloads.training),
            model_settings_payload=dict(payloads.model),
            logger=logger_factory.get_logger_for(
                TrainingArtifactManifestWriter
            ),
            project_root=settings.paths.root,
            file_writer=file_writer,
            artifact_identity=artifact_identity,
            release_stage=settings.training.release_stage,
            now=now,
        ),
    )


def _build_artifact_identity(
    *,
    settings: Settings,
    payloads: SettingsPayloads,
    run_context: RunContext,
    settings_fingerprint_calculator: SettingsFingerprintCalculator,
) -> RunArtifactIdentity:
    project_calculator = ProjectFingerprintCalculator()
    environment_name = settings.application.resolved_environment()
    environment_payload = {
        "environment_name": environment_name,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "implementation": sys.implementation.name,
    }
    return RunArtifactIdentity(
        generation_id=run_context.generation_id,
        workflow_id=run_context.workflow_id,
        project_fingerprint=project_calculator.calculate(
            project_root=settings.paths.root,
        ),
        config_fingerprint=settings_fingerprint_calculator.calculate(
            payload=payloads,
        ),
        environment_name=environment_name,
        environment_fingerprint=settings_fingerprint_calculator.calculate(
            payload=environment_payload,
        ),
        python_version=platform.python_version(),
        dependency_lock_fingerprint=(
            project_calculator.dependency_lock_fingerprint(
                project_root=settings.paths.root,
            )
        ),
    )


def _build_manifest_file_writer(
    *,
    settings: Settings,
    now: Now,
    generate_id: GenerateId,
) -> ManifestFileWriter:
    """Build the common manifest file writer."""

    return ManifestFileWriter(
        now=now,
        generate_id=generate_id,
        replace_retry_attempts=(
            settings.application.manifest_replace_retry_attempts
        ),
        replace_retry_delay_seconds=(
            settings.application.manifest_replace_retry_delay_seconds
        ),
        replace_retry_jitter_seconds=(
            settings.application.manifest_replace_retry_jitter_seconds
        ),
    )
