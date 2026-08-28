"""Application containers and dependency assembly."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from config.path_resolution.project_paths import normalize_project_path
from config.settings.root import Settings
from datachecker.workflow_decision import WorkflowAction
from orchestration.composition.adapters.runtime_dependencies import (
    build_runtime_primitives,
)
from orchestration.errors import (
    ApplicationContainerBuildError,
    BootstrapBuildFailure,
    BootstrapError,
    SettingsLoadError,
)
from orchestration.resource_shutdown import ResourceShutdownManager
from orchestration.settings_loader import load, validate_runtime_configuration

if TYPE_CHECKING:
    from pathlib import Path

    from config.collection.processors import PageProcessorSettings
    from crawler.runtime.crawler import Crawler
    from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
    from datachecker.manifests.crawl_state_manifest_writer import (
        CrawlStateManifestWriter,
    )
    from logger.factory import ProjectLoggerFactory
    from orchestration.bootstrap.run_context import RunContext
    from orchestration.bootstrap.workflow_executor import (
        WorkflowPhaseExecutor,
    )
    from orchestration.cli.argument_parser import RuntimeOptions
    from orchestration.settings_loader import RuntimeReadiness
    from orchestration.workflow.crawl.phase_runner import CrawlExecutionResult


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    """Runtime dependencies and managed resources for one crawler run."""

    crawler: Crawler
    dataset_writer: DatasetWriter
    logger_factory: ProjectLoggerFactory
    shutdown_manager: ResourceShutdownManager

    async def aclose(self) -> None:
        """Close all shutdown-managed runtime resources."""

        await self.shutdown_manager.aclose()


def build_application_container(
    *,
    project_root: Path,
    environment: str,
    run_context: RunContext,
    settings: Settings | None = None,
    configure_logging: bool = True,
    crawl_attempt_id: str | None = None,
    crawl_state_manifest_writer: CrawlStateManifestWriter | None = None,
    config_root: Path | None = None,
    runtime_readiness: RuntimeReadiness | None = None,
    page_settings_override: PageProcessorSettings | None = None,
) -> ApplicationContainer:
    """Load settings and assemble one crawler application container."""

    shutdown_manager: ResourceShutdownManager | None = None

    try:
        if settings is None:
            try:
                loaded_settings = load(
                    project_root=project_root,
                    config_root=config_root,
                    environment=environment,
                    profile=environment,
                )
            except SettingsLoadError:
                raise
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                raise SettingsLoadError(
                    str(exc),
                    stage="bootstrap",
                    component="settings",
                    cause=exc,
                ) from exc
        else:
            loaded_settings = settings
            _validate_supplied_settings_identity(
                settings=loaded_settings,
                project_root=project_root,
                environment=environment,
            )

        readiness = runtime_readiness or validate_runtime_configuration(
            settings=loaded_settings,
            config_root=config_root,
        )

        shutdown_manager = ResourceShutdownManager(
            shutdown_timeout_seconds=(
                loaded_settings.application.resource_shutdown_timeout_seconds
            ),
        )

        from orchestration.bootstrap.logging import (
            build_logger_factory,
        )

        logger_factory = build_logger_factory(
            settings=loaded_settings,
            context=run_context,
            configure=configure_logging,
        )

        logger = logger_factory.get_logger(__name__)
        logger.debug(
            "settings_loaded_initializing_container",
            environment=loaded_settings.application.environment,
        )

        try:
            from orchestration.composition.runtime.crawler import build_crawler

            runtime_primitives = build_runtime_primitives()
            crawler, dataset_writer = build_crawler(
                settings=loaded_settings,
                processing_activity_registry=(
                    readiness.processing_activity_registry
                ),
                optional_dependency_report=readiness.dependency_report,
                logger_factory=logger_factory,
                run_context=run_context,
                shutdown_manager=shutdown_manager,
                clock=runtime_primitives.clock,
                id_generator=runtime_primitives.id_generator,
                crawl_attempt_id=crawl_attempt_id,
                crawl_state_manifest_writer=crawl_state_manifest_writer,
                page_settings_override=page_settings_override,
            )

        except BootstrapError as exc:
            logger.error(
                "application_runtime_wiring_failed",
                component=exc.component,
                issue=exc.kind,
            )
            raise

        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.exception(
                "application_runtime_build_failed",
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )
            raise ApplicationContainerBuildError(
                str(exc),
                stage="bootstrap",
                component="container",
                cause=exc,
            ) from exc

        logger.debug(
            "application_container_built",
            environment=loaded_settings.application.environment,
        )

        return ApplicationContainer(
            crawler=crawler,
            dataset_writer=dataset_writer,
            logger_factory=logger_factory,
            shutdown_manager=shutdown_manager,
        )

    except BootstrapError as exc:
        raise BootstrapBuildFailure(
            build_error=exc,
            shutdown_manager=shutdown_manager,
        ) from exc

    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise BootstrapBuildFailure(
            build_error=exc,
            shutdown_manager=shutdown_manager,
        ) from exc


def _validate_supplied_settings_identity(
    *,
    settings: Settings,
    project_root: Path,
    environment: str,
) -> None:
    """Reject a caller request that disagrees with injected settings.

    A supplied settings tree is the concrete configuration used to assemble
    the runtime. Accepting different selector values would make the public
    application boundary report one workspace or environment while writing to
    another, so verify both before constructing any dependencies.
    """

    if normalize_project_path(settings.paths.root) != normalize_project_path(
        project_root
    ):
        raise ValueError(
            "supplied settings project_root does not match requested "
            "project_root"
        )

    requested_environment = str(environment).strip().lower()
    if settings.application.resolved_environment() != requested_environment:
        raise ValueError(
            "supplied settings environment does not match requested "
            "environment"
        )


def build_workflow_phase_executor(
    options: RuntimeOptions,
    *,
    workflow_context: RunContext,
    settings: Settings,
    runtime_readiness: RuntimeReadiness,
    execute_crawl_application: Callable[..., Awaitable[CrawlExecutionResult]],
) -> WorkflowPhaseExecutor:
    """Assemble the autonomous DataChecker-driven workflow executor."""

    if options.command != "run":
        raise ValueError("Only the run command can build a workflow executor.")

    from config.path_resolution.project_paths import ProjectPaths
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from config.settings.fingerprint_sections import (
        build_settings_payloads,
    )
    from crawler.coverage.focus import focus_kinds, focused_page_settings
    from crawler.coverage.gaps import CoverageGapAnalyzer
    from crawler.coverage.progress import CoverageProgressTracker
    from crawler.storage.datasets.run_layout.dataset_path_layout import (
        snapshot_directory,
    )
    from datachecker.fingerprints import (
        SettingsFingerprintCalculator,
        SourceFingerprintCalculator,
    )
    from mmcrawler_datasets.snapshots.training_builder import (
        build_training_snapshot,
    )
    from orchestration.bootstrap.crawl_state_reconciler import (
        reconcile_crawl_state,
    )
    from orchestration.bootstrap.logging import (
        build_logger_factory,
    )
    from orchestration.bootstrap.workflow_executor import (
        WorkflowPhaseExecutor,
        run_blocking,
    )
    from orchestration.composition.curated_snapshot import (
        build_curated_snapshot_runtime,
    )
    from orchestration.composition.preprocessing_dependencies import (
        build_audio_materializer_factory,
        build_video_materializer_factory,
    )
    from orchestration.composition.privacy.privacy_inspection_services import (
        build_privacy_inspection_services,
    )
    from orchestration.composition.runtime.augmentation import (
        build_augmentation_workflow,
    )
    from orchestration.composition.runtime.dataset import (
        build_data_checker,
    )
    from orchestration.composition.runtime.training_workflow import (
        build_training_workflow,
    )
    from orchestration.composition.runtime.workflow_manifest_writers import (
        build_workflow_manifest_writers,
    )
    from orchestration.workflow.augmentation.phase_runner import (
        AugmentPhaseRunner,
    )
    from orchestration.workflow.crawl.phase_runner import (
        CrawlPhaseRunner,
    )
    from orchestration.workflow.dataset_preprocessing import (
        run_preprocessing_phase,
    )
    from orchestration.workflow.preprocessing.phase_runner import (
        PreprocessPhaseRunner,
    )
    from training.runtime.checkpoint.contract import CheckpointContract

    runtime_primitives = build_runtime_primitives()
    clock = runtime_primitives.clock
    id_generator = runtime_primitives.id_generator

    logger_factory = build_logger_factory(
        settings=settings,
        context=workflow_context,
    )

    logger = logger_factory.get_logger(__name__)
    logger.debug(
        "workflow_executor_dependencies_building",
        environment=settings.application.environment,
        fresh_run=options.fresh_run,
        resume=options.resume,
        use_cuda=options.use_cuda,
    )

    config_root = options.config_root

    checker = build_data_checker(
        settings=settings,
        logger_factory=logger_factory,
    )

    manifest_writers = build_workflow_manifest_writers(
        settings=settings,
        logger_factory=logger_factory,
        clock=clock,
        id_generator=id_generator,
        run_context=workflow_context,
    )

    app = settings.application
    blocking_semaphore = asyncio.Semaphore(app.workflow_blocking_task_limit)
    run_blocking_bound = partial(run_blocking, blocking_semaphore)
    io_timeout_seconds = float(app.workflow_io_timeout_seconds)

    executor_logger = logger_factory.get_logger_for(WorkflowPhaseExecutor)

    augmentation_workflow = build_augmentation_workflow(
        settings=settings,
        logger=logger,
    )

    # Bootstrap binds the process-specific crawl lifecycle parameters up
    # front; the workflow runner only supplies runtime crawl inputs. The
    # explicit adapter (not a cast partial) translates the workflow's
    # focused page policy into the application-level override, and reuses
    # the canonical workflow settings so fingerprints and the executed
    # crawler can never diverge.
    async def execute_crawl(
        *,
        crawl_attempt_id: str,
        crawl_state_manifest_writer: CrawlStateManifestWriter,
        page_settings: PageProcessorSettings,
    ) -> "CrawlExecutionResult":
        return await execute_crawl_application(
            project_root=settings.paths.root,
            environment=str(settings.application.environment or "dev"),
            stage="crawl",
            settings=settings,
            configure_logging=False,
            parent_run_context=workflow_context,
            crawl_attempt_id=crawl_attempt_id,
            crawl_state_manifest_writer=crawl_state_manifest_writer,
            config_root=(
                None if config_root is None else config_root.resolve()
            ),
            runtime_readiness=runtime_readiness,
            page_settings_override=page_settings,
        )

    # Crawl input fingerprints are canonical-workflow constants; compute
    # them once here so the workflow runner never touches config.
    crawl_settings_fingerprint_calculator = SettingsFingerprintCalculator()
    crawl_source_fingerprint_calculator = SourceFingerprintCalculator(
        settings_fingerprint_calculator=crawl_settings_fingerprint_calculator,
    )
    crawl_payloads = build_settings_payloads(
        settings=settings,
        checker_settings=settings.collection.datachecker,
    )
    crawl_source_registry_hash = crawl_source_fingerprint_calculator.calculate(
        seed_urls=tuple(settings.sources.active.seed_urls),
        source_profile=settings.sources.active,
    )
    crawl_settings_hash = crawl_settings_fingerprint_calculator.calculate(
        payload=crawl_payloads.crawl,
    )

    coverage_gap_analyzer = CoverageGapAnalyzer(settings=settings.coverage)
    resolve_focus_kinds = partial(
        focus_kinds,
        settings=settings.coverage,
    )
    resolve_focused_page_settings = partial(
        focused_page_settings,
        page_settings=settings.collection.processors.page,
        focus_settings=settings.coverage.focus,
    )

    # Preprocessing: every static config slice and service is bound here;
    # the workflow layer only sees typed capabilities.
    gap_analyzer = CoverageGapAnalyzer(settings=settings.coverage)
    progress_tracker = CoverageProgressTracker(settings=settings.coverage)

    privacy_services = build_privacy_inspection_services(
        settings=settings.preprocessing.privacy_detection,
    )
    audio_materializer_factory = build_audio_materializer_factory(
        settings=settings.multimodal.audio_tokenizer,
    )
    video_materializer_factory = build_video_materializer_factory(
        settings=settings.multimodal.video_generator,
    )

    curated_runtime = build_curated_snapshot_runtime(
        settings=settings,
        logger=executor_logger,
        clock=clock,
        id_generator=id_generator,
    )

    training_directory_for = partial(
        snapshot_directory,
        project_root=settings.paths.root,
        base_output_directory=(
            settings.datasets.paths.training_output_directory
        ),
        configured_subdirectory=(settings.datasets.paths.output_subdirectory),
    )
    training_snapshot_builder = partial(
        build_training_snapshot,
        settings=settings.datasets.training.snapshot_builder,
        split_settings=settings.datasets.splits.training,
        validator_settings=settings.datasets.training.dataset_validator,
        dataset_paths=settings.datasets.paths,
        training_settings=settings.training,
        project_root=settings.paths.root,
        logger=executor_logger,
        pii_detector=privacy_services.pii_detector,
        output_settings=settings.datasets.training.writer,
        audio_materializer_factory=audio_materializer_factory,
        video_materializer_factory=video_materializer_factory,
        require_transcript_for_audio_text_pair=(
            settings.preprocessing.audio_validation.require_transcript_for_audio_text_pair
        ),
    )
    preprocess_dataset = partial(
        run_preprocessing_phase,
        manifest_filename=settings.datasets.paths.manifest_filename,
        raw_run_selection_mode=(
            settings.datasets.raw.manifest_reader.run_selection_mode
        ),
        build_curated_snapshot=curated_runtime.build,
        resolve_training_directory=training_directory_for,
        build_training_snapshot=training_snapshot_builder,
    )

    artifact_paths = ArtifactPathRegistry(
        settings=settings.collection.datachecker,
        dataset_paths=settings.datasets.paths,
    )
    augmentation_output_root = artifact_paths.augmented_training_sets_root()

    runners = {
        WorkflowAction.CRAWL: CrawlPhaseRunner(
            logger=executor_logger,
            run_blocking=run_blocking_bound,
            io_timeout_seconds=io_timeout_seconds,
            source_registry_hash=crawl_source_registry_hash,
            crawl_settings_hash=crawl_settings_hash,
            missing_by_media_kind=(
                coverage_gap_analyzer.missing_by_media_kind
            ),
            resolve_focus_kinds=resolve_focus_kinds,
            resolve_focused_page_settings=resolve_focused_page_settings,
            crawl_state_manifest_writer=(
                manifest_writers.crawl_state_manifest_writer
            ),
            commit_crawl=manifest_writers.crawl_promotion.commit,
            execute_crawl=execute_crawl,
        ).run,
        WorkflowAction.PREPROCESS: PreprocessPhaseRunner(
            logger=executor_logger,
            gap_analyzer=gap_analyzer,
            progress_tracker=progress_tracker,
            run_preprocessing=preprocess_dataset,
            write_preprocessing_manifest=(
                manifest_writers.preprocessing.write_preprocessing_manifest
            ),
            run_blocking=run_blocking_bound,
            io_timeout_seconds=io_timeout_seconds,
        ).run,
        WorkflowAction.AUGMENT: AugmentPhaseRunner(
            logger=executor_logger,
            write_augmentation_manifest=(
                manifest_writers.augmentation.write_augmentation_manifest
            ),
            run_blocking=run_blocking_bound,
            io_timeout_seconds=io_timeout_seconds,
            augmentation_workflow=augmentation_workflow,
            clock=clock,
            dataset_paths=settings.datasets.paths,
            writer_settings=settings.datasets.training.writer,
            project_paths=ProjectPaths(project_root=settings.paths.root),
            output_root=augmentation_output_root,
        ).run,
        # TrainPhaseRunner.run(plan) uses plan.training_snapshot_id as the
        # canonical TrainingJobIdentity.snapshot_id (never training_root.name).
        WorkflowAction.TRAIN: build_training_workflow(
            settings=settings,
            logger=executor_logger,
            logger_factory=logger_factory,
            manifest_writers=manifest_writers,
            run_blocking=run_blocking_bound,
            clock=clock,
            id_generator=id_generator,
            checkpoint_contract=CheckpointContract(
                checkpoint_headers=options.checkpoint_headers,
                checkpoint_blob_storage=options.checkpoint_blob_storage,
                staging_lock=options.staging_lock,
            ),
        ).run,
    }

    check_workflow = partial(
        checker.check,
        timeout_seconds=float(app.data_checker_timeout_seconds),
    )

    cleanup: Callable[[], Awaitable[list[str]]] | None = None
    if options.fresh_run:
        from orchestration.bootstrap.runtime_cleanup import (
            clean_runtime_state,
        )

        cleanup = partial(
            run_blocking_bound,
            clean_runtime_state,
            settings=settings,
            timeout_seconds=io_timeout_seconds,
        )

    reconcile_crawl: Callable[[], Awaitable[bool]] | None = None
    if options.resume:
        reconcile_crawl = partial(
            reconcile_crawl_state,
            settings=settings,
            logger=executor_logger,
            manifest_writers=manifest_writers,
            run_blocking=run_blocking_bound,
            io_timeout_seconds=io_timeout_seconds,
        )

    executor = WorkflowPhaseExecutor(
        check=check_workflow,
        runners=runners,
        cleanup=cleanup,
        reconcile_crawl=reconcile_crawl,
        max_iterations=app.max_workflow_iterations,
        iteration_pause_seconds=float(app.workflow_iteration_pause_seconds),
        logger=executor_logger,
    )

    logger.debug(
        "workflow_executor_dependencies_built",
        environment=settings.application.environment,
    )

    return executor
