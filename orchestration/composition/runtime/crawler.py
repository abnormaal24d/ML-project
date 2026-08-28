"""Crawler runtime composition.

``build_crawler`` is the single composition point for the crawler runtime.
It returns ``(crawler, dataset_writer)``; both objects are the actual
runtime artifacts handed to the application container.

This is a thin facade that delegates to subgraph builders for infrastructure,
governance, state, and execution services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings.root import Settings
from crawler.governance.processing_activity import ProcessingActivityRegistry
from logger.factory import ProjectLoggerFactory
from orchestration.bootstrap.run_context import RunContext
from orchestration.composition.runtime.crawler_graph import (
    build_crawler_graph,
)
from orchestration.errors import (
    ApplicationWiringError,
    BootstrapError,
)
from orchestration.resource_shutdown import ResourceShutdownManager
from shared.runtime_primitives import Clock, IdGenerator

if TYPE_CHECKING:
    from config.collection.processors import PageProcessorSettings
    from crawler.runtime.crawler import Crawler
    from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
    from datachecker.manifests.crawl_state_manifest_writer import (
        CrawlStateManifestWriter,
    )
    from logger.project_logger import ProjectLogger
    from orchestration.runtime_dependency_preflight import (
        OptionalDependencyReport,
    )


def build_crawler(
    *,
    settings: Settings,
    processing_activity_registry: ProcessingActivityRegistry,
    optional_dependency_report: OptionalDependencyReport,
    logger_factory: ProjectLoggerFactory,
    run_context: RunContext,
    shutdown_manager: ResourceShutdownManager,
    clock: Clock,
    id_generator: IdGenerator,
    crawl_attempt_id: str | None = None,
    crawl_state_manifest_writer: CrawlStateManifestWriter | None = None,
    page_settings_override: PageProcessorSettings | None = None,
) -> tuple[Crawler, DatasetWriter]:
    """Build the crawler runtime and its dataset writer.

    Delegates to subgraph builders for infrastructure, governance, state,
    and execution services.
    """
    try:
        crawler_logger = logger_factory.get_logger_for("crawler.composition")

        crawler_logger.debug(
            "runtime_services_build_started",
            dataset_subdirectory=(
                settings.datasets.paths.output_subdirectory or "multimodal"
            ),
            environment=settings.application.environment,
        )

        # Build complete crawler graph via subgraph composition
        graph = build_crawler_graph(
            settings=settings,
            processing_activity_registry=processing_activity_registry,
            logger_factory=logger_factory,
            run_context=run_context,
            shutdown_manager=shutdown_manager,
            clock=clock,
            id_generator=id_generator,
            crawl_attempt_id=crawl_attempt_id,
            crawl_state_manifest_writer=crawl_state_manifest_writer,
            page_settings_override=page_settings_override,
        )

        # Log composition result
        _log_runtime_composition_result(
            settings=settings,
            logger=crawler_logger,
            seed_tasks=graph.seed_plan.tasks,
            dependency_report=optional_dependency_report,
        )

        return graph.crawler, graph.dataset_writer

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
            component="crawler",
            cause=exc,
        ) from exc


def _log_runtime_composition_result(
    *,
    settings: Settings,
    logger: "ProjectLogger",
    seed_tasks: tuple,
    dependency_report: "OptionalDependencyReport",
) -> None:
    """Emit the runtime-ready summary log event."""

    state_settings = settings.crawler.state
    checkpoint_enabled = state_settings.enabled
    dead_letter_enabled = (
        state_settings.enabled and state_settings.dead_letter_enabled
    )
    dependency_status = dependency_report.optional_dependency_status
    metrics_enabled = bool(settings.collection.metrics.enabled)
    processors = settings.collection.processors
    dataset_subdirectory = (
        settings.datasets.paths.output_subdirectory or "multimodal"
    )
    features = [
        name
        for name, enabled in (
            ("state", checkpoint_enabled),
            ("dead_letters", dead_letter_enabled),
            ("metrics", metrics_enabled),
        )
        if enabled
    ]
    disabled_features = [
        name
        for name, enabled in (
            ("image_ocr", processors.image.run_ocr),
            ("audio_transcription", processors.audio.run_transcription),
            ("video_transcription", processors.video.run_transcription),
            ("video_ocr", processors.video.run_ocr),
            ("video_keyframes", processors.video.keyframes.enabled),
            ("document_ocr", processors.document.run_ocr),
        )
        if not enabled
    ]
    autoscaler_settings = settings.collection.autoscaler
    workers = (
        f"{autoscaler_settings.min_workers}..{autoscaler_settings.max_workers}"
    )

    logger.info(
        "runtime_services_ready",
        message=(
            "runtime ready | "
            f"features={','.join(features) or 'none'} "
            f"workers={workers} seeds={len(seed_tasks)}"
        ),
        dataset_subdirectory=dataset_subdirectory,
        environment=settings.application.environment,
        seed_count=len(seed_tasks),
        features=tuple(features),
        disabled_features=tuple(disabled_features),
        workers=workers,
        deps=dependency_report.summary,
        state_persistence_enabled=checkpoint_enabled,
        dead_letter_persistence_enabled=dead_letter_enabled,
        metrics_enabled=metrics_enabled,
        min_workers=autoscaler_settings.min_workers,
        max_workers=autoscaler_settings.max_workers,
        image_ocr=processors.image.run_ocr,
        audio_transcription=processors.audio.run_transcription,
        video_transcription=processors.video.run_transcription,
        video_ocr=processors.video.run_ocr,
        document_ocr=processors.document.run_ocr,
        **dependency_status,
    )

    if not dependency_status.get(
        "ffmpeg_available"
    ) and not dependency_status.get("media_decoder_available"):
        logger.warning(
            "optional_dependency_missing",
            dependency=settings.augmentation.video.ffmpeg_exporter,
            impact="audio_video_metadata_probe_limited",
            suggested_action="install_ffmpeg",
        )


__all__ = ["build_crawler"]
