"""Crawler execution subgraph composition.

Assembles execution services from extracted subgraph builders.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.runtime.control.crawler_control_directory import (
        CrawlerControlDirectory,
    )
    from logger.factory import ProjectLoggerFactory
    from orchestration.bootstrap.run_context import RunContext
    from orchestration.composition.runtime.crawler_governance import (
        CrawlerGovernance,
    )
    from orchestration.composition.runtime.crawler_infrastructure import (
        CrawlerInfrastructure,
    )
    from orchestration.composition.runtime.crawler_state import (
        CrawlerStatePersistence,
    )
    from orchestration.resource_shutdown import ResourceShutdownManager

from config.settings.root import Settings
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)
from logger.factory import ProjectLoggerFactory
from orchestration.bootstrap.run_context import RunContext
from orchestration.composition.runtime.crawler_execution.contracts import (
    CrawlerExecutionOverrides,
    CrawlerExecutionServices,
)
from orchestration.composition.runtime.crawler_execution.extraction import (
    build_extraction_runtime,
)
from orchestration.composition.runtime.crawler_execution.feedback import (
    build_execution_feedback,
)
from orchestration.composition.runtime.crawler_execution.processing import (
    build_processing_runtime,
)
from orchestration.composition.runtime.crawler_execution.session import (
    build_runtime_session_factory,
)
from orchestration.composition.runtime.crawler_execution.state_runtime import (
    build_execution_state_runtime,
)
from orchestration.composition.runtime.crawler_execution.workers import (
    build_worker_runtime,
)
from orchestration.composition.runtime.crawler_governance import (
    CrawlerGovernance,
)
from orchestration.composition.runtime.crawler_infrastructure import (
    CrawlerInfrastructure,
)
from orchestration.composition.runtime.crawler_state import (
    CrawlerStatePersistence,
)
from orchestration.resource_shutdown import ResourceShutdownManager
from shared.runtime_primitives import IdGenerator


def build_crawler_execution(
    *,
    settings: Settings,
    infrastructure: CrawlerInfrastructure,
    governance: CrawlerGovernance,
    state: CrawlStatePersistence,
    logger_factory: ProjectLoggerFactory,
    shutdown_manager: ResourceShutdownManager,
    run_context: RunContext,
    control_directory: CrawlControlDirectory,
    id_generator: IdGenerator,
    seed_tasks: tuple[CrawlTask, ...],
    overrides: CrawlerExecutionOverrides | None = None,
) -> CrawlerExecutionServices:
    """Build crawler execution services."""
    overrides = overrides or CrawlerExecutionOverrides()

    project_root = Path(settings.paths.root)
    url_normalizer = infrastructure.url_normalizer
    clock = infrastructure.clock

    page_content_extractor, discovery_task_builder = build_extraction_runtime(
        settings=settings,
        url_normalizer=url_normalizer,
        logger_factory=logger_factory,
        id_generator=id_generator,
    )
    scheduler, dataset_writer, task_processor = build_processing_runtime(
        project_root=project_root,
        settings=settings,
        logger_factory=logger_factory,
        infrastructure=infrastructure,
        governance=governance,
        state=state,
        clock=clock,
        shutdown_manager=shutdown_manager,
        run_context=run_context,
        url_normalizer=url_normalizer,
        id_generator=id_generator,
        page_content_extractor=page_content_extractor,
        discovery_task_builder=discovery_task_builder,
        overrides=overrides,
    )
    worker_pool, worker_scaler = build_worker_runtime(
        settings=settings,
        logger_factory=logger_factory,
        scheduler=scheduler,
        task_processor=task_processor,
    )
    state_writer, seed_enqueuer = build_execution_state_runtime(
        settings=settings,
        logger_factory=logger_factory,
        scheduler=scheduler,
        state=state,
        seed_tasks=seed_tasks,
        worker_pool=worker_pool,
        metrics=infrastructure.metrics,
    )
    task_feedback = build_execution_feedback(
        logger_factory=logger_factory,
        scheduler=scheduler,
        robots_request_gate=governance.robots_request_gate,
        host_budget_tracker=infrastructure.host_budget_tracker,
        host_media_byte_budget=infrastructure.host_media_byte_budget,
        host_normalizer=infrastructure.host_normalizer,
        rate_limiter=infrastructure.rate_limiter,
        host_extractor=infrastructure.host_extractor,
    )
    create_runtime_session = build_runtime_session_factory(
        settings=settings,
        logger_factory=logger_factory,
        worker_pool=worker_pool,
        worker_scaler=worker_scaler,
        state_writer=state_writer,
        control_directory=control_directory,
        metrics=infrastructure.metrics,
        prometheus_exporter=infrastructure.prometheus_exporter,
        dataset_writer=dataset_writer,
        analysis_router=task_processor.analysis_router,
    )
    return CrawlerExecutionServices(
        scheduler=scheduler,
        worker_pool=worker_pool,
        worker_scaler=worker_scaler,
        dataset_writer=dataset_writer,
        seed_enqueuer=seed_enqueuer,
        build_runtime_session=create_runtime_session,
        control_directory=control_directory,
        task_feedback=task_feedback,
    )


__all__ = ["build_crawler_execution"]
