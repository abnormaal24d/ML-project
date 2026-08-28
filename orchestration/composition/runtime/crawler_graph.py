"""Crawler graph assembly.

Assembles the final Crawler object from all subgraphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from config.settings.root import Settings
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)
from crawler.runtime.loop.crawl_run_supervisor import CrawlRunSupervisor
from logger.factory import ProjectLoggerFactory
from logger.project_logger import ProjectLogger
from orchestration.bootstrap.run_context import RunContext
from orchestration.composition.runtime.crawler_execution import (
    CrawlerExecutionOverrides,
    CrawlerExecutionServices,
    build_crawler_execution,
)
from orchestration.composition.runtime.crawler_governance import (
    build_crawler_governance,
)
from orchestration.composition.runtime.crawler_infrastructure import (
    build_crawler_infrastructure,
)
from orchestration.composition.runtime.crawler_state import (
    build_crawler_state,
)
from orchestration.resource_shutdown import ResourceShutdownManager
from shared.runtime_primitives import Clock, IdGenerator

if TYPE_CHECKING:
    from config.collection.processors import PageProcessorSettings
    from crawler.governance.processing_activity import (
        ProcessingActivityRegistry,
    )
    from crawler.runtime.crawler import Crawler
    from crawler.scheduling.seed_plan import CrawlSeedPlan
    from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
    from datachecker.manifests.crawl_state_manifest_writer import (
        CrawlStateManifestWriter,
    )


def _build_run_supervisor(
    *,
    settings: Settings,
    execution: CrawlerExecutionServices,
    logger: ProjectLogger,
) -> CrawlRunSupervisor:
    return CrawlRunSupervisor(
        drain_delayed_backlog_before_finish=(
            settings.crawler.drain_delayed_backlog_before_finish
        ),
        max_idle_delay_wait_seconds=(
            settings.crawler.max_idle_delay_wait_seconds
        ),
        drain_stall_timeout_seconds=(
            settings.crawler.drain_stall_timeout_seconds
        ),
        drain_watch_interval_seconds=(
            settings.crawler.drain_watch_interval_seconds
        ),
        scheduler=execution.scheduler,
        worker_pool=execution.worker_pool,
        worker_scaler=execution.worker_scaler,
        logger=logger,
        seed_enqueuer=execution.seed_enqueuer,
        min_workers=settings.collection.autoscaler.min_workers,
    )


@dataclass(frozen=True, slots=True)
class CrawlSeedGraph:
    """Complete crawler runtime graph."""

    crawler: "Crawler"
    dataset_writer: "DatasetWriter"
    seed_plan: "CrawlSeedPlan"


def build_crawler_graph(
    *,
    settings: Settings,
    processing_activity_registry: ProcessingActivityRegistry,
    logger_factory: ProjectLoggerFactory,
    run_context: RunContext,
    shutdown_manager: ResourceShutdownManager,
    clock: Clock,
    id_generator: IdGenerator,
    crawl_attempt_id: str | None = None,
    crawl_state_manifest_writer: CrawlStateManifestWriter | None = None,
    page_settings_override: "PageProcessorSettings | None" = None,
) -> CrawlSeedGraph:
    """Build the complete crawler graph from all subgraphs."""
    from crawler.runtime.crawler import Crawler
    from crawler.scheduling.seed_plan import CrawlSeedPlanBuilder

    project_root = Path(settings.paths.root)

    # Build the canonical HostNormalizer once; shared by UrlNormalizer,
    # governance, scheduling, and infrastructure.
    host_normalizer = HostNormalizer()

    # Build the canonical URL normalizer once; scheme *validation* stays a
    # governance capability (UrlSchemeRules), normalization is an
    # infrastructure capability (UrlNormalizer).
    url_normalizer = UrlNormalizer(
        settings=settings.collection.url_normalizer,
        logger=logger_factory.get_logger_for(UrlNormalizer),
        host_normalizer=host_normalizer,
    )

    # Build infrastructure subgraph
    infrastructure = build_crawler_infrastructure(
        settings=settings,
        logger_factory=logger_factory,
        shutdown_manager=shutdown_manager,
        clock=clock,
        url_normalizer=url_normalizer,
        host_normalizer=host_normalizer,
    )

    # Build governance subgraph (needs infrastructure)
    governance = build_crawler_governance(
        settings=settings,
        infrastructure=infrastructure,
        logger_factory=logger_factory,
        shutdown_manager=shutdown_manager,
    )

    # Build control directory (needed by state)
    control_directory = CrawlerControlDirectory(
        settings=settings.crawler,
        project_root=project_root,
    )

    # Build seed plan (domain logic)
    seed_plan_builder = CrawlSeedPlanBuilder(
        seed_entries=settings.sources.active,
        seed_source_type=settings.crawler.seed_source_type,
        feed_alternates_by_primary=settings.sources.active.feed_alternate_urls,
        url_normalizer=url_normalizer,
        host_normalizer=infrastructure.host_normalizer,
        source_scope_registry=governance.source_scope_registry,
        id_generator=id_generator,
    )
    seed_plan = seed_plan_builder.build()

    # Build state persistence primitives. The runtime reader/writer pair
    # is built in execution once scheduler and worker pool exist.
    state = build_crawler_state(
        state_settings=settings.crawler.state,
        control_directory=control_directory,
        clock=clock,
        logger_factory=logger_factory,
        crawl_session_id=run_context.crawl_session_id,
    )

    # Build execution subgraph (needs all above)
    overrides = CrawlerExecutionOverrides(
        crawl_attempt_id=crawl_attempt_id,
        crawl_state_manifest_writer=crawl_state_manifest_writer,
        processing_activity_registry=processing_activity_registry,
        page_settings=page_settings_override,
    )
    execution = build_crawler_execution(
        settings=settings,
        infrastructure=infrastructure,
        governance=governance,
        state=state,
        logger_factory=logger_factory,
        shutdown_manager=shutdown_manager,
        run_context=run_context,
        control_directory=control_directory,
        id_generator=id_generator,
        seed_tasks=seed_plan.tasks,
        overrides=overrides,
    )

    # Assemble final crawler
    crawler_logger = logger_factory.get_logger_for(Crawler)

    crawler = Crawler(
        enabled=settings.crawler.enabled,
        worker_pool=execution.worker_pool,
        logger=crawler_logger,
        control_directory=execution.control_directory,
        task_feedback=execution.task_feedback,
        run_supervisor=_build_run_supervisor(
            settings=settings,
            execution=execution,
            logger=crawler_logger,
        ),
        build_runtime_session=execution.build_runtime_session,
    )

    return CrawlSeedGraph(
        crawler=crawler,
        dataset_writer=execution.dataset_writer,
        seed_plan=seed_plan,
    )
