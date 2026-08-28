"""Processing subgraph for crawler execution.

Assembles the core fetch → schedule → persist → process chain as a
cohesive subgraph with explicit internal dependency flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from config.collection.processors import PageProcessorSettings
from config.settings.root import Settings
from crawler.discovery.feed_alternate_resolver import (
    FeedAlternateResolver,
    expand_seed_tasks_with_feed_alternates,
)
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.governance.processing_activity import TRAINING_DATASET_ACTIVITY_ID
from crawler.processing.task_processor import CrawlTaskProcessor
from crawler.scheduling.url_scheduler import UrlScheduler
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from orchestration.composition.runtime.crawler_execution.contracts import (
    CrawlerExecutionOverrides,
)
from orchestration.composition.runtime.dataset import (
    build_dataset_writer,
)
from orchestration.composition.runtime.fetch import build_fetcher
from orchestration.composition.runtime.handlers import (
    build_task_processor,
)
from orchestration.composition.runtime.scheduler import (
    build_scheduler,
)

if TYPE_CHECKING:
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
    from shared.runtime_primitives import Clock, IdGenerator


def build_processing_runtime(
    *,
    project_root: Path,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    infrastructure: CrawlerInfrastructure,
    governance: CrawlerGovernance,
    state: CrawlStatePersistence,
    clock: Clock,
    shutdown_manager: ResourceShutdownManager,
    run_context: RunContext,
    url_normalizer: UrlNormalizer,
    id_generator: IdGenerator,
    page_content_extractor: object,
    discovery_task_builder: object,
    overrides: CrawlerExecutionOverrides,
) -> tuple[UrlScheduler, DatasetWriter, CrawlTaskProcessor]:
    """Assemble the processing chain: fetch → schedule → persist → process.

    Returns the three core processing services that downstream workers
    and session factories depend on.
    """
    collection = settings.collection

    # Build FeedAlternateResolver directly (no longer comes from seed_plan)
    feed_alternate_resolver = FeedAlternateResolver(
        alternates_by_primary=settings.sources.active.feed_alternate_urls,
        url_normalizer=url_normalizer,
        host_normalizer=infrastructure.host_normalizer,
    )

    fetcher = build_fetcher(
        project_root=project_root,
        settings=settings,
        logger_factory=logger_factory,
        session_manager=infrastructure.session_manager,
        rate_limiter=infrastructure.rate_limiter,
        metrics=infrastructure.metrics,
        host_normalizer=infrastructure.host_normalizer,
        clock=clock,
        feed_alternate_resolver=feed_alternate_resolver,
        url_validator=governance.url_validator,
        host_extractor=infrastructure.host_extractor,
        blacklist_repository=governance.blacklist_repository,
        redirector=governance.redirector,
        robots_request_gate=governance.robots_request_gate,
        network_access_guard=infrastructure.network_access_guard,
        host_suppression_store=infrastructure.host_suppression_store,
        source_scope_registry=governance.source_scope_registry,
        conditional_representation_cache=(
            infrastructure.conditional_representation_cache
        ),
    )

    scheduler = build_scheduler(
        scheduling_settings=collection.scheduling,
        url_normalizer=url_normalizer,
        url_filter=governance.url_filter,
        host_extractor=infrastructure.host_extractor,
        host_normalizer=infrastructure.host_normalizer,
        priority_resolver=infrastructure.priority_resolver,
        blacklist_repository=governance.blacklist_repository,
        metrics=infrastructure.metrics,
        host_budget_tracker=infrastructure.host_budget_tracker,
        source_scope_registry=governance.source_scope_registry,
        host_media_byte_budget=infrastructure.host_media_byte_budget,
        rate_limiter=infrastructure.rate_limiter,
        id_generator=id_generator,
        logger_factory=logger_factory,
        dead_letter_writer=state.dead_letter_writer,
        host_suppression_reader=infrastructure.host_suppression_store,
    )

    dataset_writer = build_dataset_writer(
        settings=settings,
        logger_factory=logger_factory,
        coverage_tracker=infrastructure.coverage_tracker,
        url_normalizer=url_normalizer,
        shutdown_manager=shutdown_manager,
        clock=clock,
        id_generator=id_generator,
        host_normalizer=infrastructure.host_normalizer,
        crawl_attempt_id=overrides.crawl_attempt_id,
        crawl_state_manifest_writer=overrides.crawl_state_manifest_writer,
        run_context=run_context,
        processing_activity_registry=overrides.processing_activity_registry,
        processing_activity_id=TRAINING_DATASET_ACTIVITY_ID,
        conditional_representation_cache=(
            infrastructure.conditional_representation_cache
        ),
    )

    effective_page_settings: PageProcessorSettings = (
        overrides.page_settings
        if overrides.page_settings is not None
        else settings.collection.processors.page
    )

    task_processor = build_task_processor(
        settings=settings,
        page_settings=effective_page_settings,
        logger_factory=logger_factory,
        fetcher=fetcher,
        coverage_tracker=infrastructure.coverage_tracker,
        scheduler=scheduler,
        dataset_writer=dataset_writer,
        url_filter=governance.url_filter,
        url_normalizer=url_normalizer,
        host_normalizer=infrastructure.host_normalizer,
        page_content_extractor=page_content_extractor,
        discovery_task_builder=discovery_task_builder,
        rejected_discovery_reporter=None,
        network_access_guard=infrastructure.network_access_guard,
        redirector=governance.redirector,
        id_generator=id_generator,
    )

    return scheduler, dataset_writer, task_processor
