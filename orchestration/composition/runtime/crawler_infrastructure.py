"""Crawler infrastructure subgraph composition.

Constructs core infrastructure services: metrics, HTTP transport, rate limiting,
coverage tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING

from config.settings.root import Settings
from crawler.coverage.state import CoverageState
from crawler.extraction.hosts_extractor import HostExtractor
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.fetching.response.cache import ConditionalRepresentationCache
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.host_suppression import HostSuppressionStore
from crawler.governance.rate_limit.rate_limiter import RateLimiter
from crawler.metrics.prometheus_exporter import PrometheusExporter
from crawler.runtime.metrics.collection_metrics import CollectionMetrics
from crawler.scheduling.host_control.host_budget_tracker import (
    HostBudgetTracker,
)
from crawler.scheduling.host_control.host_media_byte_budget import (
    HostMediaByteBudget,
)
from crawler.scheduling.priority.crawl_task_priority_calculator import (
    CrawlTaskPriorityCalculator,
)
from logger.factory import ProjectLoggerFactory
from orchestration.composition.runtime.fetch import (
    build_host_suppression_store,
    build_http_transport,
)
from orchestration.composition.runtime.governance import build_rate_limiter
from orchestration.composition.runtime.scheduler import (
    build_host_scheduling_controls,
)
from orchestration.resource_shutdown import ResourceShutdownManager
from shared.runtime_primitives import Clock

if TYPE_CHECKING:
    from crawler.runtime.http.network_guard import HttpNetworkAccessGuard
    from crawler.runtime.http.session import HttpSessionManager


@dataclass(frozen=True, slots=True)
class CrawlerInfrastructure:
    """Core infrastructure services for the crawler."""

    host_normalizer: HostNormalizer
    host_extractor: HostExtractor
    metrics: CollectionMetrics
    prometheus_exporter: PrometheusExporter | None
    network_access_guard: "HttpNetworkAccessGuard"
    session_manager: "HttpSessionManager"
    rate_limiter: RateLimiter
    coverage_tracker: CoverageState
    host_budget_tracker: HostBudgetTracker
    host_media_byte_budget: HostMediaByteBudget
    priority_resolver: CrawlTaskPriorityCalculator
    host_suppression_store: HostSuppressionStore
    conditional_representation_cache: ConditionalRepresentationCache
    clock: Clock
    url_normalizer: UrlNormalizer


def build_crawler_infrastructure(
    *,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    shutdown_manager: ResourceShutdownManager,
    clock: Clock,
    url_normalizer: UrlNormalizer,
    host_normalizer: HostNormalizer,
) -> CrawlerInfrastructure:
    """Build crawler infrastructure services."""

    host_extractor = HostExtractor(
        logger=logger_factory.get_logger_for(HostExtractor),
    )
    metrics = CollectionMetrics(
        enabled=settings.collection.metrics.enabled,
        logger=logger_factory.get_logger_for(CollectionMetrics),
        host_normalizer=host_normalizer,
    )

    prometheus_exporter: PrometheusExporter | None = None
    metrics_export_settings = settings.collection.metrics
    if metrics_export_settings.prometheus_enabled:
        prometheus_exporter = PrometheusExporter(
            port=metrics_export_settings.prometheus_port,
        )
        # NOTE: prometheus_exporter.start() moved to application lifecycle
        shutdown_manager.add_step(
            name="prometheus_exporter",
            close=prometheus_exporter.aclose,
        )

    network_access_guard, session_manager = build_http_transport(
        settings=settings,
        host_normalizer=host_normalizer,
        logger_factory=logger_factory,
    )
    shutdown_manager.add_step(
        name="http_session",
        close=session_manager.aclose,
    )

    rate_limiter = build_rate_limiter(
        settings=settings,
        logger_factory=logger_factory,
        host_normalizer=host_normalizer,
    )

    coverage_tracker = CoverageState.from_settings(settings.coverage)

    (
        host_budget_tracker,
        host_media_byte_budget,
        priority_resolver,
    ) = build_host_scheduling_controls(
        settings=settings,
        logger_factory=logger_factory,
        host_extractor=host_extractor,
        host_normalizer=host_normalizer,
    )

    host_suppression_store = build_host_suppression_store(
        settings=settings,
        host_normalizer=host_normalizer,
        logger_factory=logger_factory,
    )

    conditional_cache_settings = (
        settings.collection.cache.conditional_representation_cache
    )
    conditional_representation_cache = ConditionalRepresentationCache(
        enabled=conditional_cache_settings.enabled,
        max_entries=conditional_cache_settings.max_entries,
        ttl_seconds=conditional_cache_settings.ttl_seconds,
        clock=monotonic,
    )

    return CrawlerInfrastructure(
        host_normalizer=host_normalizer,
        host_extractor=host_extractor,
        metrics=metrics,
        prometheus_exporter=prometheus_exporter,
        network_access_guard=network_access_guard,
        session_manager=session_manager,
        rate_limiter=rate_limiter,
        coverage_tracker=coverage_tracker,
        host_budget_tracker=host_budget_tracker,
        host_media_byte_budget=host_media_byte_budget,
        priority_resolver=priority_resolver,
        host_suppression_store=host_suppression_store,
        conditional_representation_cache=conditional_representation_cache,
        clock=clock,
        url_normalizer=url_normalizer,
    )
