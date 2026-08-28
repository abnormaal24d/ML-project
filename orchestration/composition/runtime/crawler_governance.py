"""Crawler governance subgraph composition.

Constructs governance services: URL validation, filtering, robots, redirects,
source scope, blacklist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from config.settings.root import Settings
from crawler.governance.redirect.redirect_rules_validator import (
    RedirectRulesValidator,
)
from crawler.governance.robots.robots_request_gate import RobotsRequestGate
from crawler.governance.url_filter.url_scheme_rules import UrlSchemeRules
from logger.factory import ProjectLoggerFactory
from orchestration.resource_shutdown import ResourceShutdownManager
from orchestration.composition.runtime.crawler_infrastructure import (
    CrawlerInfrastructure,
)
from orchestration.composition.runtime.governance import (
    build_blacklist_repository,
    build_robots_checker,
    build_source_scope_registry,
    build_url_filter,
)

if TYPE_CHECKING:
    from crawler.governance.blacklist_repository import BlacklistRepository
    from crawler.governance.source_scope_registry import SourceScopeRegistry
    from crawler.governance.url_filter.url_filter import UrlFilter


@dataclass(frozen=True, slots=True)
class CrawlerGovernance:
    """Governance services for the crawler."""

    url_validator: UrlSchemeRules
    url_filter: "UrlFilter"
    robots_request_gate: RobotsRequestGate
    redirector: RedirectRulesValidator
    source_scope_registry: "SourceScopeRegistry"
    blacklist_repository: "BlacklistRepository"


def build_crawler_governance(
    *,
    settings: Settings,
    infrastructure: CrawlerInfrastructure,
    logger_factory: ProjectLoggerFactory,
    shutdown_manager: ResourceShutdownManager,
) -> CrawlerGovernance:
    """Build crawler governance services."""
    project_root = Path(settings.paths.root)
    collection = settings.collection

    # URL scheme validation
    url_scheme_validation_settings = collection.url_scheme_validator
    url_validator = UrlSchemeRules(
        settings=url_scheme_validation_settings,
        logger=logger_factory.get_logger_for(UrlSchemeRules),
    )

    # Blacklist
    blacklist_repository = build_blacklist_repository(
        blacklist_settings=collection.blacklist_manager,
        project_root=project_root,
        logger_factory=logger_factory,
    )

    # Source scope registry
    source_scope_registry = build_source_scope_registry(
        source_scopes=settings.sources.active.source_scopes,
        host_normalizer=infrastructure.host_normalizer,
    )

    # Redirector
    redirector = RedirectRulesValidator(
        settings=collection.http_rules.redirects,
        host_extractor=infrastructure.host_extractor,
        url_validator=url_validator,
        blacklist_repository=blacklist_repository,
        host_normalizer=infrastructure.host_normalizer,
        logger=logger_factory.get_logger_for(RedirectRulesValidator),
        network_access_guard=infrastructure.network_access_guard,
        metrics=infrastructure.metrics,
        source_scope_registry=source_scope_registry,
    )

    # URL filter
    url_filter = build_url_filter(
        url_filter_settings=collection.url_filter,
        seed_hosts=settings.sources.active.seed_hosts,
        logger_factory=logger_factory,
        url_validator=url_validator,
        host_extractor=infrastructure.host_extractor,
        host_normalizer=infrastructure.host_normalizer,
        source_scope_registry=source_scope_registry,
    )

    # Robots checker
    robots_checker = build_robots_checker(
        robots_settings=collection.robots,
        identity_settings=collection.identity,
        pacing_settings=collection.pacing,
        fetcher_settings=collection.fetcher,
        cache_settings=collection.cache,
        timeout_rules_settings=collection.http_rules.timeouts,
        logger_factory=logger_factory,
        blacklist_repository=blacklist_repository,
        session_provider=infrastructure.session_manager,
        rate_limiter=infrastructure.rate_limiter,
        redirector=redirector,
        host_normalizer=infrastructure.host_normalizer,
        network_address_guard=infrastructure.network_access_guard,
        metrics=infrastructure.metrics,
        clock=infrastructure.clock,
    )
    shutdown_manager.add_step(
        name="robots_checker",
        close=robots_checker.aclose,
    )

    # Robots request gate
    robots_request_gate = RobotsRequestGate(
        checker=robots_checker,
        settings=collection.robots,
        rate_limiter=infrastructure.rate_limiter,
        host_normalizer=infrastructure.host_normalizer,
        metrics=infrastructure.metrics,
        logger=logger_factory.get_logger_for(RobotsRequestGate),
    )

    return CrawlerGovernance(
        url_validator=url_validator,
        url_filter=url_filter,
        robots_request_gate=robots_request_gate,
        redirector=redirector,
        source_scope_registry=source_scope_registry,
        blacklist_repository=blacklist_repository,
    )