"""Governance builders for the crawler runtime.

Public builders own governance-specific composition: blacklist storage,
source scopes, rate limiting, URL admission, and robots enforcement.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from config.collection.caching import (
    CacheNamespaceSettings,
    CollectionCacheSettings,
)
from config.collection.fetching import FetcherSettings
from config.collection.governance import RobotsSettings, UrlFilterSettings
from config.collection.http_rules import TimeoutRulesSettings
from config.collection.identity import IdentitySettings
from config.collection.pacing import PacingSettings
from config.path_resolution.project_paths import ProjectPaths
from config.settings.root import Settings
from config.source_catalog.catalog_settings import SourceScopeSettings
from crawler.extraction.hosts_extractor import HostExtractor
from crawler.fetching.network.robots.fetcher import AiohttpRobotsFetcher
from crawler.governance.blacklist.storage.blacklist_connection import (
    BlacklistConnection,
)
from crawler.governance.blacklist.storage.blacklist_repository import (
    BlacklistRepository,
)
from crawler.governance.blacklist.storage.blacklist_table_schema import (
    BlacklistTableSchema,
)
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.network_access.network_address_guard import (
    NetworkAddressGuard,
)
from crawler.governance.rate_limit.rate_limit_rules import RateLimitRules
from crawler.governance.rate_limit.rate_limit_slot_scheduler import (
    RateLimitSlotScheduler,
)
from crawler.governance.rate_limit.rate_limit_state_registry import (
    RateLimitStateRegistry,
)
from crawler.governance.rate_limit.rate_limiter import RateLimiter
from crawler.governance.redirect.redirect_rules_validator import (
    RedirectRulesValidator,
)
from crawler.governance.robots.robots_checker import RobotsChecker
from crawler.governance.robots.robots_decision_evaluator import (
    RobotsDecisionEvaluator,
)
from crawler.governance.robots.robots_error_classifier import (
    RobotsErrorClassifier,
)
from crawler.governance.robots.robots_error_resolver import (
    RobotsErrorResolver,
    RobotsErrorResolverDependencies,
    RobotsErrorResolverRules,
)
from crawler.governance.robots.robots_fallback_rules import RobotsFallbackRules
from crawler.governance.robots.robots_host_rules_store import (
    RobotsHostRulesStore,
)
from crawler.governance.robots.robots_parser_cache import RobotsParserCache
from crawler.governance.robots.robots_parser_loader import RobotsParserLoader
from crawler.governance.robots.robots_unknown_result_suppressor import (
    RobotsUnknownResultSuppressor,
)
from crawler.governance.robots.robots_url_resolver import RobotsUrlResolver
from crawler.governance.source_scope.source_scope_registry import (
    SourceScope,
    SourceScopeRegistry,
)
from crawler.governance.url_filter.embedded_asset_rules import (
    EmbeddedAssetRules,
)
from crawler.governance.url_filter.host_denylist_rules import (
    HostDenylistRules,
)
from crawler.governance.url_filter.ip_literal_rules import IpLiteralRules
from crawler.governance.url_filter.url_admission_filter import (
    UrlAdmissionFilter,
)
from crawler.governance.url_filter.url_scheme_rules import UrlSchemeRules
from crawler.governance.url_filter.url_syntax_rules import UrlSyntaxRules
from crawler.runtime.metrics.collection_metrics import CollectionMetrics
from crawler.runtime.runtime_dependencies import (
    HttpClientSessionProvider,
)
from logger.factory import ProjectLoggerFactory
from shared.runtime_primitives import Clock

_ROBOTS_PATH: Final[str] = "/robots.txt"
_UNKNOWN_RESULT_PRUNE_INTERVAL: Final[int] = 100


def build_blacklist_repository(
    *,
    blacklist_settings: Any,
    project_root: Path,
    logger_factory: ProjectLoggerFactory,
) -> BlacklistRepository:
    """Build and initialize the persistent URL blacklist."""

    logger = logger_factory.get_logger_for(BlacklistRepository)
    database_path = ProjectPaths(project_root=project_root).resolve(
        blacklist_settings.blacklist_database_path,
    )
    resolved_settings = blacklist_settings.model_copy(
        update={
            "blacklist_database_path": database_path.as_posix(),
        },
    )
    database_handler = BlacklistConnection(settings=resolved_settings)
    if resolved_settings.blacklist_auto_initialize:
        BlacklistTableSchema(
            database_handler=database_handler,
            database_path=database_path,
            table_name=resolved_settings.blacklist_table_name,
        ).initialize()
        logger.info("blacklist_initialized", path=str(database_path))

    return BlacklistRepository(
        database_handler=database_handler,
        table_name=resolved_settings.blacklist_table_name,
    )


def build_source_scope_registry(
    *,
    source_scopes: Iterable[SourceScopeSettings],
    host_normalizer: HostNormalizer,
) -> SourceScopeRegistry:
    """Build canonical source scopes from validated source settings.

    Composition is the config→domain boundary here: external host
    representations are canonicalized through ``HostNormalizer.require()``,
    while scope-level and collection-level invariants stay owned by the
    domain objects.
    """

    return SourceScopeRegistry(
        SourceScope(
            source_name=configured_scope.source_name,
            page_hosts=frozenset(
                host_normalizer.require(host)
                for host in configured_scope.page_hosts
            ),
            asset_hosts=frozenset(
                host_normalizer.require(host)
                for host in configured_scope.asset_hosts
            ),
            redirect_hosts=frozenset(
                host_normalizer.require(host)
                for host in configured_scope.redirect_hosts
            ),
            allow_subdomains=configured_scope.allow_subdomains,
        )
        for configured_scope in source_scopes
    )


def build_rate_limiter(
    *,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    host_normalizer: HostNormalizer,
) -> RateLimiter:
    """Build the crawl rate limiter and its host-state rules graph."""

    collection = settings.collection
    pacing = collection.pacing
    logger = logger_factory.get_logger_for(RateLimiter)
    coerce = RateLimitRules.coerce_non_negative_finite_float_or_default

    minimum_requests_per_second = coerce(
        pacing.min_rps,
        default=0.0,
    )
    maximum_requests_per_second = coerce(
        pacing.max_rps,
        default=minimum_requests_per_second,
    )

    default_crawl_delay_seconds = None
    default_adaptive_requests_per_second = pacing.default_rps or 0.33
    default_effective_requests_per_second = (
        RateLimitRules.effective_requests_per_second_from_values(
            adaptive_requests_per_second=(
                default_adaptive_requests_per_second
            ),
            crawl_delay_seconds=0.0,
        )
    )

    state_registry = RateLimitStateRegistry(
        host_normalizer=host_normalizer,
        default_adaptive_requests_per_second=(
            default_adaptive_requests_per_second
        ),
        default_effective_requests_per_second=(
            default_effective_requests_per_second
        ),
    )
    slot_scheduler = RateLimitSlotScheduler(
        burst_size=pacing.burst,
        logger=logger,
        reservation_log_threshold_seconds=1.0,
        random_delay_min_seconds=pacing.jitter_min_seconds,
        random_delay_max_seconds=pacing.jitter_max_seconds,
    )
    adaptive_rules = RateLimitRules(
        logger=logger,
        min_requests_per_second_value=minimum_requests_per_second,
        max_requests_per_second_value=maximum_requests_per_second,
        backoff_factor=coerce(pacing.backoff_factor, default=0.5),
        ramp_up_factor=coerce(pacing.ramp_up_factor, default=1.05),
        error_cooldown_seconds=coerce(
            pacing.error_cooldown_seconds,
            default=30.0,
        ),
        feedback_status_codes=frozenset(
            collection.http_rules.statuses.rate_limiter_feedback
        ),
        default_crawl_delay_seconds=default_crawl_delay_seconds,
    )

    return RateLimiter(
        state_registry=state_registry,
        slot_scheduler=slot_scheduler,
        adaptive_rules=adaptive_rules,
        honor_retry_after=pacing.honor_retry_after,
        max_retry_after_seconds=pacing.max_retry_after_seconds,
        logger=logger,
        default_effective_requests_per_second=(
            default_effective_requests_per_second
        ),
    )


def build_url_filter(
    *,
    url_filter_settings: UrlFilterSettings,
    seed_hosts: tuple[str, ...],
    logger_factory: ProjectLoggerFactory,
    url_validator: UrlSchemeRules,
    host_extractor: HostExtractor,
    host_normalizer: HostNormalizer,
    source_scope_registry: SourceScopeRegistry,
) -> UrlAdmissionFilter:
    """Build the executable URL-admission rules graph."""

    logger = logger_factory.get_logger_for(UrlAdmissionFilter)

    syntax_validator = UrlSyntaxRules(
        max_page_number=(url_filter_settings.max_pagination_page_number),
        pagination_query_keys=(url_filter_settings.pagination_query_keys),
        blocked_path_fragments=(url_filter_settings.blocked_path_fragments),
        blocked_query_keys=url_filter_settings.blocked_query_keys,
        blocked_query_value_patterns=(
            url_filter_settings.blocked_query_value_patterns
        ),
        tracking_query_tokens=(url_filter_settings.tracking_query_tokens),
        low_value_image_path_fragments=(
            url_filter_settings.low_value_image_path_fragments
        ),
        low_value_image_filenames=(
            url_filter_settings.low_value_image_filenames
        ),
        social_icon_tokens=url_filter_settings.social_icon_tokens,
        logger=logger,
    )

    return UrlAdmissionFilter(
        url_scheme_validator=url_validator,
        syntax_validator=syntax_validator,
        host_denylist_validator=HostDenylistRules(
            blocked_hosts=set(url_filter_settings.blocked_hosts),
            host_normalizer=host_normalizer,
        ),
        ip_literal_validator=IpLiteralRules(
            blocked_ip_literals=set(
                url_filter_settings.blocked_ip_literals,
            ),
        ),
        embedded_asset_validator=EmbeddedAssetRules(
            settings=url_filter_settings,
            host_normalizer=host_normalizer,
            source_scope_registry=source_scope_registry,
            logger=logger_factory.get_logger_for(
                EmbeddedAssetRules,
            ),
        ),
        host_extractor=host_extractor,
        host_normalizer=host_normalizer,
        logger=logger,
    )


def build_robots_checker(
    *,
    robots_settings: RobotsSettings,
    identity_settings: IdentitySettings,
    pacing_settings: PacingSettings,
    fetcher_settings: FetcherSettings,
    cache_settings: CollectionCacheSettings,
    timeout_rules_settings: TimeoutRulesSettings,
    logger_factory: ProjectLoggerFactory,
    blacklist_repository: BlacklistRepository,
    session_provider: HttpClientSessionProvider,
    rate_limiter: RateLimiter,
    redirector: RedirectRulesValidator,
    host_normalizer: HostNormalizer,
    network_address_guard: NetworkAddressGuard,
    metrics: CollectionMetrics,
    clock: Clock,
) -> RobotsChecker:
    """Build the executable robots fetch, parse, cache, and decision graph."""

    parser_cache_ttl = _cache_ttl_seconds(
        settings=cache_settings.robots_parser,
        namespace="robots_parser",
    )
    error_cache_ttl = _cache_ttl_seconds(
        settings=cache_settings.robots_error,
        namespace="robots_error",
    )

    parser_loader = RobotsParserLoader(
        user_agent=identity_settings.user_agent,
        accept_language_header=(fetcher_settings.accept_language_header),
        accept_compressed=fetcher_settings.accept_compressed,
        fetcher=_build_robots_fetcher(
            session_provider=session_provider,
            rate_limiter=rate_limiter,
            redirector=redirector,
            host_normalizer=host_normalizer,
            network_address_guard=network_address_guard,
            clock=clock,
            logger_factory=logger_factory,
        ),
        logger=logger_factory.get_logger_for(RobotsParserLoader),
    )

    parser_cache = RobotsParserCache(
        cache_ttl_s=parser_cache_ttl,
        error_cache_ttl_s=error_cache_ttl,
        stale_ttl_s=(cache_settings.robots_parser.stale_ttl_seconds),
        parser_loader=parser_loader,
        host_normalizer=host_normalizer,
        logger=logger_factory.get_logger_for(RobotsParserCache),
    )

    error_rules = RobotsErrorResolver(
        rules=RobotsErrorResolverRules(),
        dependencies=RobotsErrorResolverDependencies(
            classifier=RobotsErrorClassifier(),
            fallback_rules=RobotsFallbackRules(
                http_403_allow_host_suffixes=(),
                host_normalizer=host_normalizer,
            ),
        ),
    )

    return RobotsChecker(
        settings=robots_settings,
        timeout_rules=timeout_rules_settings,
        decision_evaluator=RobotsDecisionEvaluator(
            respect_crawl_delay=True,
            max_crawl_delay_s=(pacing_settings.max_crawl_delay_seconds),
            user_agent=identity_settings.name,
            logger=logger_factory.get_logger_for(
                RobotsDecisionEvaluator,
            ),
        ),
        error_rules=error_rules,
        robots_url_resolver=RobotsUrlResolver(
            robots_path=_ROBOTS_PATH,
            host_normalizer=host_normalizer,
            logger=logger_factory.get_logger_for(
                RobotsUrlResolver,
            ),
        ),
        parser_cache=parser_cache,
        host_rules_store=RobotsHostRulesStore(
            host_normalizer=host_normalizer,
            logger=logger_factory.get_logger_for(
                RobotsHostRulesStore,
            ),
        ),
        user_agent=identity_settings.name,
        blacklist_repository=blacklist_repository,
        metrics=metrics,
        host_normalizer=host_normalizer,
        duplicate_result_tracker=RobotsUnknownResultSuppressor(
            ttl_seconds=float(error_cache_ttl),
            prune_every=_UNKNOWN_RESULT_PRUNE_INTERVAL,
            max_entries=cache_settings.robots_error.max_entries,
            host_normalizer=host_normalizer,
        ),
        logger=logger_factory.get_logger_for(RobotsChecker),
    )


def _build_robots_fetcher(
    *,
    session_provider: HttpClientSessionProvider,
    rate_limiter: RateLimiter,
    redirector: RedirectRulesValidator,
    host_normalizer: HostNormalizer,
    network_address_guard: NetworkAddressGuard,
    clock: Clock,
    logger_factory: ProjectLoggerFactory,
) -> AiohttpRobotsFetcher:
    """Build the canonical shared-session robots transport."""

    return AiohttpRobotsFetcher(
        session_provider=session_provider,
        rate_limiter=rate_limiter,
        redirector=redirector,
        host_normalizer=host_normalizer,
        network_address_guard=network_address_guard,
        clock=clock,
        logger=logger_factory.get_logger_for(
            AiohttpRobotsFetcher,
        ),
    )


def _cache_ttl_seconds(
    *,
    settings: CacheNamespaceSettings,
    namespace: str,
) -> int:
    """Return the effective TTL for one robots cache namespace.

    Config validation ensures ttl_seconds is configured when enabled.
    """
    if not settings.enabled:
        return 0

    return int(settings.ttl_seconds)
