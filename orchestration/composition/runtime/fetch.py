"""Fetch composition for the crawler runtime.

The public builders own HTTP transport, host-profile preferences and the
fetcher graph. The fetcher owns its response-body graph; the host-profile store
is injected so the scheduler and fetcher share one canonical instance.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from config.collection.content_settings import ContentProcessorSettings
from config.path_resolution.project_paths import ProjectPaths
from config.settings.classification import ClassificationSettings
from config.settings.root import Settings
from crawler.classification.content_category_detector import (
    ContentCategoryDetector,
)
from crawler.classification.content_classifier import (
    ContentClassifier,
    ContentClassifierConfig,
)
from crawler.classification.content_kind_resolver import ContentKindResolver
from crawler.classification.content_relevance import ContentRelevanceScorer
from crawler.classification.encoding_detector import EncodingDetector
from crawler.classification.language_detector import LanguageDetector
from crawler.classification.media_kind import MediaKind
from crawler.classification.media_kind_registry import definition_for
from crawler.classification.mime_signature_detector import (
    MimeSignatureDetector,
)
from crawler.classification.mime_type_resolver import MimeTypeResolver
from crawler.fetching.acceptance.body_strategy import MediaBodyReadStrategy
from crawler.fetching.acceptance.resolver import FetchAcceptanceResolver
from crawler.fetching.execution.attempt import FetchAttemptExecutor
from crawler.fetching.feedback.attempt_recorder import (
    FetchAttemptOutcomeRecorder,
)
from crawler.fetching.feedback.transport_recorder import (
    TransportFeedbackRecorder,
)
from crawler.fetching.fetcher import FetchOrchestrator
from crawler.fetching.media.metadata_builder import MediaMetadataResultBuilder
from crawler.fetching.media.strategy import MediaFetchStrategyResolver
from crawler.fetching.network.body.failure_processor import (
    ResponseBodyFailureProcessor,
)
from crawler.fetching.network.body.partial_store import PartialPayloadStorage
from crawler.fetching.network.body.reader import AiohttpResponseBodyReader
from crawler.fetching.network.body.stream_writer import PayloadStreamWriter
from crawler.fetching.network.preflight.executor import HeadPreflightExecutor
from crawler.fetching.network.preflight.response_evaluator import (
    HeadPreflightResponseEvaluator,
)
from crawler.fetching.network.request import AiohttpRequestRunner
from crawler.fetching.network.session import AiohttpClientSessionProvider
from crawler.fetching.request.body_plan_resolver import BodyReadPlanResolver
from crawler.fetching.request.context_builder import FetchRequestContextBuilder
from crawler.fetching.request.headers.builder import RequestHeaderBuilder
from crawler.fetching.response.body_reader import FetchResponseBodyReader
from crawler.fetching.response.cache import ConditionalRepresentationCache
from crawler.fetching.response.processor import (
    FetchResponseProcessor,
)
from crawler.fetching.response.status_rules import FetchResponseStatusRules
from crawler.fetching.response.validator import FetchResponseValidator
from crawler.fetching.results.materializer import FetchedPayloadMaterializer
from crawler.governance.circuit_breaker.host_circuit_breaker import (
    HostCircuitBreaker,
)
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.host_suppression import HostSuppressionStore
from crawler.governance.network_access.network_address_guard import (
    NetworkAddressGuard,
)
from crawler.governance.rate_limit.rate_limiter import RateLimiter
from crawler.governance.redirect.redirect_rules_validator import (
    RedirectRulesValidator,
)
from crawler.governance.retry.retry_manager import RetryManager
from crawler.governance.robots.robots_request_gate import RobotsRequestGate
from crawler.governance.url_filter.url_scheme_rules import UrlSchemeRules
from crawler.runtime.metrics.collection_metrics import CollectionMetrics
from crawler.runtime.runtime_dependencies import (
    HttpClientSessionProvider,
)
from logger.factory import ProjectLoggerFactory
from shared.runtime_primitives import Clock

if TYPE_CHECKING:
    from crawler.discovery.feed_alternate_resolver import (
        FeedAlternateResolver,
    )
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.governance.blacklist.storage.blacklist_repository import (
        BlacklistRepository,
    )
    from crawler.governance.source_scope.source_scope_registry import (
        SourceScopeRegistry,
    )


def build_http_transport(
    *,
    settings: Settings,
    host_normalizer: HostNormalizer,
    logger_factory: ProjectLoggerFactory,
) -> tuple[NetworkAddressGuard, AiohttpClientSessionProvider]:
    """Build the guarded shared HTTP session for crawler network traffic."""

    network_access_guard = NetworkAddressGuard(
        settings=settings.collection.http_rules.network_access,
        host_normalizer=host_normalizer,
        logger=logger_factory.get_logger_for(NetworkAddressGuard),
    )
    session_manager = AiohttpClientSessionProvider(
        timeout_rules=settings.collection.http_rules.timeouts,
        connection_pool=settings.collection.http_rules.connection_pool,
        logger=logger_factory.get_logger_for(AiohttpClientSessionProvider),
        network_access_guard=network_access_guard,
    )
    return network_access_guard, session_manager


def build_host_suppression_store(
    *,
    settings: Settings,
    host_normalizer: HostNormalizer,
    logger_factory: ProjectLoggerFactory,
) -> HostSuppressionStore:
    """Build the host suppression store."""

    host_profile_settings = settings.collection.cache.host_profile
    ttl_seconds = host_profile_settings.ttl_seconds
    # Config validation ensures host_profile is enabled and ttl_seconds is positive

    fetcher_settings = settings.collection.fetcher
    return HostSuppressionStore(
        ttl_seconds=ttl_seconds,
        max_size=host_profile_settings.max_entries,
        suppress_after_forbidden_responses=(
            fetcher_settings.host_profile_forbidden_host_threshold
        ),
        forbidden_host_cooldown_seconds=(
            fetcher_settings.host_profile_forbidden_host_cooldown_seconds
        ),
        host_normalizer=host_normalizer,
        monotonic_seconds=monotonic,
        logger=logger_factory.get_logger_for(HostSuppressionStore),
    )


def build_fetcher(
    *,
    project_root: Path,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    session_manager: HttpClientSessionProvider,
    rate_limiter: RateLimiter,
    metrics: CollectionMetrics,
    host_normalizer: HostNormalizer,
    clock: Clock,
    feed_alternate_resolver: FeedAlternateResolver | None = None,
    url_validator: UrlSchemeRules,
    host_extractor: HostExtractor,
    blacklist_repository: BlacklistRepository,
    redirector: RedirectRulesValidator,
    robots_request_gate: RobotsRequestGate,
    network_access_guard: NetworkAddressGuard,
    host_suppression_store: HostSuppressionStore,
    source_scope_registry: SourceScopeRegistry,
    conditional_representation_cache: ConditionalRepresentationCache,
) -> FetchOrchestrator:
    """Build the fetch orchestrator and its full collaborator graph."""

    collection = settings.collection
    fetcher_settings = collection.fetcher
    http_rules = collection.http_rules
    body_settings = collection.response_body_reader
    content_classifier = _build_content_classifier(
        classification_settings=settings.classification,
        processor_settings=collection.content_processor,
        logger_factory=logger_factory,
    )

    temporary_directory = ProjectPaths(project_root=project_root).resolve(
        body_settings.temporary_directory,
    )
    temporary_directory.mkdir(parents=True, exist_ok=True)
    partial_storage = PartialPayloadStorage(
        temporary_directory=temporary_directory,
        preserve_partial_files=body_settings.preserve_partial_files,
    )
    body_materializer = FetchedPayloadMaterializer(
        logger=logger_factory.get_logger_for(FetchedPayloadMaterializer),
        partial_payload_storage=partial_storage,
        temporary_directory=temporary_directory,
        sniff_byte_count=body_settings.sniff_byte_count,
        monotonic_seconds=monotonic,
    )
    failure_processor = ResponseBodyFailureProcessor(
        payload_materializer=body_materializer,
        logger=logger_factory.get_logger_for(ResponseBodyFailureProcessor),
        monotonic_seconds=monotonic,
    )
    stream_writer = PayloadStreamWriter(
        logger=logger_factory.get_logger_for(PayloadStreamWriter),
        monotonic_seconds=monotonic,
        max_in_flight_bytes=body_settings.max_in_flight_bytes,
        bytes_per_second=body_settings.download_bytes_per_second,
    )
    raw_reader = AiohttpResponseBodyReader(
        settings=body_settings,
        document_content_types=frozenset(
            definition_for(MediaKind.DOCUMENT).mime_types,
        ),
        timeout_rules=http_rules.timeouts,
        temporary_directory=temporary_directory,
        partial_payload_storage=partial_storage,
        failure_processor=failure_processor,
        stream_writer=stream_writer,
        payload_materializer=body_materializer,
        monotonic_seconds=monotonic,
        logger=logger_factory.get_logger_for(AiohttpResponseBodyReader),
    )
    response_body_reader = FetchResponseBodyReader(
        response_body_reader=raw_reader,
        partial_payload_storage=partial_storage,
        logger=logger_factory.get_logger_for(FetchResponseBodyReader),
    )

    feedback = TransportFeedbackRecorder(
        settings=fetcher_settings,
        rate_limiter=rate_limiter,
        metrics=metrics,
        logger=logger_factory.get_logger_for(TransportFeedbackRecorder),
    )
    attempt_feedback = FetchAttemptOutcomeRecorder(
        feedback_recorder=feedback,
        host_suppression_store=host_suppression_store,
        status_rules=http_rules.statuses,
        logger=logger_factory.get_logger_for(FetchAttemptOutcomeRecorder),
    )

    header_builder = RequestHeaderBuilder(
        settings=fetcher_settings,
        identity=collection.identity,
        logger=logger_factory.get_logger_for(RequestHeaderBuilder),
    )
    context_builder = FetchRequestContextBuilder(
        url_validator=url_validator,
        host_extractor=host_extractor,
        host_normalizer=host_normalizer,
        acceptance_resolver=FetchAcceptanceResolver(
            modality_acceptance=collection.modality_acceptance,
        ),
        blacklist_repository=blacklist_repository,
        network_access_guard=network_access_guard,
        metrics=metrics,
        logger=logger_factory.get_logger_for(FetchRequestContextBuilder),
    )
    request_runner = AiohttpRequestRunner(
        redirector=redirector,
        host_normalizer=host_normalizer,
        rate_limiter=rate_limiter,
        robots_gate=robots_request_gate,
        network_address_guard=network_access_guard,
    )

    validator = FetchResponseValidator(
        redirector=redirector,
        metrics=metrics,
        logger=logger_factory.get_logger_for(FetchResponseValidator),
    )
    status_rules = FetchResponseStatusRules(
        settings=fetcher_settings,
        status_rules=http_rules.statuses,
        logger=logger_factory.get_logger_for(FetchResponseStatusRules),
    )
    body_plan_resolver = BodyReadPlanResolver(
        settings=fetcher_settings,
        logger=logger_factory.get_logger_for(BodyReadPlanResolver),
        partial_payload_storage=(
            partial_storage if body_settings.preserve_partial_files else None
        ),
    )
    media_strategy_resolver = MediaFetchStrategyResolver()
    response_processor = FetchResponseProcessor(
        fetch_response_body_reader=response_body_reader,
        response_validator=validator,
        response_status_rules=status_rules,
        content_classifier=content_classifier,
        host_suppression_store=host_suppression_store,
        media_body_read_strategy=MediaBodyReadStrategy(
            logger=logger_factory.get_logger_for(MediaBodyReadStrategy),
        ),
        conditional_representation_cache=conditional_representation_cache,
        media_semaphore=asyncio.Semaphore(
            http_rules.connection_pool.media_max_connections,
        ),
        logger=logger_factory.get_logger_for(FetchResponseProcessor),
        now_utc=clock.now,
    )

    retry_manager = RetryManager(
        settings=http_rules.retries,
        status_rules=http_rules.statuses,
        logger=logger_factory.get_logger_for(RetryManager),
        random_generator=random.Random(),  # nosec B311
        total_budget_seconds=float(
            collection.worker_pool.processing_timeout_seconds,
        ),
        minimum_attempt_seconds=float(
            http_rules.timeouts.connect_timeout_seconds,
        ),
    )
    circuit_breaker = HostCircuitBreaker(
        failure_threshold=http_rules.circuit_breaker.failure_threshold,
        cooldown_seconds=http_rules.circuit_breaker.cooldown_seconds,
        monotonic_seconds=monotonic,
        host_normalizer=host_normalizer,
    )
    attempt_executor = FetchAttemptExecutor(
        status_rules=http_rules.statuses,
        rate_limiter=rate_limiter,
        now_utc=clock.now,
        attempt_feedback=attempt_feedback,
        response_processor=response_processor,
        conditional_representation_cache=conditional_representation_cache,
        timeout_rules=http_rules.timeouts,
        large_body_threshold_bytes=(
            fetcher_settings.large_media_timeout_threshold_bytes
        ),
        body_read_plan_resolver=body_plan_resolver,
        circuit_breaker=circuit_breaker,
        host_normalizer=host_normalizer,
        request_runner=request_runner,
    )
    preflight_evaluator = HeadPreflightResponseEvaluator(
        settings=fetcher_settings,
        logger=logger_factory.get_logger_for(HeadPreflightResponseEvaluator),
        metrics=metrics,
        media_strategy_resolver=media_strategy_resolver,
        status_settings=http_rules.statuses,
        retry_manager=retry_manager,
    )
    preflight_executor = HeadPreflightExecutor(
        settings=fetcher_settings,
        timeout_rules=http_rules.timeouts,
        request_runner=request_runner,
        response_evaluator=preflight_evaluator,
        logger=logger_factory.get_logger_for(HeadPreflightExecutor),
        feedback_reporter=feedback.record,
        host_normalizer=host_normalizer,
        host_allowlist=normalized_host_allowlist(
            settings=settings,
            host_normalizer=host_normalizer,
        ),
        monotonic_seconds=monotonic,
    )
    metadata_builder = MediaMetadataResultBuilder(
        payload_materializer=body_materializer,
        logger=logger_factory.get_logger_for(MediaMetadataResultBuilder),
        now_utc=clock.now,
    )

    return FetchOrchestrator(
        settings=fetcher_settings,
        session_provider=session_manager,
        head_preflight_executor=preflight_executor,
        request_header_builder=header_builder,
        request_context_builder=context_builder,
        attempt_executor=attempt_executor,
        media_strategy_resolver=media_strategy_resolver,
        media_metadata_result_builder=metadata_builder,
        retry_manager=retry_manager,
        feed_alternate_resolver=feed_alternate_resolver,
        logger=logger_factory.get_logger_for(FetchOrchestrator),
    )


def normalized_host_allowlist(
    *,
    settings: Settings,
    host_normalizer: HostNormalizer,
) -> frozenset[str]:
    return frozenset(
        normalized_host
        for host in settings.collection.fetcher.head_preflight_host_allowlist
        if (normalized_host := host_normalizer.normalize(host)) is not None
    )


def _build_content_classifier(
    *,
    classification_settings: ClassificationSettings,
    processor_settings: ContentProcessorSettings,
    logger_factory: ProjectLoggerFactory,
) -> ContentClassifier:
    """Build the content-classification collaborator graph."""

    mime_signature_detector = MimeSignatureDetector(
        settings=classification_settings.mime_signature_detector,
        logger=logger_factory.get_logger_for(MimeSignatureDetector),
    )
    mime_type_resolver = MimeTypeResolver(
        settings=classification_settings.mime_type_resolver,
        logger=logger_factory.get_logger_for(MimeTypeResolver),
        mime_signature_detector=mime_signature_detector,
    )

    return ContentClassifier(
        mime_type_resolver=mime_type_resolver,
        encoding_detector=EncodingDetector(
            settings=classification_settings.encoding_detector,
            logger=logger_factory.get_logger_for(EncodingDetector),
        ),
        language_detector=LanguageDetector(
            settings=classification_settings.language_detector,
            logger=logger_factory.get_logger_for(LanguageDetector),
        ),
        kind_resolver=ContentKindResolver(
            settings=classification_settings.kind_resolver,
        ),
        category_detector=ContentCategoryDetector(
            settings=classification_settings.content_category_detector,
            logger=logger_factory.get_logger_for(ContentCategoryDetector),
        ),
        relevance_scorer=ContentRelevanceScorer(
            settings=classification_settings.content_relevance_scorer,
        ),
        config=ContentClassifierConfig(
            default_language=(
                classification_settings.language_detector.default_language
            ),
            text_metadata_sample_bytes=(
                processor_settings.text_metadata_sample_bytes
            ),
        ),
        logger=logger_factory.get_logger_for(ContentClassifier),
    )
