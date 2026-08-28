"""Scheduler builder for the crawler runtime composition."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from config.collection.discovery import SchedulingSettings
from config.settings.root import Settings
from crawler.scheduling.host_control.discovery_signal_scorer import (
    DiscoverySignalScorer,
)
from crawler.scheduling.host_control.host_budget_tracker import (
    HostBudgetTracker,
)
from crawler.scheduling.host_control.host_feedback_aggregator import (
    HostFeedbackAggregator,
)
from crawler.scheduling.host_control.host_media_byte_budget import (
    HostMediaByteBudget,
)
from crawler.scheduling.priority.crawl_task_priority_calculator import (
    CrawlTaskPriorityCalculator,
)
from logger.factory import ProjectLoggerFactory

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.governance.blacklist.storage.blacklist_repository import (
        BlacklistRepository,
    )
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.host_suppression import HostSuppressionStore
    from crawler.governance.rate_limit.rate_limiter import RateLimiter
    from crawler.governance.source_scope.source_scope_registry import (
        SourceScopeRegistry,
    )
    from crawler.governance.url_filter.url_admission_filter import (
        UrlAdmissionFilter,
    )
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics
    from crawler.scheduling.completion.dead_letter_writer import (
        DeadLetterWriter,
    )
    from crawler.scheduling.url_scheduler import UrlScheduler
    from shared.runtime_primitives import IdGenerator


def build_host_scheduling_controls(
    *,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    host_extractor: HostExtractor,
    host_normalizer: HostNormalizer,
) -> tuple[
    HostBudgetTracker,
    HostMediaByteBudget,
    CrawlTaskPriorityCalculator,
]:
    """Build scheduler-owned host budgets, feedback, and priority policy."""

    scheduling = settings.collection.scheduling
    feedback_settings = scheduling.discovery_feedback
    signal_scorer = DiscoverySignalScorer(
        settings=feedback_settings,
    )
    host_budget_tracker = HostBudgetTracker(
        settings=scheduling,
        url_filter_settings=settings.collection.url_filter,
        logger=logger_factory.get_logger_for(HostBudgetTracker),
        signal_scorer=signal_scorer,
        feedback_aggregator=HostFeedbackAggregator(
            max_hosts=scheduling.host_feedback_max_hosts,
            default_info_gain=feedback_settings.default_info_gain,
            default_host_quality=feedback_settings.default_host_quality,
            host_extractor=host_extractor,
            host_normalizer=host_normalizer,
        ),
        host_extractor=host_extractor,
        host_normalizer=host_normalizer,
        seed_urls=settings.sources.active.seed_urls,
    )
    host_media_byte_budget = HostMediaByteBudget(
        host_normalizer=host_normalizer,
        max_bytes_per_host=scheduling.max_media_bytes_per_host,
        max_bytes_per_host_by_kind=dict(
            scheduling.max_media_bytes_per_host_by_kind
        ),
    )
    priority_resolver = CrawlTaskPriorityCalculator(
        config=settings.collection.url_priority_calculator,
        logger=logger_factory.get_logger_for(CrawlTaskPriorityCalculator),
        host_extractor=host_extractor,
        host_budget_tracker=host_budget_tracker,
    )
    return host_budget_tracker, host_media_byte_budget, priority_resolver


def build_scheduler(
    *,
    scheduling_settings: SchedulingSettings,
    url_normalizer: UrlNormalizer,
    url_filter: UrlAdmissionFilter,
    host_extractor: HostExtractor,
    host_normalizer: HostNormalizer,
    priority_resolver: CrawlTaskPriorityCalculator,
    blacklist_repository: BlacklistRepository,
    metrics: CollectionMetrics,
    host_budget_tracker: HostBudgetTracker,
    source_scope_registry: SourceScopeRegistry,
    host_media_byte_budget: HostMediaByteBudget | None = None,
    rate_limiter: RateLimiter,
    id_generator: IdGenerator,
    logger_factory: ProjectLoggerFactory,
    dead_letter_writer: DeadLetterWriter | None = None,
    host_suppression_reader: HostSuppressionStore | None = None,
) -> UrlScheduler:
    """Build the URL scheduler."""

    from crawler.scheduling.admission.scheduler_frontier import (
        SchedulerFrontier,
    )
    from crawler.scheduling.admission.scheduler_task_admitter import (
        SchedulerTaskAdmitter,
    )
    from crawler.scheduling.checkpointing.scheduler_state_exporter import (
        SchedulerStateExporter,
    )
    from crawler.scheduling.checkpointing.scheduler_state_restorer import (
        SchedulerStateRestorer,
    )
    from crawler.scheduling.checkpointing.scheduler_task_deserializer import (
        SchedulerTaskDeserializer,
    )
    from crawler.scheduling.checkpointing.scheduler_task_serializer import (
        SchedulerTaskSerializer,
    )
    from crawler.scheduling.completion.run_url_feedback import RunUrlFeedback
    from crawler.scheduling.completion.scheduler_retry_budget import (
        SchedulerRetryBudget,
    )
    from crawler.scheduling.completion.task_completion_handler import (
        SchedulerCompletionHandler,
    )
    from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry
    from crawler.scheduling.dispatch.host_dispatch_wait_reader import (
        HostDispatchWaitReader,
    )
    from crawler.scheduling.dispatch.scheduler_dispatcher import (
        SchedulerDispatcher,
    )
    from crawler.scheduling.host_control.host_advice_tracker import (
        HostAdviceTracker,
    )
    from crawler.scheduling.progress.active_task_registry import (
        ActiveTaskRegistry,
    )
    from crawler.scheduling.progress.scheduler_backlog_reader import (
        SchedulerBacklogReader,
    )
    from crawler.scheduling.progress.scheduler_progress_state import (
        SchedulerProgressState,
    )
    from crawler.scheduling.progress.scheduler_runtime_state import (
        SchedulerRuntimeState,
    )
    from crawler.scheduling.progress.scheduler_snapshot_reader import (
        SchedulerSnapshotReader,
    )
    from crawler.scheduling.queueing.delayed_task_queue import DelayedTaskQueue
    from crawler.scheduling.queueing.host_eligibility_queue import (
        HostEligibilityQueue,
    )
    from crawler.scheduling.queueing.host_task_queue import HostTaskQueue
    from crawler.scheduling.url_scheduler import (
        SchedulerClosedError,
        UrlScheduler,
    )

    seen_urls = SeenUrlRegistry(
        max_seen=scheduling_settings.max_seen,
        ttl_seconds=scheduling_settings.seen_url_ttl_seconds,
    )
    run_url_feedback = RunUrlFeedback(
        normalize_url=url_normalizer.normalize,
    )

    task_serializer = SchedulerTaskSerializer()

    task_deserializer = SchedulerTaskDeserializer(
        priority_resolver=priority_resolver,
        host_extractor=host_extractor,
        url_normalizer=url_normalizer,
        id_generator=id_generator,
    )

    progress_state = SchedulerProgressState()

    host_advice_tracker = HostAdviceTracker(
        max_hosts=scheduling_settings.robots_host_rules_advice_max_hosts,
        ttl_seconds=scheduling_settings.robots_host_rules_advice_ttl_seconds,
        host_normalizer=host_normalizer,
    )

    dispatch_wait_reader = HostDispatchWaitReader(
        rate_limiter=rate_limiter,
        max_inflight_per_host=scheduling_settings.max_inflight_per_host,
        inflight_host_wait_seconds=(
            scheduling_settings.inflight_host_wait_seconds
        ),
    )

    condition = asyncio.Condition()
    host_queue = HostTaskQueue(host_normalizer=host_normalizer)
    delayed_queue = DelayedTaskQueue(host_normalizer=host_normalizer)
    host_eligibility_queue = HostEligibilityQueue()
    active_registry = ActiveTaskRegistry(host_normalizer=host_normalizer)
    scheduler_state = SchedulerRuntimeState()

    task_admitter = SchedulerTaskAdmitter(
        settings=scheduling_settings,
        url_filter=url_filter,
        seen_urls=seen_urls,
        blacklist_repository=blacklist_repository,
        metrics=metrics,
        host_budget_tracker=host_budget_tracker,
        host_media_byte_budget=host_media_byte_budget,
        host_advice_tracker=host_advice_tracker,
        run_url_feedback=run_url_feedback,
        source_scope_registry=source_scope_registry,
    )

    state_exporter = SchedulerStateExporter(
        serializer=task_serializer,
    )

    state_restorer = SchedulerStateRestorer(
        deserializer=task_deserializer,
    )

    scheduler_logger = logger_factory.get_logger_for(UrlScheduler)
    backlog_reader = SchedulerBacklogReader(
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        max_feeds_per_host=scheduling_settings.max_feeds_per_host,
    )

    def is_scheduler_drained() -> bool:
        return (
            host_queue.queue_size == 0
            and delayed_queue.queue_size == 0
            and active_registry.total_tracked_count == 0
        )

    def canonical_host_from_url(url: str) -> str | None:
        extracted_host = host_extractor.extract(url)
        return host_normalizer.normalize(extracted_host)

    retry_rules = SchedulerRetryBudget(
        settings=scheduling_settings,
        logger=scheduler_logger,
        is_drained=is_scheduler_drained,
    )
    frontier_service = SchedulerFrontier(
        id_generator=id_generator,
        url_normalizer=url_normalizer,
        priority_resolver=priority_resolver,
        task_admitter=task_admitter,
        progress_state=progress_state,
        backlog_reader=backlog_reader,
        seen_urls=seen_urls,
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        condition=condition,
        canonical_host_from_url=canonical_host_from_url,
        is_closed=scheduler_state.is_closed,
        allocate_sequence=scheduler_state.allocate_sequence,
        logger=scheduler_logger,
        max_rejection_samples=UrlScheduler.MAX_REJECTION_SAMPLES,
    )

    def skip_reason_before_fetch(task: CrawlTask) -> str | None:
        if run_url_feedback.was_not_modified(task=task):
            return "not_modified_this_run"
        if run_url_feedback.is_forbidden_endpoint(url=task.url):
            return "forbidden_endpoint_this_run"
        return None

    dispatch_service = SchedulerDispatcher(
        dispatch_wait_reader=dispatch_wait_reader,
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        host_eligibility_queue=host_eligibility_queue,
        inflight_count_by_host=active_registry.inflight_count_by_host,
        max_inflight_per_host=scheduling_settings.max_inflight_per_host,
        condition=condition,
        active_registry=active_registry,
        abandon_suppressed_host_threshold_seconds=(
            None
            if scheduling_settings.abandon_suppressed_host_threshold_seconds
            is None
            else float(
                scheduling_settings.abandon_suppressed_host_threshold_seconds
            )
        ),
        logger=scheduler_logger,
        is_closed=scheduler_state.is_closed,
        closed_error_factory=lambda: SchedulerClosedError(
            "scheduler is closed"
        ),
        skip_reason_before_fetch=skip_reason_before_fetch,
    )

    snapshot_reader = SchedulerSnapshotReader(
        condition=condition,
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        active_registry=active_registry,
        backlog_reader=backlog_reader,
        frontier_service=frontier_service,
        task_admitter=task_admitter,
        progress_state=progress_state,
        max_inflight_per_host=scheduling_settings.max_inflight_per_host,
        host_eligibility_queue=host_eligibility_queue,
    )

    completion_handler = SchedulerCompletionHandler(
        condition=condition,
        active_registry=active_registry,
        retry_rules=retry_rules,
        seen_urls=seen_urls,
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        progress_state=progress_state,
        run_url_feedback=run_url_feedback,
        logger=scheduler_logger,
        is_closed=scheduler_state.is_closed,
        dead_letter_writer=dead_letter_writer,
        host_eligibility_queue=host_eligibility_queue,
    )

    scheduler = UrlScheduler(
        url_normalizer=url_normalizer,
        host_extractor=host_extractor,
        host_normalizer=host_normalizer,
        state=scheduler_state,
        condition=condition,
        seen_urls=seen_urls,
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        host_eligibility_queue=host_eligibility_queue,
        active_registry=active_registry,
        host_advice_tracker=host_advice_tracker,
        dispatch_wait_reader=dispatch_wait_reader,
        snapshot_reader=snapshot_reader,
        completion_handler=completion_handler,
        progress_state=progress_state,
        state_exporter=state_exporter,
        state_restorer=state_restorer,
        checkpoint_task_deserializer=task_deserializer,
        backlog_reader=backlog_reader,
        retry_rules=retry_rules,
        frontier_service=frontier_service,
        dispatch_service=dispatch_service,
        logger=scheduler_logger,
    )

    if host_suppression_reader is not None:
        scheduler.set_host_suppression_reader(host_suppression_reader)

    return scheduler
