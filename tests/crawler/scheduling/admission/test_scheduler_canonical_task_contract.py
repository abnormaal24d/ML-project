"""Regressions for the scheduler's canonical-task admission contract."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import fields
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from config.collection.discovery import SchedulingSettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.source_scope.source_scope_registry import (
    SourceScope,
    SourceScopeRegistry,
)
from crawler.scheduling.admission.admission_context import AdmissionContext
from crawler.scheduling.admission.admission_prerequisite_checker import (
    AdmissionPrerequisiteChecker,
)
from crawler.scheduling.admission.admission_task_identity import (
    scheduler_task_identity_key,
)
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecisionReason,
)
from crawler.scheduling.admission.scheduler_frontier import SchedulerFrontier
from crawler.scheduling.admission.scheduler_task_admitter import (
    SchedulerTaskAdmitter,
)
from crawler.scheduling.checkpointing.scheduler_task_envelope import (
    SchedulerTaskEnvelope,
)
from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry
from crawler.scheduling.host_control.host_advice_tracker import (
    HostAdviceTracker,
)
from crawler.scheduling.progress.scheduler_backlog_reader import (
    SchedulerBacklogReader,
)
from crawler.scheduling.progress.scheduler_progress_state import (
    SchedulerProgressState,
)
from crawler.scheduling.queueing.delayed_task_queue import DelayedTaskQueue
from crawler.scheduling.queueing.host_task_queue import HostTaskQueue
from crawler.scheduling.url_scheduler import UrlScheduler
from tests.support.logging import TEST_LOGGER

RAW_URL = "HTTPS://EXAMPLE.TEST/a/../canonical?Signature=AbC"
CANONICAL_URL = "https://example.test/canonical?Signature=AbC"


class _IdGenerator:
    def generate(self) -> str:
        return "generated-task"


class _FixedNormalizer:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def normalize(self, url: str) -> str:
        self.inputs.append(url)
        return CANONICAL_URL


class _TrackingFilter:
    restrict_to_seed_hosts_enabled = False

    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.tasks: list[CrawlTask] = []

    def evaluate_task(self, task: CrawlTask) -> object:
        self.tasks.append(task)
        return SimpleNamespace(allowed=self.allowed)


class _TrackingPriorityResolver:
    def __init__(self) -> None:
        self.tasks: list[CrawlTask] = []

    def __call__(self, task: CrawlTask) -> int:
        self.tasks.append(task)
        return 17


def _settings() -> SchedulingSettings:
    return SchedulingSettings(
        max_pending_per_host=12,
        max_pending_per_host_by_kind={},
        max_pending_per_host_under_pressure=None,
        max_pending_per_host_critical=None,
        max_pending_per_host_by_kind_under_pressure={},
        max_pending_per_host_by_kind_critical={},
        queue_high_watermark=1000,
        queue_critical_watermark=2000,
        dynamic_crawl_budget_enabled=False,
    )


def _build_frontier(
    *,
    url_filter: _TrackingFilter,
) -> tuple[
    SchedulerFrontier,
    asyncio.Condition,
    HostTaskQueue,
    SeenUrlRegistry,
    _TrackingPriorityResolver,
]:
    host_normalizer = HostNormalizer()
    host_queue = HostTaskQueue(host_normalizer=host_normalizer)
    delayed_queue = DelayedTaskQueue(host_normalizer=host_normalizer)
    seen_urls = SeenUrlRegistry(max_seen=1_000, ttl_seconds=None)
    condition = asyncio.Condition()
    host_advice_tracker = HostAdviceTracker(
        ttl_seconds=None,
        max_hosts=32,
        host_normalizer=host_normalizer,
    )
    admitter = SchedulerTaskAdmitter(
        settings=_settings(),
        seen_urls=seen_urls,
        host_advice_tracker=host_advice_tracker,
        url_filter=url_filter,  # type: ignore[arg-type]
        source_scope_registry=SourceScopeRegistry(
            scopes=[
                SourceScope(
                    source_name="test",
                    page_hosts={"example.test"},
                    asset_hosts=set(),
                    redirect_hosts=set(),
                ),
                SourceScope(
                    source_name="canonical-contract",
                    page_hosts={"example.test"},
                    asset_hosts=set(),
                    redirect_hosts=set(),
                ),
            ],
        ),
    )
    priority_resolver = _TrackingPriorityResolver()
    next_sequence = iter(range(100))
    frontier = SchedulerFrontier(
        id_generator=_IdGenerator(),
        url_normalizer=_FixedNormalizer(),  # type: ignore[arg-type]
        priority_resolver=priority_resolver,  # type: ignore[arg-type]
        task_admitter=admitter,
        progress_state=SchedulerProgressState(),
        backlog_reader=SchedulerBacklogReader(
            host_queue=host_queue,
            delayed_queue=delayed_queue,
            max_feeds_per_host=0,
        ),
        seen_urls=seen_urls,
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        condition=condition,
        canonical_host_from_url=lambda url: urlsplit(url).hostname,
        is_closed=lambda: False,
        allocate_sequence=lambda: next(next_sequence),
        logger=TEST_LOGGER,
    )
    return (
        frontier,
        condition,
        host_queue,
        seen_urls,
        priority_resolver,
    )


def _discovery_task() -> CrawlTask:
    return CrawlTask(
        url=RAW_URL,
        source_name="canonical-contract",
        task_id="raw-task",
        kind=MediaKind.PAGE,
        depth=1,
        source_type="discovered_link",
    )


@pytest.mark.asyncio
async def test_enqueue_uses_canonical_task_for_every_admission_stage() -> None:
    url_filter = _TrackingFilter()
    frontier, condition, host_queue, seen_urls, priority_resolver = (
        _build_frontier(url_filter=url_filter)
    )
    drain_tasks: list[CrawlTask] = []

    def reserve(*, task: CrawlTask, queue_size: int) -> int:
        del queue_size
        drain_tasks.append(task)
        return 0

    frontier._reserve_high_pressure_drain_slot = reserve  # type: ignore[method-assign]
    raw_task = _discovery_task()

    async with condition:
        decision, scheduled_task, sequence = frontier.enqueue_locked(
            task=raw_task
        )

    assert decision.accepted is True
    assert decision.normalized_url == CANONICAL_URL
    assert scheduled_task is not None
    assert scheduled_task.url == CANONICAL_URL
    assert sequence == 0
    assert raw_task.url == RAW_URL
    assert [task.url for task in url_filter.tasks] == [CANONICAL_URL]
    assert [task.url for task in drain_tasks] == [CANONICAL_URL]
    assert [task.url for task in priority_resolver.tasks] == [CANONICAL_URL]
    queued_task = host_queue.snapshot_items()[0][3]
    assert queued_task.url == CANONICAL_URL
    assert seen_urls.is_seen(scheduler_task_identity_key(task=scheduled_task))
    assert not seen_urls.is_seen(scheduler_task_identity_key(task=raw_task))


@pytest.mark.asyncio
async def test_discovery_batch_is_rechecked_by_scheduler_filter() -> None:
    url_filter = _TrackingFilter(allowed=False)
    frontier, condition, host_queue, _, _ = _build_frontier(
        url_filter=url_filter
    )

    async with condition:
        decisions = frontier.enqueue_many_locked(tasks=[_discovery_task()])

    assert len(decisions) == 1
    assert decisions[0].accepted is False
    assert decisions[0].reason is ScheduleDecisionReason.URL_FILTERED
    assert decisions[0].normalized_url == CANONICAL_URL
    assert [task.url for task in url_filter.tasks] == [CANONICAL_URL]
    assert host_queue.queue_size == 0


def test_restore_admission_uses_canonical_task() -> None:
    url_filter = _TrackingFilter()
    frontier, _, _, _, _ = _build_frontier(url_filter=url_filter)
    raw_task = _discovery_task()
    envelope = SchedulerTaskEnvelope(
        task=raw_task,
        host="example.test",
        priority=raw_task.priority,
        sequence=3,
    )

    prepared = frontier.prepare_restored_envelope(
        envelope=envelope,
        canonical_host=HostNormalizer().normalize,
        queue_size=0,
        ready_pending_by_host={},
        ready_kind_pending_by_host={},
        seen_identity_keys=set(),
        use_host_advice=False,
    )

    assert prepared is not None
    assert prepared.task.url == CANONICAL_URL
    assert prepared.host == "example.test"
    assert [task.url for task in url_filter.tasks] == [CANONICAL_URL]
    assert raw_task.url == RAW_URL


def test_discovery_scope_preflight_uses_canonical_task() -> None:
    frontier, _, _, _, _ = _build_frontier(url_filter=_TrackingFilter())
    scoped_tasks: list[CrawlTask] = []

    def scope_rejection_reason(*, task: CrawlTask, host: str | None) -> None:
        assert host == "example.test"
        scoped_tasks.append(task)

    frontier._task_admitter = SimpleNamespace(
        scope_rejection_reason=scope_rejection_reason
    )

    decisions = frontier.discovery_scope_decisions_locked(
        tasks=[_discovery_task()]
    )

    assert decisions[0].normalized_url == CANONICAL_URL
    assert decisions[0].allowed is True
    assert [task.url for task in scoped_tasks] == [CANONICAL_URL]


def test_filter_bypass_parameters_are_absent() -> None:
    context_fields = {field.name for field in fields(AdmissionContext)}

    assert (
        "prefiltered"
        not in inspect.signature(UrlScheduler.enqueue_many).parameters
    )
    assert (
        "prefiltered"
        not in inspect.signature(
            SchedulerFrontier.enqueue_many_locked
        ).parameters
    )
    assert "enforce_url_filter" not in context_fields
    assert "normalized_url" not in context_fields
    assert (
        "enforce_url_filter"
        not in inspect.signature(
            AdmissionPrerequisiteChecker.evaluate
        ).parameters
    )
