"""Host-first dispatch eligibility contract tests.

Host governance waits must gate dispatch at host level: a paced or
inflight-blocked host keeps its tasks queued instead of popping and
re-delaying them one by one. Only eligible hosts may have tasks popped.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.scheduling.dispatch.scheduler_dispatcher import (
    SchedulerDispatcher,
)
from crawler.scheduling.progress.active_task_registry import ActiveTaskRegistry
from crawler.scheduling.queueing.delayed_task_queue import DelayedTaskQueue
from crawler.scheduling.queueing.host_eligibility_queue import (
    HostEligibilityQueue,
)
from crawler.scheduling.queueing.host_task_queue import HostTaskQueue


class _HostNormalizer:
    def normalize(self, host: object) -> object:
        return host

    def require(self, host: str | None) -> str:
        assert host is not None
        return host


class _ScriptedReader:
    def __init__(self, waits: list[float | None] | None = None) -> None:
        self._waits: deque[float | None] = deque(waits or [])

    async def governance_wait_seconds(
        self,
        *,
        host: str | None,
    ) -> float | None:
        del host
        if self._waits:
            return self._waits.popleft()
        return None


class _Logger:
    def info(self, event: str, **fields: object) -> None:
        del event, fields

    def warning(self, event: str, **fields: object) -> None:
        del event, fields

    def debug(self, event: str, **fields: object) -> None:
        del event, fields


def _task(*, task_id: str, url: str) -> CrawlTask:
    return CrawlTask(
        url=url,
        source_name="test",
        task_id=task_id,
        kind=MediaKind.DOCUMENT,
    )


def _build_dispatcher(
    *,
    reader: _ScriptedReader,
    max_inflight_per_host: int | None = None,
) -> tuple[
    SchedulerDispatcher,
    HostTaskQueue,
    HostEligibilityQueue,
    ActiveTaskRegistry,
]:
    normalizer = _HostNormalizer()
    host_queue = HostTaskQueue(host_normalizer=normalizer)
    delayed_queue = DelayedTaskQueue(host_normalizer=normalizer)
    eligibility_queue = HostEligibilityQueue()
    registry = ActiveTaskRegistry(host_normalizer=normalizer)
    dispatcher = SchedulerDispatcher(
        dispatch_wait_reader=reader,
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        host_eligibility_queue=eligibility_queue,
        inflight_count_by_host=registry.inflight_count_by_host,
        max_inflight_per_host=max_inflight_per_host,
        condition=asyncio.Condition(),
        active_registry=registry,
        logger=_Logger(),
    )
    return dispatcher, host_queue, eligibility_queue, registry


def test_host_pacing_keeps_task_queued_until_eligible() -> None:
    async def scenario() -> None:
        reader = _ScriptedReader(waits=[0.25, None])
        dispatcher, host_queue, eligibility_queue, _registry = (
            _build_dispatcher(reader=reader)
        )
        task = _task(task_id="a", url="https://example.test/a")
        host_queue.push(
            host="example.test",
            priority=0,
            sequence=0,
            task=task,
        )

        dispatched = await dispatcher.dispatch_for_host(
            host="example.test",
        )
        assert dispatched is None
        # The task was never popped: host pacing must not requeue tasks.
        assert host_queue.queue_size == 1
        assert host_queue.pending_for_host("example.test") == 1
        assert eligibility_queue.next_eligible_at("example.test") is not None

        dispatched = await dispatcher.dispatch_for_host(
            host="example.test",
        )
        assert dispatched is task
        assert host_queue.queue_size == 0

    asyncio.run(scenario())


def test_inflight_cap_blocks_host_until_completion_releases() -> None:
    async def scenario() -> None:
        reader = _ScriptedReader()
        dispatcher, host_queue, eligibility_queue, registry = (
            _build_dispatcher(
                reader=reader,
                max_inflight_per_host=1,
            )
        )
        task_a = _task(task_id="a", url="https://example.test/a")
        task_b = _task(task_id="b", url="https://example.test/b")
        host_queue.push(
            host="example.test", priority=0, sequence=0, task=task_a
        )
        host_queue.push(
            host="example.test", priority=0, sequence=1, task=task_b
        )

        dispatched_a = await dispatcher.dispatch_for_host(
            host="example.test",
        )
        assert dispatched_a is task_a
        assert registry.count == 1

        # Host is at the inflight cap: B must stay queued, host blocked.
        dispatched_b = await dispatcher.dispatch_for_host(
            host="example.test",
        )
        assert dispatched_b is None
        assert host_queue.pending_for_host("example.test") == 1
        assert math.isinf(eligibility_queue.next_eligible_at("example.test"))

        # A completion frees the slot and releases the blocked host.
        registry.remove(task=task_a)
        eligibility_queue.release_blocked("example.test")
        dispatched_b = await dispatcher.dispatch_for_host(
            host="example.test",
        )
        assert dispatched_b is task_b
        assert host_queue.queue_size == 0

    asyncio.run(scenario())


def test_stale_eligibility_entry_is_cleaned_before_candidate_pick() -> None:
    async def scenario() -> None:
        reader = _ScriptedReader()
        dispatcher, host_queue, eligibility_queue, _registry = (
            _build_dispatcher(reader=reader)
        )
        task = _task(task_id="c", url="https://example.net/c")
        # Stale entry for a host that no longer has queued tasks.
        eligibility_queue.upsert(
            host="stale.example",
            next_eligible_at=0.0,
            inflight=0,
        )
        host_queue.push(host="example.net", priority=0, sequence=0, task=task)

        # The stale host is cleaned lazily by the pick path, then the real
        # host is dispatched.
        get_task = asyncio.create_task(dispatcher.get())
        dispatched = await asyncio.wait_for(get_task, timeout=1.0)
        assert dispatched is task
        assert eligibility_queue.next_eligible_at("stale.example") is None

    asyncio.run(scenario())


def test_release_blocked_is_ignored_for_non_blocked_host() -> None:
    async def scenario() -> None:
        reader = _ScriptedReader(waits=[0.5])
        dispatcher, host_queue, eligibility_queue, _registry = (
            _build_dispatcher(reader=reader)
        )
        task = _task(task_id="d", url="https://example.test/d")
        host_queue.push(host="example.test", priority=0, sequence=0, task=task)

        dispatched = await dispatcher.dispatch_for_host(
            host="example.test",
        )
        assert dispatched is None
        next_eligible_at = eligibility_queue.next_eligible_at("example.test")
        assert next_eligible_at is not None
        assert not math.isinf(next_eligible_at)

        eligibility_queue.release_blocked("example.test")
        assert (
            eligibility_queue.next_eligible_at("example.test")
            == next_eligible_at
        )

    asyncio.run(scenario())


def test_blocked_host_reports_no_ready_in_seconds() -> None:
    async def scenario() -> None:
        reader = _ScriptedReader()
        dispatcher, host_queue, eligibility_queue, _registry = (
            _build_dispatcher(
                reader=reader,
                max_inflight_per_host=1,
            )
        )
        task_a = _task(task_id="e", url="https://example.test/e")
        task_b = _task(task_id="f", url="https://example.test/f")
        host_queue.push(
            host="example.test", priority=0, sequence=0, task=task_a
        )
        host_queue.push(
            host="example.test", priority=0, sequence=0, task=task_b
        )

        assert (
            await dispatcher.dispatch_for_host(host="example.test") is task_a
        )
        assert await dispatcher.dispatch_for_host(host="example.test") is None

        assert eligibility_queue.next_ready_in_seconds() is None
        assert eligibility_queue.has_ready() is False

    asyncio.run(scenario())
