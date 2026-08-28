"""Dispatch-time forbidden-endpoint suppression contract tests.

A task that was already enqueued when its endpoint answered HTTP 403 must
be dropped before any fetch attempt, exactly like tasks that already
returned HTTP 304 in this run.
"""

from __future__ import annotations

import asyncio

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.scheduling.completion.run_url_feedback import RunUrlFeedback
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


class _NoWaitReader:
    async def governance_wait_seconds(
        self,
        *,
        host: str | None,
    ) -> None:
        del host
        return None


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def warning(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))

    def debug(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def _task(*, task_id: str, url: str) -> CrawlTask:
    return CrawlTask(
        url=url,
        source_name="test",
        task_id=task_id,
        kind=MediaKind.DOCUMENT,
    )


def _skip_reason(feedback: RunUrlFeedback):
    def skip_reason(task: CrawlTask) -> str | None:
        if feedback.was_not_modified(task=task):
            return "not_modified_this_run"
        if feedback.is_forbidden_endpoint(url=task.url):
            return "forbidden_endpoint_this_run"
        return None

    return skip_reason


def _build_dispatcher(
    feedback: RunUrlFeedback,
    logger: _Logger,
) -> tuple[SchedulerDispatcher, HostTaskQueue, ActiveTaskRegistry]:
    normalizer = _HostNormalizer()
    host_queue = HostTaskQueue(host_normalizer=normalizer)
    delayed_queue = DelayedTaskQueue(host_normalizer=normalizer)
    eligibility_queue = HostEligibilityQueue()
    registry = ActiveTaskRegistry(host_normalizer=normalizer)
    dispatcher = SchedulerDispatcher(
        dispatch_wait_reader=_NoWaitReader(),
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        host_eligibility_queue=eligibility_queue,
        inflight_count_by_host=registry.inflight_count_by_host,
        condition=asyncio.Condition(),
        active_registry=registry,
        logger=logger,
        skip_reason_before_fetch=_skip_reason(feedback),
    )
    return dispatcher, host_queue, registry


async def _dispatch_task(
    dispatcher: SchedulerDispatcher,
    host_queue: HostTaskQueue,
    *,
    task: CrawlTask,
) -> CrawlTask | None:
    host_queue.push(host="example.test", priority=0, sequence=0, task=task)
    return await dispatcher.dispatch_for_host(host="example.test")


def test_queued_task_is_skipped_after_endpoint_answered_403() -> None:
    async def scenario() -> None:
        feedback = RunUrlFeedback(normalize_url=lambda url: url)
        logger = _Logger()
        dispatcher, host_queue, registry = _build_dispatcher(
            feedback=feedback,
            logger=logger,
        )

        task_a = _task(
            task_id="a",
            url="https://example.test/data?q=1",
        )
        task_b = _task(
            task_id="b",
            url="https://example.test/data?q=2",
        )

        fetched: list[str] = []

        dispatched_a = await _dispatch_task(
            dispatcher,
            host_queue,
            task=task_a,
        )
        assert dispatched_a is task_a
        fetched.append(dispatched_a.url)

        # Task B was already queued when fetch of A answered HTTP 403.
        # The endpoint identity ignores query values, so B shares it.
        feedback.remember_forbidden_endpoint(url=task_a.url)

        dispatched_b = await _dispatch_task(
            dispatcher,
            host_queue,
            task=task_b,
        )

        assert dispatched_b is None
        # No second network request: B was never handed to the fetcher.
        assert fetched == [task_a.url]
        skip_events = [
            (event, fields)
            for event, fields in logger.events
            if event == "task_skipped_before_fetch"
        ]
        assert len(skip_events) == 1
        assert skip_events[0][1]["reason"] == "forbidden_endpoint_this_run"
        assert skip_events[0][1]["task_id"] == "b"

    asyncio.run(scenario())


def test_queued_task_is_skipped_after_not_modified() -> None:
    async def scenario() -> None:
        feedback = RunUrlFeedback(normalize_url=lambda url: url)
        logger = _Logger()
        dispatcher, host_queue, registry = _build_dispatcher(
            feedback=feedback,
            logger=logger,
        )

        task = _task(task_id="c", url="https://example.test/page")
        feedback.remember_not_modified(task=task)

        dispatched = await _dispatch_task(
            dispatcher,
            host_queue,
            task=task,
        )

        assert dispatched is None
        skip_events = [
            (event, fields)
            for event, fields in logger.events
            if event == "task_skipped_before_fetch"
        ]
        assert skip_events[0][1]["reason"] == "not_modified_this_run"

    asyncio.run(scenario())


def test_unforbidden_queued_task_is_still_dispatched() -> None:
    async def scenario() -> None:
        feedback = RunUrlFeedback(normalize_url=lambda url: url)
        logger = _Logger()
        dispatcher, host_queue, registry = _build_dispatcher(
            feedback=feedback,
            logger=logger,
        )

        task = _task(task_id="d", url="https://example.test/other")
        dispatched = await _dispatch_task(
            dispatcher,
            host_queue,
            task=task,
        )

        assert dispatched is task
        assert all(
            event != "task_skipped_before_fetch" for event, _ in logger.events
        )

    asyncio.run(scenario())
