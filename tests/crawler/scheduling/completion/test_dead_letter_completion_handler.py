"""Dead-letter contracts at the scheduler completion boundary."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from config.collection.discovery import SchedulingSettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.scheduling.admission.admission_task_identity import (
    scheduler_task_identity_key,
)
from crawler.scheduling.completion.run_url_feedback import RunUrlFeedback
from crawler.scheduling.completion.scheduler_retry_budget import (
    SchedulerRetryBudget,
)
from crawler.scheduling.completion.task_completion_handler import (
    SchedulerCompletionHandler,
)
from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry
from crawler.scheduling.progress.active_task_registry import ActiveTaskRegistry
from crawler.scheduling.progress.scheduler_progress_state import (
    SchedulerProgressState,
)
from crawler.scheduling.queueing.delayed_task_queue import DelayedTaskQueue
from crawler.scheduling.queueing.host_task_queue import HostTaskQueue
from tests.support.logging import TEST_LOGGER


class _SpyWriter:
    def __init__(self, condition: asyncio.Condition) -> None:
        self._condition = condition
        self.records = []

    async def append(self, record) -> None:
        assert self._condition.locked() is False
        self.records.append(record)


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, object]]] = []

    def debug(self, *_args: object, **_kwargs: object) -> None:
        pass

    def warning(self, event: str, **fields: object) -> None:
        self.warnings.append((event, fields))


def _task(outcome: str) -> CrawlTask:
    return CrawlTask(
        url=f"https://example.test/{outcome}",
        source_name="completion",
        task_id=f"task-{outcome}",
        kind=MediaKind.PAGE,
    )


def _handler(
    *,
    task: CrawlTask,
    is_closed: bool,
    max_timeouts: int,
    max_deferrals: int = 10,
    logger: Any = TEST_LOGGER,
):
    condition = asyncio.Condition()
    normalizer = HostNormalizer()
    active_registry = ActiveTaskRegistry(host_normalizer=normalizer)
    active_registry.add(
        host="example.test",
        priority=task.priority,
        sequence=0,
        task=task,
    )
    writer = _SpyWriter(condition)
    seen_urls = SeenUrlRegistry(max_seen=100)
    seen_urls.remember(scheduler_task_identity_key(task=task))
    host_queue = HostTaskQueue(host_normalizer=normalizer)
    delayed_queue = DelayedTaskQueue(host_normalizer=normalizer)
    retry_budget = SchedulerRetryBudget(
        settings=SchedulingSettings(
            max_total_attempts=10,
            max_deferrals=max_deferrals,
            max_timeouts=max_timeouts,
            dead_letter_on_drain=False,
        ),
        logger=logger,
        is_drained=lambda: False,
    )
    handler = SchedulerCompletionHandler(
        condition=condition,
        active_registry=active_registry,
        retry_rules=retry_budget,
        seen_urls=seen_urls,
        host_queue=host_queue,
        delayed_queue=delayed_queue,
        progress_state=SchedulerProgressState(),
        run_url_feedback=RunUrlFeedback(normalize_url=lambda url: url),
        logger=logger,
        is_closed=lambda: is_closed,
        dead_letter_writer=writer,
    )
    return handler, writer, host_queue, delayed_queue, seen_urls


@pytest.mark.parametrize(
    ("outcome", "is_closed", "expected_status", "max_deferrals"),
    (
        ("timeout", False, "retry_exhausted", 10),
        ("deferred", False, "retry_exhausted", 1),
        ("failure", False, "failed", 10),
        ("failure", True, "failed", 10),
        ("failed", False, "failed", 10),
        ("failed", True, "failed", 10),
        ("cancelled", True, "cancelled", 10),
        ("interrupted", False, "cancelled", 10),
        ("interrupted", True, "cancelled", 10),
    ),
)
def test_terminal_outcome_is_written_once_after_condition_unlock(
    outcome: str,
    is_closed: bool,
    expected_status: str,
    max_deferrals: int,
) -> None:
    async def scenario() -> None:
        task = _task(outcome)
        handler, writer, host_queue, delayed_queue, seen_urls = _handler(
            task=task,
            is_closed=is_closed,
            max_timeouts=1,
            max_deferrals=max_deferrals,
        )

        fields: dict[str, object] = {"reason": f"{outcome}_reason"}
        if outcome == "deferred":
            fields.update(
                {
                    "reason": "fetch_timeout",
                    "counts_toward_task_retry_budget": True,
                }
            )
        await handler.complete(
            task,
            outcome=outcome,
            fields=fields,
        )

        assert len(writer.records) == 1
        record = writer.records[0]
        assert record.status == expected_status
        assert record.original_outcome == outcome
        assert host_queue.queue_size == 0
        assert delayed_queue.queue_size == 0
        assert seen_urls.size == 0

    asyncio.run(scenario())


def test_retry_exhaustion_logs_dead_letter_action_when_writer_is_active() -> (
    None
):
    async def scenario() -> None:
        logger = _RecordingLogger()
        task = _task("timeout")
        handler, _, _, _, _ = _handler(
            task=task,
            is_closed=False,
            max_timeouts=1,
            logger=logger,
        )

        await handler.complete(task, outcome="timeout")

        retry_warning = next(
            fields
            for event, fields in logger.warnings
            if event == "scheduler_task_retry_exhausted"
        )
        assert retry_warning["action"] == "dead_letter"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("outcome", "max_timeouts"),
    (
        ("timeout", 2),
        ("cancelled", 1),
    ),
)
def test_nonterminal_retry_is_requeued_without_dead_letter(
    outcome: str,
    max_timeouts: int,
) -> None:
    async def scenario() -> None:
        task = _task(outcome)
        handler, writer, host_queue, delayed_queue, seen_urls = _handler(
            task=task,
            is_closed=False,
            max_timeouts=max_timeouts,
        )

        await handler.complete(task, outcome=outcome)

        assert writer.records == []
        assert host_queue.queue_size + delayed_queue.queue_size == 1
        assert seen_urls.size == 1

    asyncio.run(scenario())
