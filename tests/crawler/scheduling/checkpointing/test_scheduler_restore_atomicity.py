"""Atomic scheduler-checkpoint restore regression tests."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from urllib.parse import urlsplit

import pytest

from config.collection.discovery import SchedulingSettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.governance.domains.host_normalizer import HostNormalizer
from orchestration.composition.runtime.scheduler import build_scheduler
from tests.support.logging import TEST_LOGGER


class _IdGenerator:
    def __init__(self) -> None:
        self._next = 0

    def generate(self) -> str:
        self._next += 1
        return f"restore-{self._next}"


class _LoggerFactory:
    def get_logger_for(self, _component: object) -> object:
        return TEST_LOGGER


class _UrlNormalizer:
    def normalize(self, value: object) -> str:
        return value.strip() if isinstance(value, str) else ""


class _HostExtractor:
    def extract(self, url: str) -> str | None:
        return urlsplit(url).hostname


class _PriorityResolver:
    def __call__(self, task: CrawlTask) -> int:
        return task.priority


class _SourceScopeRegistry:
    def __init__(self) -> None:
        self._scopes: dict[str, object] = {}

    def require(self, source_name: str) -> object:
        from crawler.governance.source_scope.source_scope_registry import (
            SourceScope,
        )

        if source_name not in self._scopes:
            self._scopes[source_name] = SourceScope(
                source_name=source_name,
                page_hosts={"example.test"},
                asset_hosts=set(),
                redirect_hosts=set(),
            )
        return self._scopes[source_name]


def _scheduler():
    return build_scheduler(
        scheduling_settings=SchedulingSettings(
            max_total_attempts=10,
            max_deferrals=10,
            max_timeouts=3,
            timeout_retry_wait_seconds=0,
        ),
        url_normalizer=_UrlNormalizer(),
        url_filter=None,
        host_extractor=_HostExtractor(),
        host_normalizer=HostNormalizer(),
        priority_resolver=_PriorityResolver(),
        blacklist_repository=None,
        metrics=None,
        host_budget_tracker=None,
        source_scope_registry=_SourceScopeRegistry(),
        host_media_byte_budget=None,
        rate_limiter=None,
        id_generator=_IdGenerator(),
        logger_factory=_LoggerFactory(),
    )


def _task(*, task_id: str, path: str) -> CrawlTask:
    return CrawlTask(
        url=f"https://example.test/{path}",
        source_name="atomic-restore",
        task_id=task_id,
        kind=MediaKind.DOCUMENT,
        source_type="seed",
        priority=7,
    )


async def _export(scheduler) -> dict[str, object]:
    return await scheduler.export_state(
        max_queued_tasks=-1,
        include_seen_urls=True,
    )


def _malformed_delayed_wait(checkpoint: dict[str, object]) -> None:
    queued = checkpoint["queued_tasks"]
    assert isinstance(queued, list)
    delayed = checkpoint["delayed_tasks"]
    assert isinstance(delayed, list)
    item = queued.pop()
    assert isinstance(item, dict)
    item["delay_remaining_seconds"] = "invalid"
    delayed.append(item)


def _malformed_requeued_item(checkpoint: dict[str, object]) -> None:
    checkpoint["requeued_inflight_tasks"] = [{"url": "missing-envelope"}]


def _malformed_dispatching_item(checkpoint: dict[str, object]) -> None:
    checkpoint["dispatching_tasks"] = [{"url": "missing-envelope"}]


def _duplicate_identity(checkpoint: dict[str, object]) -> None:
    queued = checkpoint["queued_tasks"]
    assert isinstance(queued, list)
    duplicate = deepcopy(queued[0])
    assert isinstance(duplicate, dict)
    sequence = duplicate["sequence"]
    assert isinstance(sequence, int)
    duplicate["sequence"] = sequence + 1
    duplicate["task_id"] = "another-id"
    queued.append(duplicate)


def _duplicate_sequence(checkpoint: dict[str, object]) -> None:
    queued = checkpoint["queued_tasks"]
    assert isinstance(queued, list)
    duplicate = deepcopy(queued[0])
    assert isinstance(duplicate, dict)
    duplicate["url"] = "https://example.test/other.pdf"
    duplicate["task_id"] = "other-id"
    queued.append(duplicate)


def _malformed_seen_entry(checkpoint: dict[str, object]) -> None:
    checkpoint["seen_url_entries"] = [
        {"url": "valid", "seen_at": float("nan")}
    ]


def _malformed_progress_counter(checkpoint: dict[str, object]) -> None:
    progress = checkpoint["progress_counters"]
    assert isinstance(progress, dict)
    progress["completed_by_outcome"] = {"completed": "invalid"}


def _malformed_next_sequence(checkpoint: dict[str, object]) -> None:
    checkpoint["next_sequence"] = -1


def _malformed_schema_version(checkpoint: dict[str, object]) -> None:
    checkpoint["schema_version"] = 2.0


def _malformed_envelope_depth(checkpoint: dict[str, object]) -> None:
    queued = checkpoint["queued_tasks"]
    assert isinstance(queued, list)
    item = queued[0]
    assert isinstance(item, dict)
    item["depth"] = "invalid"


def _malformed_envelope_context(checkpoint: dict[str, object]) -> None:
    queued = checkpoint["queued_tasks"]
    assert isinstance(queued, list)
    item = queued[0]
    assert isinstance(item, dict)
    item["context"] = "invalid"


@pytest.mark.parametrize(
    "mutate",
    (
        _malformed_delayed_wait,
        _malformed_requeued_item,
        _malformed_dispatching_item,
        _duplicate_identity,
        _duplicate_sequence,
        _malformed_seen_entry,
        _malformed_progress_counter,
        _malformed_next_sequence,
        _malformed_schema_version,
        _malformed_envelope_depth,
        _malformed_envelope_context,
    ),
)
def test_late_checkpoint_validation_failure_is_atomic(mutate) -> None:
    async def scenario() -> None:
        scheduler = _scheduler()
        assert (
            await scheduler.enqueue(
                _task(task_id="late-validation", path="late.pdf")
            )
        ).accepted
        before = await _export(scheduler)
        malformed = deepcopy(before)
        mutate(malformed)

        with pytest.raises((TypeError, ValueError)):
            await scheduler.restore_state(payload=malformed)

        assert await _export(scheduler) == before

    asyncio.run(scenario())


def test_malformed_queued_tasks_preserves_all_live_scheduler_state() -> None:
    async def scenario() -> None:
        scheduler = _scheduler()
        assert (
            await scheduler.enqueue(
                _task(task_id="live-queued", path="live.pdf")
            )
        ).accepted
        before = await _export(scheduler)
        malformed = deepcopy(before)
        malformed["queued_tasks"] = "invalid"

        with pytest.raises(ValueError, match="queued_tasks must be a list"):
            await scheduler.restore_state(payload=malformed)

        assert await _export(scheduler) == before

    asyncio.run(scenario())


def test_malformed_retry_budget_preserves_all_live_scheduler_state() -> None:
    async def scenario() -> None:
        scheduler = _scheduler()
        task = _task(task_id="live-retry", path="retry.pdf")
        assert (await scheduler.enqueue(task)).accepted
        active = await scheduler.get()
        await scheduler.complete(
            active,
            outcome="timeout",
            fields={"error_type": "TimeoutError"},
        )
        before = await _export(scheduler)
        assert before["retry_budget"] != {}
        malformed = deepcopy(before)
        retry_budget = malformed["retry_budget"]
        assert isinstance(retry_budget, dict)
        retry_state = retry_budget["id:live-retry"]
        assert isinstance(retry_state, dict)
        retry_state["http_request_attempts"] = ["invalid"]

        with pytest.raises(
            ValueError, match="non-integer http_request_attempts"
        ):
            await scheduler.restore_state(payload=malformed)

        assert await _export(scheduler) == before

    asyncio.run(scenario())


def test_valid_replacement_plan_can_restore_the_same_live_task() -> None:
    async def scenario() -> None:
        scheduler = _scheduler()
        assert (
            await scheduler.enqueue(
                _task(task_id="same-live-task", path="same.pdf")
            )
        ).accepted
        checkpoint = await _export(scheduler)

        assert await scheduler.restore_state(payload=checkpoint) == 1
        assert await _export(scheduler) == checkpoint

    asyncio.run(scenario())


def test_restore_seen_entries_uses_registry_last_occurrence_semantics() -> (
    None
):
    async def scenario() -> None:
        scheduler = _scheduler()
        assert (
            await scheduler.enqueue(
                _task(task_id="seen-restore", path="seen.pdf")
            )
        ).accepted
        checkpoint = await _export(scheduler)
        checkpoint["seen_url_entries"] = [
            {"url": "id:old", "seen_at": 1.0},
            {"url": "id:other", "seen_at": 2.0},
            {"url": "id:old", "seen_at": 3.0},
        ]

        assert await scheduler.restore_state(payload=checkpoint) == 1
        restored = await _export(scheduler)
        assert restored["seen_url_entries"] == [
            {"url": "id:other", "seen_at": 2.0},
            {"url": "id:old", "seen_at": 3.0},
        ]

    asyncio.run(scenario())
