"""Integration contracts for terminal task dead-letter recovery."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from config.collection.discovery import SchedulingSettings
from config.settings.crawler import CrawlStateStoreSettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.crawl_tasks.crawl_task_context import CrawlTaskContext
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.runtime.state.crawl_checkpoint_store import CrawlerCheckpointStore
from crawler.runtime.state.crawl_dead_letter_reader import (
    CrawlerDeadLetterReader,
)
from crawler.runtime.state.crawl_dead_letter_writer import (
    CrawlerDeadLetterWriter,
)
from crawler.runtime.state.crawl_state_reader import CrawlStateReader
from crawler.runtime.state.crawl_state_writer import CrawlStateWriter
from crawler.runtime.state.runtime_checkpoint_payload_builder import (
    RuntimeCheckpointPayloadBuilder,
)
from crawler.scheduling.completion.dead_letter_writer import DeadLetterRecord
from crawler.worker.pool.worker_pool_snapshot import WorkerPoolSnapshot
from orchestration.composition.runtime.scheduler import build_scheduler
from tests.support.logging import TEST_LOGGER


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


class _IdGenerator:
    def __init__(self) -> None:
        self._next = 0

    def generate(self) -> str:
        self._next += 1
        return f"generated-{self._next}"


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


class _BlockingWriter:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.records: list[DeadLetterRecord] = []

    async def append(self, record: DeadLetterRecord) -> None:
        self.started.set()
        await self.release.wait()
        self.records.append(record)


class _FailingWriter:
    def __init__(self) -> None:
        self.calls = 0

    async def append(self, record: DeadLetterRecord) -> None:
        del record
        self.calls += 1
        raise OSError("dead-letter disk unavailable")


class _CheckpointWriter:
    def __init__(self, *, succeeds: bool = True) -> None:
        self.enabled = True
        self.succeeds = succeeds
        self.calls: list[tuple[bool, int | None]] = []

    async def write_checkpoint(
        self,
        *,
        final: bool,
        max_queued_tasks: int | None = None,
    ) -> bool:
        self.calls.append((final, max_queued_tasks))
        return self.succeeds


class _WorkerPool:
    def snapshot(self, *, now: float) -> WorkerPoolSnapshot:
        del now
        return WorkerPoolSnapshot(
            size=0,
            effective_worker_count=0,
            retiring_worker_count=0,
            busy_worker_count=0,
            idle_worker_count=0,
            completed_task_count=0,
            failure_count=0,
            non_fatal_timeout_count=0,
            retry_exhausted_count=0,
            average_processing_seconds=0.0,
            longest_busy_seconds=0.0,
            active_tasks=(),
        )


def _build_test_scheduler(
    *,
    dead_letter_writer: object | None,
    id_generator: _IdGenerator,
    max_timeouts: int = 1,
):
    return build_scheduler(
        scheduling_settings=SchedulingSettings(
            max_total_attempts=10,
            max_deferrals=10,
            max_timeouts=max_timeouts,
            dead_letter_on_drain=False,
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
        id_generator=id_generator,
        logger_factory=_LoggerFactory(),
        dead_letter_writer=dead_letter_writer,
    )


def _task(*, task_id: str, url: str) -> CrawlTask:
    return CrawlTask(
        url=url,
        source_name="integration",
        task_id=task_id,
        kind=MediaKind.DOCUMENT,
        depth=2,
        source_type="seed",
        priority=7,
        parent_url="https://example.test/index",
        context=CrawlTaskContext(
            alt_text="recovery document",
            discovery_reason="document_link",
        ),
    )


def test_terminal_timeout_is_written_and_requeued_after_restart(
    tmp_path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "dead_letters.jsonl"
        settings = CrawlStateStoreSettings(
            requeue_dead_letters_on_start=True,
        )
        writer = CrawlerDeadLetterWriter(
            settings=settings,
            dead_letter_path=path,
            logger=TEST_LOGGER,
            clock=_Clock(),
        )
        scheduler = _build_test_scheduler(
            dead_letter_writer=writer,
            id_generator=_IdGenerator(),
        )
        original = _task(
            task_id="terminal-timeout",
            url="https://example.test/document.pdf",
        )

        decision = await scheduler.enqueue(original)
        assert decision.accepted is True
        active = await scheduler.get()
        await scheduler.complete(
            active,
            outcome="timeout",
            fields={
                "error_type": "TimeoutError",
                "kind": MediaKind.DOCUMENT,
            },
        )

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["schema_version"] == 1
        assert payload["status"] == "retry_exhausted"
        assert payload["original_outcome"] == "timeout"
        assert payload["detail"] == "max_timeouts_exceeded"
        assert payload["task"]["task_id"] == original.task_id
        assert payload["task"]["context"] == original.context.to_dict()
        assert payload["fields"]["kind"] == "document"

        restarted = _build_test_scheduler(
            dead_letter_writer=None,
            id_generator=_IdGenerator(),
        )
        reader = CrawlerDeadLetterReader(
            settings=settings,
            dead_letter_path=path,
            logger=TEST_LOGGER,
            task_deserializer=(
                restarted.create_checkpoint_task_deserializer()
            ),
        )
        state_reader = CrawlStateReader(
            settings=settings,
            logger=TEST_LOGGER,
            scheduler=restarted,
            checkpoint_store=None,
            dead_letter_reader=reader,
            current_seed_urls=(),
        )

        assert await state_reader.requeue_dead_letters() == 1
        restored = await restarted.get()
        assert restored == original

    asyncio.run(scenario())


def test_join_waits_until_terminal_dead_letter_is_durable() -> None:
    async def scenario() -> None:
        writer = _BlockingWriter()
        scheduler = _build_test_scheduler(
            dead_letter_writer=writer,
            id_generator=_IdGenerator(),
        )
        task = _task(
            task_id="pending-dead-letter",
            url="https://example.test/pending.pdf",
        )
        assert (await scheduler.enqueue(task)).accepted is True
        active = await scheduler.get()

        completion = asyncio.create_task(
            scheduler.complete(active, outcome="failed")
        )
        await writer.started.wait()
        join = asyncio.create_task(scheduler.join())
        await asyncio.sleep(0)
        assert join.done() is False

        state = await scheduler.export_state(
            max_queued_tasks=-1,
            include_seen_urls=True,
        )
        assert state["active_tasks"] == 1
        pending = state["requeued_inflight_tasks"]
        assert isinstance(pending, list)
        assert [item["task_id"] for item in pending] == [task.task_id]

        writer.release.set()
        await completion
        await asyncio.wait_for(join, timeout=1.0)
        assert [record.task for record in writer.records] == [task]

    asyncio.run(scenario())


def test_dead_letter_write_failure_requeues_instead_of_losing_task() -> None:
    async def scenario() -> None:
        writer = _FailingWriter()
        scheduler = _build_test_scheduler(
            dead_letter_writer=writer,
            id_generator=_IdGenerator(),
        )
        task = _task(
            task_id="write-failure",
            url="https://example.test/write-failure.pdf",
        )
        assert (await scheduler.enqueue(task)).accepted is True
        active = await scheduler.get()

        with pytest.raises(OSError, match="disk unavailable"):
            await scheduler.complete(active, outcome="failed")

        retried = await asyncio.wait_for(scheduler.get(), timeout=1.0)
        assert retried == task
        assert writer.calls == 1
        await scheduler.complete(retried, outcome="completed")
        await asyncio.wait_for(scheduler.join(), timeout=1.0)

    asyncio.run(scenario())


def test_cancelled_completion_waits_for_durable_dead_letter() -> None:
    async def scenario() -> None:
        writer = _BlockingWriter()
        scheduler = _build_test_scheduler(
            dead_letter_writer=writer,
            id_generator=_IdGenerator(),
        )
        task = _task(
            task_id="cancelled-write",
            url="https://example.test/cancelled-write.pdf",
        )
        assert (await scheduler.enqueue(task)).accepted is True
        active = await scheduler.get()
        completion = asyncio.create_task(
            scheduler.complete(active, outcome="failed")
        )
        await writer.started.wait()

        completion.cancel()
        await asyncio.sleep(0)
        assert completion.done() is False
        writer.release.set()
        with pytest.raises(asyncio.CancelledError):
            await completion

        assert [record.task for record in writer.records] == [task]
        await asyncio.wait_for(scheduler.join(), timeout=1.0)

    asyncio.run(scenario())


def test_reader_zero_limit_returns_no_tasks(tmp_path) -> None:
    async def scenario() -> None:
        path = tmp_path / "dead_letters.jsonl"
        write_settings = CrawlStateStoreSettings()
        writer = CrawlerDeadLetterWriter(
            settings=write_settings,
            dead_letter_path=path,
            logger=TEST_LOGGER,
            clock=_Clock(),
        )
        await writer.append(
            DeadLetterRecord(
                task=_task(
                    task_id="zero-limit",
                    url="https://example.test/zero.pdf",
                ),
                status="failed",
                original_outcome="failed",
                detail="failure",
                fields={},
            )
        )

        scheduler = _build_test_scheduler(
            dead_letter_writer=None,
            id_generator=_IdGenerator(),
        )
        reader = CrawlerDeadLetterReader(
            settings=CrawlStateStoreSettings(
                max_dead_letters_to_requeue=0,
            ),
            dead_letter_path=path,
            logger=TEST_LOGGER,
            task_deserializer=(
                scheduler.create_checkpoint_task_deserializer()
            ),
        )
        assert reader.load_batch().tasks == ()

    asyncio.run(scenario())


def test_bounded_batch_requires_explicit_acknowledgement(tmp_path) -> None:
    async def scenario() -> None:
        path = tmp_path / "dead_letters.jsonl"
        settings = CrawlStateStoreSettings(
            clear_dead_letters_on_requeue=True,
            max_dead_letters_to_requeue=1,
        )
        writer = CrawlerDeadLetterWriter(
            settings=settings,
            dead_letter_path=path,
            logger=TEST_LOGGER,
            clock=_Clock(),
        )
        tasks = (
            _task(
                task_id="first-bounded",
                url="https://example.test/first-bounded.pdf",
            ),
            _task(
                task_id="second-bounded",
                url="https://example.test/second-bounded.pdf",
            ),
        )
        for task in tasks:
            await writer.append(
                DeadLetterRecord(
                    task=task,
                    status="failed",
                    original_outcome="failed",
                    detail="bounded_recovery",
                    fields={},
                )
            )

        scheduler = _build_test_scheduler(
            dead_letter_writer=None,
            id_generator=_IdGenerator(),
        )
        reader = CrawlerDeadLetterReader(
            settings=settings,
            dead_letter_path=path,
            logger=TEST_LOGGER,
            task_deserializer=(
                scheduler.create_checkpoint_task_deserializer()
            ),
        )

        batch = reader.load_batch()
        assert batch.tasks == (tasks[0],)
        original_lines = path.read_text(encoding="utf-8").splitlines()
        assert len(original_lines) == 2

        assert reader.acknowledge(batch=batch, accepted=(False,)) == 0
        assert path.read_text(encoding="utf-8").splitlines() == original_lines

        assert reader.acknowledge(batch=batch, accepted=(True,)) == 1
        remaining = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert [payload["task"]["task_id"] for payload in remaining] == [
            tasks[1].task_id
        ]

    asyncio.run(scenario())


def test_clear_acknowledges_only_tasks_durable_in_full_checkpoint(
    tmp_path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "dead_letters.jsonl"
        settings = CrawlStateStoreSettings(
            clear_dead_letters_on_requeue=True,
            requeue_dead_letters_on_start=True,
        )
        writer = CrawlerDeadLetterWriter(
            settings=settings,
            dead_letter_path=path,
            logger=TEST_LOGGER,
            clock=_Clock(),
        )
        tasks = (
            _task(
                task_id="accepted",
                url="https://example.test/accepted.pdf",
            ),
            _task(
                task_id="rejected",
                url="https://example.test/rejected.pdf",
            ),
        )
        for task in tasks:
            await writer.append(
                DeadLetterRecord(
                    task=task,
                    status="failed",
                    original_outcome="failure",
                    detail="processor_failure",
                    fields={},
                )
            )

        scheduler = _build_test_scheduler(
            dead_letter_writer=None,
            id_generator=_IdGenerator(),
        )
        reader = CrawlerDeadLetterReader(
            settings=settings,
            dead_letter_path=path,
            logger=TEST_LOGGER,
            task_deserializer=(
                scheduler.create_checkpoint_task_deserializer()
            ),
        )
        batch = reader.load_batch()
        assert batch.tasks == tasks
        duplicate = await scheduler.enqueue(tasks[1])
        assert duplicate.accepted is True
        checkpoint_path = tmp_path / "crawler_runtime_checkpoint.json"
        checkpoint_store = CrawlerCheckpointStore(
            settings=settings,
            state_directory=tmp_path,
            checkpoint_path=checkpoint_path,
            logger=TEST_LOGGER,
            payload_builder=RuntimeCheckpointPayloadBuilder(clock=_Clock()),
        )
        checkpoint_writer = CrawlStateWriter(
            settings=settings,
            scheduler=scheduler,
            worker_pool=_WorkerPool(),
            checkpoint_store=checkpoint_store,
            metrics=None,
            run_context={},
            logger=TEST_LOGGER,
        )
        state_reader = CrawlStateReader(
            settings=settings,
            logger=TEST_LOGGER,
            scheduler=scheduler,
            checkpoint_store=checkpoint_store,
            dead_letter_reader=reader,
            checkpoint_writer=checkpoint_writer,
            current_seed_urls=(),
        )
        assert await state_reader.requeue_dead_letters() == 1

        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        queued = checkpoint["scheduler"]["queued_tasks"]
        assert {item["task_id"] for item in queued} == {
            task.task_id for task in tasks
        }

        remaining = reader.load_batch().tasks
        assert remaining == ()
        payloads = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert payloads == []

    asyncio.run(scenario())


def test_clear_keeps_dead_letter_when_checkpoint_write_fails(
    tmp_path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "dead_letters.jsonl"
        settings = CrawlStateStoreSettings(
            clear_dead_letters_on_requeue=True,
            requeue_dead_letters_on_start=True,
        )
        task = _task(
            task_id="checkpoint-failure",
            url="https://example.test/checkpoint-failure.pdf",
        )
        writer = CrawlerDeadLetterWriter(
            settings=settings,
            dead_letter_path=path,
            logger=TEST_LOGGER,
            clock=_Clock(),
        )
        await writer.append(
            DeadLetterRecord(
                task=task,
                status="failed",
                original_outcome="failed",
                detail="processor_failure",
                fields={},
            )
        )
        scheduler = _build_test_scheduler(
            dead_letter_writer=None,
            id_generator=_IdGenerator(),
        )
        reader = CrawlerDeadLetterReader(
            settings=settings,
            dead_letter_path=path,
            logger=TEST_LOGGER,
            task_deserializer=(
                scheduler.create_checkpoint_task_deserializer()
            ),
        )
        checkpoint_writer = _CheckpointWriter(succeeds=False)
        state_reader = CrawlStateReader(
            settings=settings,
            logger=TEST_LOGGER,
            scheduler=scheduler,
            checkpoint_store=None,
            dead_letter_reader=reader,
            checkpoint_writer=checkpoint_writer,
            current_seed_urls=(),
        )

        assert await state_reader.requeue_dead_letters() == 1
        assert checkpoint_writer.calls == [(False, -1)]
        assert reader.load_batch().tasks == (task,)

    asyncio.run(scenario())


def test_concurrent_appends_produce_complete_jsonl_records(tmp_path) -> None:
    async def scenario() -> None:
        path = tmp_path / "dead_letters.jsonl"
        writer = CrawlerDeadLetterWriter(
            settings=CrawlStateStoreSettings(),
            dead_letter_path=path,
            logger=TEST_LOGGER,
            clock=_Clock(),
        )
        records = tuple(
            DeadLetterRecord(
                task=_task(
                    task_id=f"concurrent-{index}",
                    url=f"https://example.test/{index}.pdf",
                ),
                status="failed",
                original_outcome="failed",
                detail="concurrent_failure",
                fields={"index": index},
            )
            for index in range(24)
        )

        await asyncio.gather(*(writer.append(record) for record in records))

        payloads = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(payloads) == len(records)
        assert {payload["task"]["task_id"] for payload in payloads} == {
            record.task.task_id for record in records
        }

    asyncio.run(scenario())


def test_dead_letter_statuses_only_accept_canonical_values() -> None:
    canonical = (
        "retry_exhausted",
        "failed",
        "cancelled",
    )
    assert CrawlStateStoreSettings().dead_letter_statuses == canonical
    for alias in (
        "failure",
        "canceled",
        "timeout",
        "deferred",
        "retry-exhausted",
    ):
        with pytest.raises(ValueError):
            CrawlStateStoreSettings(dead_letter_statuses=(alias,))

    project_root = Path(__file__).resolve().parents[4]
    schema = json.loads(
        (project_root / "docs" / "configuration_schema.json").read_text(
            encoding="utf-8"
        )
    )
    status_schema = schema["$defs"]["CrawlStateStoreSettings"]["properties"][
        "dead_letter_statuses"
    ]
    assert tuple(status_schema["default"]) == canonical
    assert set(status_schema["items"]["enum"]) == set(canonical)
