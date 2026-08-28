"""Regression tests for bounded worker-pool shutdown phases."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from crawler.worker.pool.worker_pool import WorkerPoolShutdown


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    def debug(self, event: str, **fields: object) -> None:
        self.events.append(("debug", event, fields))

    def info(self, event: str, **fields: object) -> None:
        self.events.append(("info", event, fields))

    def warning(self, event: str, **fields: object) -> None:
        self.events.append(("warning", event, fields))

    def error(self, event: str, **fields: object) -> None:
        self.events.append(("error", event, fields))


class _FailureRecorder:
    def __init__(self) -> None:
        self.runtime_failures: list[tuple[int, BaseException]] = []

    def record_runtime_failure(
        self,
        *,
        worker_id: int,
        cause: BaseException,
    ) -> None:
        self.runtime_failures.append((worker_id, cause))


@dataclass
class _Worker:
    worker_task: asyncio.Task[None]


class _Pool:
    def __init__(
        self,
        *,
        workers: dict[int, _Worker] | None = None,
        finalizer_tasks: set[asyncio.Task[None]] | None = None,
    ) -> None:
        self.lock = asyncio.Lock()
        self.closed = False
        self.workers = {} if workers is None else workers
        self.finalizer_tasks = (
            set() if finalizer_tasks is None else finalizer_tasks
        )
        self.logger = _RecordingLogger()


def _shutdown(
    pool: _Pool,
    *,
    stop_timeout_seconds: float,
    finalizer_drain_timeout_seconds: float,
) -> WorkerPoolShutdown:
    return WorkerPoolShutdown(
        pool,  # type: ignore[arg-type]
        failure_recorder=_FailureRecorder(),  # type: ignore[arg-type]
        stop_timeout_seconds=stop_timeout_seconds,
        finalizer_drain_timeout_seconds=finalizer_drain_timeout_seconds,
    )


def _fields_contain_worker_id(
    fields: dict[str, object],
    worker_id: int,
) -> bool:
    for value in fields.values():
        if value == worker_id:
            return True
        if isinstance(value, (list, set, tuple)) and worker_id in value:
            return True
    return False


@pytest.mark.asyncio
async def test_aclose_cancels_worker_that_stops_within_budget() -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()

    async def worker_runtime() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            raise

    worker_task = asyncio.create_task(worker_runtime())
    await started.wait()
    pool = _Pool(workers={3: _Worker(worker_task=worker_task)})
    shutdown = _shutdown(
        pool,
        stop_timeout_seconds=0.1,
        finalizer_drain_timeout_seconds=0.1,
    )

    await shutdown.aclose()

    assert pool.closed is True
    assert cancellation_seen.is_set()
    assert worker_task.cancelled()
    assert not [event for event in pool.logger.events if event[0] == "error"]


@pytest.mark.asyncio
async def test_aclose_succeeds_when_pool_has_no_tasks_to_wait_for() -> None:
    pool = _Pool()
    shutdown = _shutdown(
        pool,
        stop_timeout_seconds=0.02,
        finalizer_drain_timeout_seconds=0.02,
    )

    await shutdown.aclose()

    assert pool.closed is True
    assert not [event for event in pool.logger.events if event[0] == "error"]


@pytest.mark.asyncio
async def test_aclose_does_not_wait_a_second_stop_budget_for_live_worker() -> (
    None
):
    stop_timeout_seconds = 0.02
    started = asyncio.Event()
    release = asyncio.Event()
    cancellation_count = 0

    async def cancellation_delayed_worker() -> None:
        nonlocal cancellation_count
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_count += 1

    worker_task = asyncio.create_task(cancellation_delayed_worker())
    await started.wait()
    pool = _Pool(workers={7: _Worker(worker_task=worker_task)})
    shutdown = _shutdown(
        pool,
        stop_timeout_seconds=stop_timeout_seconds,
        finalizer_drain_timeout_seconds=0.1,
    )
    loop = asyncio.get_running_loop()
    started_at = loop.time()

    try:
        with pytest.raises(RuntimeError, match=r"7"):
            await shutdown.aclose()
        elapsed = loop.time() - started_at

        assert elapsed < stop_timeout_seconds * 2
        assert cancellation_count >= 2
        error_fields = [
            fields
            for level, _event, fields in pool.logger.events
            if level == "error"
        ]
        assert error_fields
        assert any(
            _fields_contain_worker_id(fields, 7) for fields in error_fields
        )
    finally:
        release.set()
        await asyncio.wait_for(worker_task, timeout=0.2)


@pytest.mark.asyncio
async def test_finalizer_drain_has_its_own_bounded_timeout() -> None:
    finalizer_drain_timeout_seconds = 0.02
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()

    async def slow_finalizer() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            raise

    finalizer_task = asyncio.create_task(slow_finalizer())
    await started.wait()
    pool = _Pool(finalizer_tasks={finalizer_task})
    shutdown = _shutdown(
        pool,
        stop_timeout_seconds=0.5,
        finalizer_drain_timeout_seconds=finalizer_drain_timeout_seconds,
    )
    loop = asyncio.get_running_loop()
    started_at = loop.time()

    await shutdown.aclose()

    elapsed = loop.time() - started_at
    assert finalizer_drain_timeout_seconds * 0.5 <= elapsed < 0.1
    assert cancellation_seen.is_set()
    assert finalizer_task.cancelled()
    assert pool.finalizer_tasks == set()
    timeout_warnings = [
        fields
        for level, _event, fields in pool.logger.events
        if level == "warning"
        and fields.get("timeout_seconds") == finalizer_drain_timeout_seconds
    ]
    assert timeout_warnings
