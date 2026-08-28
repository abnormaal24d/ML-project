"""Regression tests for direct worker-loop retirement."""

from __future__ import annotations

import asyncio

import pytest

from crawler.worker.task_iteration.worker_task_runner import IterationAction
from crawler.worker.worker_loop.worker_loop import WorkerLoop
from crawler.worker.worker_loop.worker_state import WorkerState


class _Logger:
    def debug(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None


class _BlockingTaskRunner:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def run_iteration(
        self,
        *,
        worker_id: int,
        state: WorkerState,
        fail_fast_on_processing_error: bool,
    ) -> IterationAction:
        del worker_id, fail_fast_on_processing_error
        state.busy = True
        self.started.set()
        await self.release.wait()
        state.busy = False
        self.finished.set()
        return IterationAction.CONTINUE


def _worker_loop() -> WorkerLoop:
    worker = object.__new__(WorkerLoop)
    worker.worker_id = 7
    worker.state = WorkerState(worker_id=worker.worker_id)
    worker.worker_task = None
    worker._logger = _Logger()  # type: ignore[assignment]
    return worker


@pytest.mark.asyncio
async def test_idle_worker_retirement_cancels_worker_task() -> None:
    worker = _worker_loop()
    started = asyncio.Event()

    async def wait_until_cancelled() -> None:
        started.set()
        await asyncio.Event().wait()

    worker.worker_task = asyncio.create_task(wait_until_cancelled())
    await started.wait()

    result = worker.retire()

    assert result is None
    assert worker.state.retire_when_idle is True
    with pytest.raises(asyncio.CancelledError):
        await worker.worker_task


@pytest.mark.asyncio
async def test_busy_worker_retirement_allows_iteration_to_finish() -> None:
    worker = _worker_loop()
    task_runner = _BlockingTaskRunner()
    worker._task_runner = task_runner  # type: ignore[assignment]
    worker.worker_task = asyncio.create_task(
        worker.run(fail_fast_on_processing_error=False)
    )
    await task_runner.started.wait()

    worker.retire()

    assert worker.state.retire_when_idle is True
    assert worker.worker_task.cancelled() is False

    task_runner.release.set()
    await worker.worker_task

    assert task_runner.finished.is_set()
    assert worker.worker_task.cancelled() is False
