"""Worker pool lifecycle and scaling coordination."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from itertools import count
from typing import TYPE_CHECKING, Any

from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.exceptions.crawler_error import (
    WorkerPoolError,
    WorkerPoolFailedError,
)
from crawler.fetching.errors.exceptions import (
    IgnoredFetchError,
    RetryableFetchError,
)
from crawler.worker.pool.worker_pool_snapshot import (
    WorkerPoolSnapshot,
    build_worker_pool_snapshot,
)
from crawler.worker.worker_loop.worker_loop import WorkerLoop
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.discovery import WorkerPoolSettings
    from crawler.worker.pool.worker_task_counters import WorkerTaskCounters


class WorkerPool:
    """Manage the lifecycle and scaling of crawl worker loops."""

    def __init__(
        self,
        *,
        settings: WorkerPoolSettings,
        logger: ProjectLogger,
        task_counters: WorkerTaskCounters,
        worker_runtime_factory: Callable[..., WorkerLoop],
        failure_event: asyncio.Event,
    ) -> None:
        self._settings = settings
        self.logger = logger
        self._task_counters = task_counters
        self.worker_runtime_factory = worker_runtime_factory
        self.failure_event = failure_event

        self.lock = asyncio.Lock()
        self.workers: dict[int, WorkerLoop] = {}
        self.finalizer_tasks: set[asyncio.Task[None]] = set()
        self.last_failure: BaseException | None = None
        self.closed = False
        self.task_result_callback: Callable[..., Any] | None = None
        self._worker_counter = count(1)

        self._failure_recorder = WorkerFailureRecorder(
            self,
            task_counters=task_counters,
        )
        self._shutdown = WorkerPoolShutdown(
            self,
            failure_recorder=self._failure_recorder,
            stop_timeout_seconds=settings.stop_timeout_seconds,
            finalizer_drain_timeout_seconds=(
                settings.finalizer_drain_timeout_seconds
            ),
        )

    def set_task_result_callback(
        self, callback: Callable[..., Any] | None
    ) -> None:
        self.task_result_callback = callback

    @property
    def fail_fast_on_processing_error(self) -> bool:
        return bool(self._settings.fail_fast_on_processing_error)

    def snapshot(self, *, now: float) -> WorkerPoolSnapshot:
        """Return one consistent observation of live state and counters."""

        return build_worker_pool_snapshot(
            workers=self.workers,
            task_counters=self._task_counters,
            now=now,
        )

    async def scale_to(self, target: int) -> None:
        normalized_target = max(0, int(target))

        async with self.lock:
            self.raise_if_closed()

            current_effective = self.effective_size
            if normalized_target == current_effective:
                return

            if normalized_target > current_effective:
                self._scale_up_locked(target=normalized_target)
            else:
                self._scale_down_locked(target=normalized_target)

    def raise_if_closed(self) -> None:
        if self.closed:
            raise WorkerPoolError("worker pool is closed")

    def _scale_up_locked(self, *, target: int) -> None:
        while self.effective_size < target:
            reusable_worker_id = self._select_retiring_worker_for_reuse()
            if reusable_worker_id is not None:
                worker = self.workers.get(reusable_worker_id)
                if worker is not None:
                    worker.state.retire_when_idle = False
                continue

            self._start_worker_locked()

    def _scale_down_locked(self, *, target: int) -> None:
        while self.effective_size > target:
            worker_id = self._select_worker_for_retirement()
            if worker_id is None:
                break

            worker = self.workers.get(worker_id)
            if worker is None:
                continue

            worker.retire()

    @property
    def effective_size(self) -> int:
        """Return lifecycle capacity without constructing a report snapshot."""

        return sum(
            1
            for worker in self.workers.values()
            if not worker.state.retire_when_idle
        )

    def _select_worker_for_retirement(self) -> int | None:
        """Retire idle workers first, busy workers only as a last resort.

        Idle-first retirement turns a plain scale-down into an immediate
        cancel/exit instead of marking a busy worker retiring while its
        capacity still exists on the physical pool.
        """

        busy_candidate: int | None = None
        for worker_id in sorted(self.workers):
            state = self.workers[worker_id].state
            if state.retire_when_idle:
                continue
            if not state.busy:
                return worker_id
            if busy_candidate is None:
                busy_candidate = worker_id
        return busy_candidate

    def _select_retiring_worker_for_reuse(self) -> int | None:
        """Return a busy retiring worker whose retirement can be cancelled.

        A busy worker that is retiring still has real capacity: it runs its
        current task and would exit afterwards. Reactivating it avoids
        starting a new worker while the retiring one is still alive.
        """

        for worker_id in sorted(self.workers):
            state = self.workers[worker_id].state
            if not state.busy or not state.retire_when_idle:
                continue
            return worker_id
        return None

    def _start_worker_locked(self) -> None:
        """Construct a worker loop and dispatch its crawl task inside the lock."""

        worker_id = next(self._worker_counter)
        runtime = self.worker_runtime_factory(
            worker_id=worker_id,
            register_failure=self._failure_recorder.record_failure,
            task_result_callback=self.task_result_callback,
        )
        self.workers[worker_id] = runtime

        task = asyncio.create_task(
            runtime.run(
                fail_fast_on_processing_error=self.fail_fast_on_processing_error,
            ),
            name=f"crawler-worker-{worker_id}",
        )
        runtime.worker_task = task

        def _handle_task_done(completed_task: asyncio.Task[None]) -> None:
            self._shutdown.on_worker_task_done(
                worker_id=worker_id,
                task=completed_task,
            )

        task.add_done_callback(_handle_task_done)

    async def wait_for_failure(self) -> None:
        await self.failure_event.wait()

    def raise_if_failed(self) -> None:
        if self.last_failure is None:
            return

        raise WorkerPoolFailedError(
            "worker pool recorded a fatal processing failure"
        ) from self.last_failure

    async def aclose(self) -> None:
        await self._shutdown.aclose()


class WorkerFailureRecorder:
    """Classify failures, update counters, and set fatal pool events."""

    def __init__(
        self,
        pool: WorkerPool,
        *,
        task_counters: WorkerTaskCounters,
    ) -> None:
        self._pool = pool
        self._task_counters = task_counters

    def record_failure(
        self,
        *,
        worker_id: int,
        task: CrawlTask | None,
        cause: BaseException,
        fatal: bool,
    ) -> bool:
        if isinstance(cause, asyncio.CancelledError):
            self._pool.logger.debug(
                "worker_cancellation_not_recorded_as_failure",
                worker_id=worker_id,
                task_id=str(task.task_id) if task is not None else None,
                kind=None if task is None else task.kind,
                url=None if task is None else task.url,
            )
            return False

        if isinstance(cause, RetryableFetchError):
            # Retry exhaustion is non-fatal worker control flow, but remains a
            # cumulative task outcome that must be visible in pool snapshots.
            self._task_counters.register_failure(
                cause=cause,
                fatal=False,
            )

        if isinstance(cause, (IgnoredFetchError, RetryableFetchError)):
            self._pool.logger.warning(
                "worker_control_flow_error_ignored",
                worker_id=worker_id,
                task_id=str(task.task_id) if task is not None else None,
                kind=None if task is None else task.kind,
                url=None if task is None else task.url,
                error_type=type(cause).__name__,
                error=str(cause),
            )
            return False

        if fatal:
            # Only a fatal task failure can surface again as the same worker
            # runtime failure. Non-fatal task failures must not suppress a
            # later, unrelated runtime-crash observation for this worker.
            worker = self._pool.workers.get(worker_id)
            if worker is not None:
                worker.state.fatal_failure_recorded = True
        if isinstance(cause, (asyncio.TimeoutError, TimeoutError)):
            recorded_failure_class = (
                "timeout_fatal" if fatal else "timeout_nonfatal"
            )
        elif isinstance(cause, RetryableFetchError):
            recorded_failure_class = "retry_exhausted"
        else:
            recorded_failure_class = "fatal" if fatal else "nonfatal"

        self._task_counters.register_failure(
            cause=cause,
            fatal=fatal,
        )

        self._pool.logger.error(
            "worker_failure_recorded",
            worker_id=worker_id,
            task_id=str(task.task_id) if task is not None else None,
            kind=None if task is None else task.kind,
            url=None if task is None else task.url,
            error_type=type(cause).__name__,
            error=str(cause),
            fatal=fatal,
            failure_scope="task",
            failure_class=recorded_failure_class,
            non_fatal_timeout=(recorded_failure_class == "timeout_nonfatal"),
        )

        if fatal:
            self._pool.last_failure = cause
            self._pool.failure_event.set()

        return bool(fatal)

    def record_runtime_failure(
        self,
        *,
        worker_id: int,
        cause: BaseException,
    ) -> None:
        if isinstance(cause, asyncio.CancelledError):
            self._pool.logger.debug(
                "worker_runtime_cancellation_not_recorded_as_failure",
                worker_id=worker_id,
            )
            return

        worker = self._pool.workers.get(worker_id)
        state = None if worker is None else worker.state
        if state is None or not state.fatal_failure_recorded:
            if state is not None:
                state.fatal_failure_recorded = True
            self._task_counters.register_failure(
                cause=cause,
                fatal=True,
            )

        if self._pool.last_failure is None:
            self._pool.last_failure = cause

        self._pool.failure_event.set()
        self._pool.logger.error(
            "worker_runtime_failed",
            worker_id=worker_id,
            error_type=type(cause).__name__,
            error=str(cause),
        )


class WorkerPoolShutdown:
    """Manage pool close, worker stop, and task finalization."""

    def __init__(
        self,
        pool: WorkerPool,
        *,
        failure_recorder: WorkerFailureRecorder,
        stop_timeout_seconds: float,
        finalizer_drain_timeout_seconds: float,
    ) -> None:
        self._pool = pool
        self._failure_recorder = failure_recorder
        self._stop_timeout_seconds = stop_timeout_seconds
        self._finalizer_drain_timeout_seconds = finalizer_drain_timeout_seconds

    async def aclose(self) -> None:
        async with self._pool.lock:
            if self._pool.closed:
                return

            self._pool.closed = True
            workers = list(self._pool.workers.items())

        worker_tasks_by_id = {
            worker_id: worker.worker_task
            for worker_id, worker in workers
            if worker.worker_task is not None and not worker.worker_task.done()
        }
        worker_tasks = set(worker_tasks_by_id.values())

        for task in worker_tasks:
            task.cancel()

        if worker_tasks:
            _, pending = await asyncio.wait(
                worker_tasks,
                timeout=self._stop_timeout_seconds,
            )
        else:
            pending = set()

        timed_out_worker_ids = tuple(
            sorted(
                worker_id
                for worker_id, task in worker_tasks_by_id.items()
                if task in pending
            )
        )

        for task in pending:
            task.cancel()

        await asyncio.sleep(0)
        await self._drain_finalizer_tasks()

        async with self._pool.lock:
            live_worker_ids = tuple(
                worker_id
                for worker_id, worker in self._pool.workers.items()
                if worker.worker_task is not None
                and not worker.worker_task.done()
            )

        if timed_out_worker_ids:
            self._pool.logger.error(
                "worker_pool_worker_stop_timeout",
                timeout_seconds=self._stop_timeout_seconds,
                timed_out_worker_ids=timed_out_worker_ids,
                live_worker_ids=live_worker_ids,
            )
            raise RuntimeError(
                "worker pool workers did not stop within "
                f"{self._stop_timeout_seconds} seconds; timed out worker IDs: "
                + ", ".join(
                    str(worker_id) for worker_id in timed_out_worker_ids
                )
                + "; still live worker IDs: "
                + (
                    ", ".join(str(worker_id) for worker_id in live_worker_ids)
                    if live_worker_ids
                    else "none"
                )
            )

        if live_worker_ids:
            self._pool.logger.error(
                "worker_pool_close_left_live_workers",
                worker_ids=live_worker_ids,
            )
            raise RuntimeError(
                "worker pool shutdown completed with live worker tasks: "
                + ", ".join(str(worker_id) for worker_id in live_worker_ids)
            )

        self._pool.logger.debug("worker_pool_closed", workers=0)

    async def _drain_finalizer_tasks(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._finalizer_drain_timeout_seconds

        while self._pool.finalizer_tasks:
            finalizer_tasks = tuple(self._pool.finalizer_tasks)
            remaining = max(0.0, deadline - loop.time())

            if remaining > 0.0:
                done, pending = await asyncio.wait(
                    finalizer_tasks,
                    timeout=remaining,
                )
            else:
                done = {task for task in finalizer_tasks if task.done()}
                pending = {task for task in finalizer_tasks if not task.done()}

            for finalizer_task in done:
                self._pool.finalizer_tasks.discard(finalizer_task)

            if pending:
                for finalizer_task in pending:
                    finalizer_task.cancel()
                    self._pool.logger.warning(
                        "worker_finalizer_cancelled_after_timeout",
                        timeout_seconds=self._finalizer_drain_timeout_seconds,
                    )

                # Give cancellation one event-loop turn, without granting a
                # second finalizer timeout budget to cancellation-resistant
                # tasks.
                await asyncio.sleep(0)
                for finalizer_task in pending:
                    if finalizer_task.done():
                        self._pool.finalizer_tasks.discard(finalizer_task)

                live_finalizer_count = sum(
                    not task.done() for task in self._pool.finalizer_tasks
                )
                if live_finalizer_count:
                    self._pool.logger.error(
                        "worker_finalizer_drain_left_live_tasks",
                        timeout_seconds=self._finalizer_drain_timeout_seconds,
                        live_task_count=live_finalizer_count,
                    )
                return

            # Let done callbacks discard/log completed finalizers and catch
            # finalizers that were scheduled while this batch was draining.
            await asyncio.sleep(0)

    def on_worker_task_done(
        self,
        *,
        worker_id: int,
        task: asyncio.Task[None],
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_closed():
                self._pool.logger.error(
                    "worker_finalizer_not_started",
                    worker_id=worker_id,
                    reason="event_loop_closed",
                )
                return

            finalizer_task = asyncio.create_task(
                self.finalize_worker_task(worker_id=worker_id, task=task)
            )
            self._pool.finalizer_tasks.add(finalizer_task)
            finalizer_task.add_done_callback(self._on_finalizer_task_done)
        except RuntimeError as exc:
            self._pool.logger.error(
                "worker_finalizer_not_started",
                worker_id=worker_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def _on_finalizer_task_done(self, task: asyncio.Task[None]) -> None:
        self._pool.finalizer_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            self._pool.logger.warning("worker_finalize_task_cancelled")
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self._pool.logger.error(
                "worker_finalize_task_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )

    async def finalize_worker_task(
        self,
        *,
        worker_id: int,
        task: asyncio.Task[None],
    ) -> None:
        async with self._pool.lock:
            worker = self._pool.workers.get(worker_id)
            state = None if worker is None else worker.state
            was_retiring = bool(state is not None and state.retire_when_idle)
            was_shutdown = self._pool.closed
            current_task_id = None if state is None else state.current_task_id
            current_url = None if state is None else state.current_url
            current_kind = None if state is None else state.current_kind

        try:
            await task
        except asyncio.CancelledError:
            self._pool.logger.info(
                "worker_runtime_cancelled",
                worker_id=worker_id,
                reason=(
                    "shutdown"
                    if was_shutdown
                    else (
                        "scale_down_retiring" if was_retiring else "cancelled"
                    )
                ),
                task_id=current_task_id,
                kind=current_kind,
                url=current_url,
            )
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            self._failure_recorder.record_runtime_failure(
                worker_id=worker_id,
                cause=error,
            )
        finally:
            async with self._pool.lock:
                self._pool.workers.pop(worker_id, None)
