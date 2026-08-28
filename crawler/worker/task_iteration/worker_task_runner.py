"""Execute one worker task iteration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import TYPE_CHECKING, Any

from crawler.exceptions.crawler_error import CrawlerTimeoutError
from crawler.processing.outcomes.processor_outcome import ProcessorOutcome
from crawler.worker.activity.worker_activity import (
    WorkerActivityController,
    bind_worker_activity,
)
from crawler.worker.task_iteration.worker_task_finalizer import (
    WorkerTaskFinalizer,
)
from crawler.worker.task_iteration.worker_task_result_persister import (
    WorkerTaskResultPersister,
)
from crawler.worker.task_iteration.worker_task_timeout_handler import (
    WorkerTaskTimeoutHandler,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from collections.abc import Callable

    from config.collection.discovery import WorkerPoolSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.processing.task_processor import CrawlTaskProcessor
    from crawler.scheduling.url_scheduler import UrlScheduler
    from crawler.worker.worker_loop.worker_session_tracker import (
        WorkerSessionTracker,
    )
    from crawler.worker.worker_loop.worker_state import WorkerState


class IterationAction(Enum):
    """Control-flow action returned to the enclosing worker loop."""

    CONTINUE = auto()
    STOP = auto()
    SCHEDULER_CLOSED = auto()


class RuntimeOutcome(StrEnum):
    """Exclusive non-processor outcome for one worker iteration."""

    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    DEFERRED = "deferred"
    TIMEOUT = "timeout"
    FAILED = "failed"


class _ProcessingTimeout(Exception):
    """Wrap a processor timeout with explicit outer-timeout provenance."""

    def __init__(
        self,
        *,
        cause: TimeoutError,
        worker_timeout_expired: bool,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.worker_timeout_expired = worker_timeout_expired


@dataclass(slots=True)
class _IterationState:
    """Runner-private mutable state for one scheduler task."""

    task: CrawlTask | None = None
    outcome: ProcessorOutcome | None = None
    runtime_outcome: RuntimeOutcome | None = None
    error: BaseException | None = None
    should_stop: bool = False
    processing_started_at: float | None = None
    timeout_elapsed_seconds: float | None = None
    timeout_origin: str | None = None
    wait_seconds: float | None = None


class WorkerTaskRunner:
    """Run one scheduler task through processing and finalization."""

    def __init__(
        self,
        *,
        settings: WorkerPoolSettings,
        scheduler: UrlScheduler,
        processor: CrawlTaskProcessor,
        logger: ProjectLogger,
        session_tracker: WorkerSessionTracker,
        register_failure: Callable[..., bool],
        task_result_callback: Callable[..., Any] | None = None,
    ) -> None:
        self._settings = settings
        self._scheduler = scheduler
        self._processor = processor
        self._logger = logger
        self._session_tracker = session_tracker
        self._register_failure = register_failure
        self._retry = WorkerTaskTimeoutHandler(
            settings=settings,
            logger=logger,
            register_failure=register_failure,
        )
        persister = WorkerTaskResultPersister(
            settings=settings,
            scheduler=scheduler,
            task_result_callback=task_result_callback,
        )
        self._finalizer = WorkerTaskFinalizer(
            settings=settings,
            logger=logger,
            persister=persister,
            session_tracker=session_tracker,
            register_failure=register_failure,
        )

    async def run_iteration(
        self,
        *,
        worker_id: int,
        state: WorkerState,
        fail_fast_on_processing_error: bool,
    ) -> IterationAction:
        from crawler.scheduling.url_scheduler import SchedulerClosedError

        iteration = _IterationState()

        try:
            iteration.task = await self._fetch_task(
                iteration=iteration,
                worker_id=worker_id,
                state=state,
            )
            iteration.outcome = await self._process_task(
                task=iteration.task,
                state=state,
                iteration=iteration,
            )
            state.record_outcome(iteration.outcome.status)

        except SchedulerClosedError:
            self._logger.debug(
                "worker_exit_scheduler_closed",
                worker_id=worker_id,
            )
            return IterationAction.SCHEDULER_CLOSED

        except KeyboardInterrupt:
            iteration.runtime_outcome = RuntimeOutcome.INTERRUPTED
            state.record_outcome(RuntimeOutcome.INTERRUPTED.value)
            raise

        except asyncio.CancelledError:
            iteration.runtime_outcome = RuntimeOutcome.CANCELLED
            state.record_outcome(RuntimeOutcome.CANCELLED.value)
            raise

        except _ProcessingTimeout as timeout:
            task = _required_task(iteration.task)
            resolution = self._retry.handle_timeout(
                exc=timeout.cause,
                worker_timeout_expired=timeout.worker_timeout_expired,
                task=task,
                worker_id=worker_id,
                state=state,
                processing_started_at=iteration.processing_started_at,
                fail_fast_on_processing_error=(fail_fast_on_processing_error),
            )
            _apply_retry_resolution(
                iteration=iteration,
                runtime_outcome=(
                    RuntimeOutcome.TIMEOUT
                    if resolution.runtime_outcome == "timeout"
                    else RuntimeOutcome.DEFERRED
                ),
                error=resolution.error,
                should_stop=resolution.should_stop,
                timeout_origin=resolution.timeout_origin,
                elapsed_seconds=resolution.elapsed_seconds,
                wait_seconds=resolution.wait_seconds,
            )

        except (RuntimeError, OSError, ValueError) as exc:
            iteration.error = exc
            iteration.runtime_outcome = RuntimeOutcome.FAILED
            state.record_outcome(RuntimeOutcome.FAILED.value)
            iteration.should_stop = self._register_failure(
                worker_id=worker_id,
                task=iteration.task,
                cause=exc,
                fatal=fail_fast_on_processing_error,
            )
            if not iteration.should_stop:
                await asyncio.sleep(self._settings.empty_backoff_seconds)

        finally:
            if iteration.task is not None:
                await self._finalizer.finalize(
                    task=iteration.task,
                    outcome=iteration.outcome,
                    runtime_outcome=(
                        iteration.runtime_outcome.value
                        if iteration.runtime_outcome is not None
                        else None
                    ),
                    error=iteration.error,
                    timeout_origin=iteration.timeout_origin,
                    timeout_elapsed_seconds=(
                        iteration.timeout_elapsed_seconds
                    ),
                    wait_seconds=iteration.wait_seconds,
                    worker_id=worker_id,
                    state=state,
                )

        return (
            IterationAction.STOP
            if iteration.should_stop
            else IterationAction.CONTINUE
        )

    async def _fetch_task(
        self,
        *,
        iteration: _IterationState,
        worker_id: int,
        state: WorkerState,
    ) -> CrawlTask:
        task = await self._scheduler.get()

        # Persist ownership immediately after scheduler dispatch. If session
        # state tracking or logging fails, finalization can still release it.
        iteration.task = task

        self._session_tracker.mark_task_started(
            state=state,
            task=task,
            worker_id=worker_id,
        )
        return task

    async def _process_task(
        self,
        *,
        task: CrawlTask,
        state: WorkerState,
        iteration: _IterationState,
    ) -> ProcessorOutcome:
        loop = asyncio.get_running_loop()
        activity = WorkerActivityController(
            state=state,
            clock=loop.time,
        )
        iteration.processing_started_at = loop.time()

        timeout_context = asyncio.timeout(
            float(self._settings.processing_timeout_seconds)
        )
        try:
            async with timeout_context:
                with bind_worker_activity(activity):
                    return await self._processor.process(task)
        except TimeoutError as exc:
            if isinstance(exc, CrawlerTimeoutError):
                raise
            raise _ProcessingTimeout(
                cause=exc,
                worker_timeout_expired=timeout_context.expired(),
            ) from exc


def _required_task(task: CrawlTask | None) -> CrawlTask:
    if task is None:
        raise RuntimeError(
            "worker iteration failure occurred before a task was retained"
        )
    return task


def _apply_retry_resolution(
    *,
    iteration: _IterationState,
    runtime_outcome: RuntimeOutcome,
    error: BaseException,
    should_stop: bool,
    timeout_origin: str | None,
    elapsed_seconds: float | None,
    wait_seconds: float | None,
) -> None:
    iteration.runtime_outcome = runtime_outcome
    iteration.error = error
    iteration.should_stop = should_stop
    iteration.timeout_origin = timeout_origin
    iteration.timeout_elapsed_seconds = elapsed_seconds
    iteration.wait_seconds = wait_seconds
