"""Single worker loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from config.collection.discovery import WorkerPoolSettings
from crawler.worker.pool.worker_task_counters import WorkerTaskCounters
from crawler.worker.task_iteration.worker_task_runner import (
    IterationAction,
    WorkerTaskRunner,
)
from crawler.worker.worker_loop.worker_session_tracker import (
    WorkerSessionTracker,
)
from crawler.worker.worker_loop.worker_state import WorkerState
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.processing.task_processor import CrawlTaskProcessor
    from crawler.scheduling.url_scheduler import UrlScheduler


class WorkerLoop:
    """Run a single worker loop with explicit collaborators only."""

    def __init__(
        self,
        *,
        worker_id: int,
        settings: WorkerPoolSettings,
        scheduler: UrlScheduler,
        processor: CrawlTaskProcessor,
        logger: ProjectLogger,
        task_counters: WorkerTaskCounters,
        register_failure: Callable[..., bool],
        task_result_callback: Callable[..., Any] | None = None,
    ) -> None:
        self.worker_id: int = worker_id
        self.state: WorkerState = WorkerState(worker_id=worker_id)
        self.worker_task: asyncio.Task[None] | None = None
        self._settings = settings
        self._scheduler = scheduler
        self._processor = processor
        self._logger = logger
        self._register_failure = register_failure
        self._task_result_callback = task_result_callback

        session_tracker = WorkerSessionTracker(
            task_counters=task_counters,
            logger=logger,
        )
        self._task_runner = WorkerTaskRunner(
            settings=settings,
            scheduler=scheduler,
            processor=processor,
            logger=logger,
            session_tracker=session_tracker,
            register_failure=register_failure,
            task_result_callback=task_result_callback,
        )

    async def run(
        self,
        *,
        fail_fast_on_processing_error: bool,
    ) -> None:
        while True:
            state = self.state
            if state.retire_when_idle and not state.busy:
                self._logger.debug(
                    "worker_exit_retire_when_idle",
                    worker_id=self.worker_id,
                )
                return

            action = await self._task_runner.run_iteration(
                worker_id=self.worker_id,
                state=state,
                fail_fast_on_processing_error=fail_fast_on_processing_error,
            )

            if action is IterationAction.SCHEDULER_CLOSED:
                return

            if action is IterationAction.STOP:
                self._logger.warning(
                    "worker_exit_should_stop",
                    worker_id=self.worker_id,
                )
                return

    def retire(self) -> None:
        self.state.retire_when_idle = True

        if self.state.busy:
            return

        task = self.worker_task
        if task is not None and not task.done():
            task.cancel()
