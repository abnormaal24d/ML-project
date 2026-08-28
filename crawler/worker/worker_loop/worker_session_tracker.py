"""Per-worker session state transitions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.worker.pool.worker_task_counters import (
        WorkerTaskCounters,
    )
    from crawler.worker.worker_loop.worker_state import WorkerState


class WorkerSessionTracker:
    """Track per-worker task start and finish state."""

    def __init__(
        self,
        *,
        task_counters: WorkerTaskCounters,
        logger: ProjectLogger,
    ) -> None:
        self._task_counters = task_counters
        self._logger = logger

    def mark_task_started(
        self,
        *,
        state: WorkerState,
        task: Any,
        worker_id: int,
    ) -> None:
        started_at = asyncio.get_running_loop().time()
        state.start_task(
            started_at=started_at,
            task_id=str(task.task_id),
            url=task.url,
            kind=task.kind,
        )
        self._logger.debug(
            "url_started",
            worker_id=worker_id,
            task_id=task.task_id,
            url=task.url,
            kind=task.kind,
            depth=task.depth,
            source=task.source_type,
            parent=task.parent_url,
        )

    def mark_task_finished(
        self,
        *,
        state: WorkerState,
        outcome: str | None,
        task: Any = None,
    ) -> float:
        if not state.busy:
            return 0.0

        finished_at = asyncio.get_running_loop().time()

        _total_seconds, processing_seconds = state.finish_task(
            finished_at=finished_at
        )

        duration = self._task_counters.record_task_completed(
            processing_seconds=processing_seconds,
            outcome=outcome,
        )

        if (
            task is not None
            and getattr(task, "depth", 1) == 0
            and outcome is not None
        ):
            self._task_counters.record_root_seed(outcome=outcome)

        return duration
