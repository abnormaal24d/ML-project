"""Handle timed-out worker task failures."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.discovery import WorkerPoolSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.worker.worker_loop.worker_state import WorkerState


@dataclass(frozen=True, slots=True)
class RetryResolution:
    """Resolved runtime outcome for one timed-out processing failure."""

    runtime_outcome: Literal["deferred", "timeout"]
    error: BaseException
    should_stop: bool = False
    timeout_origin: str | None = None
    elapsed_seconds: float | None = None
    wait_seconds: float | None = None


def _elapsed_seconds_since(started_at: float | None) -> float | None:
    if started_at is None:
        return None
    return round(
        max(0.0, asyncio.get_running_loop().time() - started_at),
        3,
    )


class WorkerTaskTimeoutHandler:
    """Translate timeout failures into runtime outcomes."""

    def __init__(
        self,
        *,
        settings: WorkerPoolSettings,
        logger: ProjectLogger,
        register_failure: Callable[..., bool],
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._register_failure = register_failure

    def handle_timeout(
        self,
        *,
        exc: BaseException,
        worker_timeout_expired: bool,
        task: CrawlTask,
        worker_id: int,
        state: WorkerState,
        processing_started_at: float | None,
        fail_fast_on_processing_error: bool,
    ) -> RetryResolution:
        elapsed_seconds = _elapsed_seconds_since(processing_started_at)
        processing_timeout_seconds = float(
            self._settings.processing_timeout_seconds,
        )

        if worker_timeout_expired:
            state.record_outcome("timeout")
            should_stop = self._register_failure(
                worker_id=worker_id,
                task=task,
                cause=exc,
                fatal=fail_fast_on_processing_error,
            )
            self._logger.warning(
                "worker_processor_timeout",
                worker_id=worker_id,
                task_id=task.task_id,
                url=task.url,
                kind=task.kind,
                timeout_seconds=processing_timeout_seconds,
                configured_worker_timeout_seconds=processing_timeout_seconds,
                elapsed_seconds=elapsed_seconds,
                timeout_origin="worker_processing_timeout",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            return RetryResolution(
                runtime_outcome="timeout",
                error=exc,
                should_stop=should_stop,
                timeout_origin="worker_processing_timeout",
                elapsed_seconds=elapsed_seconds,
            )

        state.record_outcome("deferred")
        self._logger.warning(
            "worker_nested_async_timeout_deferred",
            worker_id=worker_id,
            task_id=task.task_id,
            url=task.url,
            kind=task.kind,
            configured_worker_timeout_seconds=processing_timeout_seconds,
            elapsed_seconds=elapsed_seconds,
            timeout_origin="processor_or_nested_async_timeout",
            retry_class="fetch_timeout",
            wait_seconds=1.0,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        return RetryResolution(
            runtime_outcome="deferred",
            error=exc,
            timeout_origin="processor_or_nested_async_timeout",
            elapsed_seconds=elapsed_seconds,
            wait_seconds=1.0,
        )
