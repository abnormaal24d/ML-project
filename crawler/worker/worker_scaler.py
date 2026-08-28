"""Worker scaling lifecycle and loop execution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.collection.autoscaling import AutoscalerSettings
    from crawler.scheduling.progress.scheduler_snapshot_reader import (
        SchedulerSnapshot,
    )
    from crawler.scheduling.url_scheduler import UrlScheduler
    from crawler.worker.pool.worker_pool import WorkerPool
    from crawler.worker.pool.worker_pool_snapshot import WorkerPoolSnapshot
    from logger.project_logger import ProjectLogger


class WorkerScaler:
    """Adjust effective worker count to match runnable scheduler capacity."""

    def __init__(
        self,
        *,
        settings: AutoscalerSettings,
        worker_pool: WorkerPool,
        scheduler: UrlScheduler,
        logger: ProjectLogger,
        create_task: Callable[..., asyncio.Task[None]] = asyncio.create_task,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._worker_pool = worker_pool
        self._scheduler = scheduler
        self._logger = logger
        self._create_task = create_task
        self._sleep = sleep

        self._task: asyncio.Task[None] | None = None
        self._failure_future: asyncio.Future[BaseException] | None = None

        self._scale_up_since: float | None = None
        self._scale_down_since: float | None = None

        self._last_resize_at: float | None = None

        self._previous_failure_count = 0

    @property
    def is_enabled(self) -> bool:
        return bool(self._settings.enabled)

    def start(self) -> None:
        if not self._settings.enabled:
            return

        task = self._task
        if task is not None and not task.done():
            return

        loop = asyncio.get_running_loop()

        if self._failure_future is None or self._failure_future.done():
            self._failure_future = loop.create_future()

        self._scale_up_since = None
        self._scale_down_since = None
        self._last_resize_at = None

        self._task = self._create_task(
            self._run(),
            name="worker-scaler",
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None

        self._scale_up_since = None
        self._scale_down_since = None
        self._last_resize_at = None

        if task is None or task.done():
            return

        task.cancel()

        try:
            await asyncio.wait_for(
                task,
                timeout=self._settings.cancel_timeout_seconds,
            )
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            self._logger.warning(
                "worker_scaler_stop_timeout",
            )

    async def aclose(self) -> None:
        await self.stop()

    async def wait_for_failure(self) -> None:
        failure_future = self._failure_future

        if failure_future is None:
            raise RuntimeError("worker scaler has not been started")

        failure = await asyncio.shield(failure_future)
        raise failure

    async def _run(self) -> None:
        try:
            while True:
                await self._scale_once()
                await self._sleep(self._settings.check_interval_seconds)

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            failure_future = self._failure_future

            if failure_future is not None and not failure_future.done():
                failure_future.set_result(exc)

            self._logger.exception(
                "worker_scaler_failed",
                error_type=type(exc).__name__,
                error_message=str(exc) or None,
            )

    async def _scale_once(self) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()

        worker = self._worker_pool.snapshot(now=now)
        scheduler = await self._scheduler.snapshot()

        failure_delta = max(
            0,
            worker.failure_count - self._previous_failure_count,
        )
        self._previous_failure_count = worker.failure_count

        current_workers = worker.effective_worker_count

        desired_workers = self._desired_workers(
            worker=worker,
            scheduler=scheduler,
        )

        if desired_workers > current_workers:
            self._scale_down_since = None

            if self._scale_up_since is None:
                self._scale_up_since = now

            sustained = (
                now - self._scale_up_since
                >= self._settings.scale_up_delay_seconds
            )

            cooldown_elapsed = (
                self._last_resize_at is None
                or now - self._last_resize_at
                >= self._settings.scale_up_cooldown_seconds
            )

            if (
                sustained
                and cooldown_elapsed
                and failure_delta <= self._settings.failure_burst_threshold
            ):
                target = min(
                    desired_workers,
                    current_workers + self._settings.max_scale_up_step,
                )

                await self._worker_pool.scale_to(target)

                self._last_resize_at = now
                self._scale_up_since = None

                self._logger.info(
                    "worker_count_increased",
                    previous_workers=current_workers,
                    target_workers=target,
                )

            return

        if desired_workers < current_workers:
            self._scale_up_since = None

            if self._scale_down_since is None:
                self._scale_down_since = now

            sustained = (
                now - self._scale_down_since
                >= self._settings.scale_down_delay_seconds
            )

            cooldown_elapsed = (
                self._last_resize_at is None
                or now - self._last_resize_at
                >= self._settings.scale_down_cooldown_seconds
            )

            safe_to_reduce = (
                failure_delta == 0
                and worker.longest_busy_seconds
                < self._settings.slow_task_seconds_threshold
            )

            if sustained and cooldown_elapsed and safe_to_reduce:
                target = max(
                    desired_workers,
                    current_workers - self._settings.max_scale_down_step,
                )

                await self._worker_pool.scale_to(target)

                self._last_resize_at = now
                self._scale_down_since = None

                self._logger.info(
                    "worker_count_decreased",
                    previous_workers=current_workers,
                    target_workers=target,
                )

            return

        self._scale_up_since = None
        self._scale_down_since = None

    def _desired_workers(
        self,
        *,
        worker: WorkerPoolSnapshot,
        scheduler: SchedulerSnapshot,
    ) -> int:
        settings = self._settings

        active_workers = max(
            settings.min_workers,
            worker.busy_worker_count,
        )

        runnable_workers = scheduler.inflight + scheduler.runnable_slots

        return min(
            settings.max_workers,
            max(
                active_workers,
                runnable_workers,
            ),
        )
