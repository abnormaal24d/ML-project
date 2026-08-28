"""Worker pool and scaler composition."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from config.settings.root import Settings
from crawler.processing.task_processor import CrawlTaskProcessor
from crawler.scheduling.url_scheduler import UrlScheduler
from logger.factory import ProjectLoggerFactory


def build_worker_runtime(
    *,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    scheduler: UrlScheduler,
    task_processor: CrawlTaskProcessor,
) -> tuple[WorkerPool, WorkerScaler]:
    """Build the worker pool and autoscaler that execute scheduled tasks."""
    from crawler.worker.pool.worker_pool import WorkerPool
    from crawler.worker.pool.worker_task_counters import WorkerTaskCounters
    from crawler.worker.worker_scaler import WorkerScaler

    worker_pool_settings = settings.collection.worker_pool
    worker_task_counters = WorkerTaskCounters()
    worker_failure_event = asyncio.Event()

    def worker_runtime_factory(
        *,
        worker_id: int,
        register_failure: Callable[..., bool],
        task_result_callback: Callable[..., object] | None,
    ) -> WorkerLoop:
        from crawler.worker.worker_loop.worker_loop import WorkerLoop

        return WorkerLoop(
            worker_id=worker_id,
            settings=worker_pool_settings,
            scheduler=scheduler,
            processor=task_processor,
            logger=logger_factory.get_logger_for(WorkerLoop),
            task_counters=worker_task_counters,
            register_failure=register_failure,
            task_result_callback=task_result_callback,
        )

    worker_pool = WorkerPool(
        settings=worker_pool_settings,
        logger=logger_factory.get_logger_for(WorkerPool),
        task_counters=worker_task_counters,
        worker_runtime_factory=worker_runtime_factory,
        failure_event=worker_failure_event,
    )
    worker_scaler = WorkerScaler(
        settings=settings.collection.autoscaler,
        worker_pool=worker_pool,
        scheduler=scheduler,
        logger=logger_factory.get_logger_for(WorkerScaler),
    )
    return worker_pool, worker_scaler
