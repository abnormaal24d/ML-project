"""Crawler runtime lifecycle orchestrator."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)
from crawler.runtime.crawler_runtime_session import CrawlerRuntimeSession
from crawler.runtime.feedback.crawler_task_feedback import CrawlerTaskFeedback
from crawler.runtime.loop.crawl_run_summary import (
    CrawlRunResult,
    CrawlStopTrigger,
    CrawlTerminalOutcome,
    build_run_result,
)
from crawler.runtime.loop.crawl_run_supervisor import CrawlRunSupervisor
from crawler.worker.pool.worker_pool import WorkerPool
from logger.project_logger import ProjectLogger


class Crawler:
    """Pure runtime orchestrator with constructor injection only."""

    def __init__(
        self,
        *,
        enabled: bool,
        worker_pool: WorkerPool,
        logger: ProjectLogger,
        control_directory: CrawlerControlDirectory,
        task_feedback: CrawlerTaskFeedback,
        run_supervisor: CrawlRunSupervisor,
        build_runtime_session: Callable[[], CrawlerRuntimeSession],
    ) -> None:
        self._enabled = enabled
        self._worker_pool = worker_pool
        self._logger = logger
        self._control_directory = control_directory
        self._task_feedback = task_feedback
        self._run_supervisor = run_supervisor
        self._build_runtime_session = build_runtime_session

    async def crawl(self) -> CrawlRunResult:
        """Run one complete crawler lifecycle."""

        if not self._enabled:
            self._logger.info("crawler_disabled")
            return build_run_result(
                worker_snapshot=self._worker_pool.snapshot(
                    now=asyncio.get_running_loop().time()
                ),
                stop_trigger=CrawlStopTrigger.FAILED,
                terminal_outcome=CrawlTerminalOutcome.INCOMPLETE,
            )

        runtime_session = self._build_runtime_session()
        self._control_directory.ensure_exists()
        self._worker_pool.set_task_result_callback(self.on_task_processed)

        return await self._run_supervisor.run(
            runtime_session=runtime_session,
        )

    async def on_task_processed(
        self,
        *,
        task: CrawlTask,
        outcome: str,
        fields: dict[str, object] | None = None,
        result: object | None = None,
    ) -> None:
        await self._task_feedback.on_task_processed(
            task=task,
            outcome=outcome,
            fields=fields,
            result=result,
        )
