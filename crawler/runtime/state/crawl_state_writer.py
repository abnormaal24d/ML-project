"""Crawler runtime checkpoint writer."""

from __future__ import annotations

import asyncio

from config.settings.crawler import CrawlStateStoreSettings
from crawler.runtime.metrics.collection_metrics import CollectionMetrics
from crawler.runtime.state.crawl_checkpoint_store import CrawlerCheckpointStore
from crawler.scheduling.url_scheduler import UrlScheduler
from crawler.worker.pool.worker_pool import WorkerPool
from logger.project_logger import ProjectLogger


class CrawlStateWriter:
    """Persist crawler runtime checkpoint state."""

    def __init__(
        self,
        *,
        settings: CrawlStateStoreSettings,
        scheduler: UrlScheduler,
        worker_pool: WorkerPool,
        checkpoint_store: CrawlerCheckpointStore | None,
        metrics: CollectionMetrics | None,
        run_context: dict[str, object],
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._scheduler = scheduler
        self._worker_pool = worker_pool
        self._checkpoint_store = checkpoint_store
        self._metrics = metrics
        self._run_context = dict(run_context)
        self._logger = logger

    @property
    def enabled(self) -> bool:
        return (
            self._checkpoint_store is not None
            and self._checkpoint_store.enabled
        )

    def checkpoint_interval_seconds(self) -> float | None:
        if not self.enabled:
            return None
        return max(0.5, self._settings.checkpoint_interval_seconds)

    async def write_checkpoint(
        self,
        *,
        final: bool,
        max_queued_tasks: int | None = None,
    ) -> bool:
        """Persist a checkpoint and report whether it became durable."""

        if not self.enabled:
            return False
        checkpoint_store = self._checkpoint_store
        if checkpoint_store is None:
            return False

        try:
            scheduler_state = await self._scheduler.export_state(
                max_queued_tasks=(
                    self._settings.checkpoint_max_queued_tasks
                    if max_queued_tasks is None
                    else max_queued_tasks
                ),
                include_seen_urls=(
                    self._settings.include_seen_urls_in_checkpoint
                ),
            )
            worker_snapshot = self._worker_pool.snapshot(
                now=asyncio.get_running_loop().time()
            )
            await asyncio.wait_for(
                asyncio.to_thread(
                    checkpoint_store.write_runtime_checkpoint,
                    final=final,
                    scheduler_state=scheduler_state,
                    worker_snapshot=worker_snapshot,
                    metrics=self._metrics,
                    run_context=self._run_context,
                ),
                timeout=120.0,  # per audit P0 timeouts on IO
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._logger.warning(
                "crawler_checkpoint_write_failed_nonfatal",
                final=final,
                checkpoint_path=str(checkpoint_store.checkpoint_path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return False
        return True

    async def persist_on_completion(self) -> None:
        if not self.enabled:
            return
        checkpoint_store = self._checkpoint_store
        if checkpoint_store is None:
            return

        idle = await self._scheduler.is_idle()
        if idle:
            try:
                await asyncio.to_thread(checkpoint_store.clear_checkpoint)
            except OSError as exc:
                self._logger.warning(
                    "crawler_checkpoint_clear_failed_nonfatal",
                    checkpoint_path=str(checkpoint_store.checkpoint_path),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            return

        await self.write_checkpoint(final=True)
