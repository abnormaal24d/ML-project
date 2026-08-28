"""Execution state runtime composition.

Builds state reader/writer, dead-letter handling, and seed enqueueing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings.root import Settings
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.runtime.loop.crawl_seed_enqueuer import CrawlerSeedEnqueuer
from crawler.runtime.metrics.collection_metrics import CollectionMetrics
from crawler.runtime.state.crawl_state_writer import CrawlStateWriter
from crawler.scheduling.url_scheduler import UrlScheduler
from crawler.worker.pool.worker_pool import WorkerPool
from logger.factory import ProjectLoggerFactory
from orchestration.composition.runtime.crawler_state import (
    CrawlerStatePersistence,
)

if TYPE_CHECKING:
    from logger.factory import ProjectLoggerFactory


def build_execution_state_runtime(
    *,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    scheduler: UrlScheduler,
    state: CrawlerStatePersistence,
    seed_tasks: tuple[CrawlTask, ...],
    worker_pool: WorkerPool,
    metrics: CollectionMetrics | None,
) -> tuple[CrawlStateWriter, CrawlerSeedEnqueuer]:
    """Build state reader/writer and seed enqueueing."""
    from crawler.runtime.loop.crawl_seed_enqueuer import CrawlerSeedEnqueuer
    from crawler.runtime.state.crawl_dead_letter_reader import (
        CrawlerDeadLetterReader,
    )
    from crawler.runtime.state.crawl_state_reader import CrawlStateReader
    from crawler.runtime.state.crawl_state_writer import CrawlStateWriter

    # Dead letter reader
    if state.dead_letter_path is not None:
        dead_letter_reader = CrawlerDeadLetterReader(
            settings=settings.crawler.state,
            dead_letter_path=state.dead_letter_path,
            logger=logger_factory.get_logger_for(CrawlerDeadLetterReader),
            task_deserializer=scheduler.create_checkpoint_task_deserializer(),
        )
    else:
        dead_letter_reader = None

    # Extract seed URLs
    seed_tasks_local = seed_tasks
    seed_urls = sorted(
        {
            task.url.strip()
            for task in seed_tasks_local
            if isinstance(task.url, str) and task.url.strip()
        }
    )
    checkpoint_run_context = {
        "seed_urls": seed_urls,
        "seed_count": len(seed_urls),
    }

    state_writer = CrawlStateWriter(
        settings=settings.crawler.state,
        scheduler=scheduler,
        worker_pool=worker_pool,
        checkpoint_store=state.checkpoint_store,
        metrics=metrics,
        run_context=checkpoint_run_context,
        logger=logger_factory.get_logger_for(CrawlStateWriter),
    )

    state_reader = CrawlStateReader(
        settings=settings.crawler.state,
        logger=logger_factory.get_logger_for(CrawlStateReader),
        scheduler=scheduler,
        checkpoint_store=state.checkpoint_store,
        dead_letter_reader=dead_letter_reader,
        checkpoint_writer=state_writer,
        current_seed_urls=tuple(seed_urls),
    )

    seed_enqueuer = CrawlerSeedEnqueuer(
        scheduler=scheduler,
        logger=logger_factory.get_logger_for(CrawlerSeedEnqueuer),
        seeds=seed_tasks,
        state_restorer=state_reader,
    )

    return state_writer, seed_enqueuer
