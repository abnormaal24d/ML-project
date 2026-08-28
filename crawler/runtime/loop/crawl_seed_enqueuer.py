"""Crawler seed restore and enqueue workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.scheduling.url_scheduler import UrlScheduler
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.runtime.state.crawl_state_reader import CrawlStateReader


@dataclass(frozen=True, slots=True)
class CrawlerSeedEnqueueResult:
    """Result of restoring and enqueueing initial crawler work."""

    total_seeds: int
    seed_source: str
    accepted_seeds: int
    rejected_seeds: int
    restored_tasks: int
    requeued_dead_letters: int


class CrawlerSeedEnqueuer:
    """Restore state and enqueue initial crawler seeds."""

    def __init__(
        self,
        *,
        scheduler: UrlScheduler,
        logger: ProjectLogger,
        seeds: Sequence[CrawlTask],
        state_restorer: CrawlStateReader,
    ) -> None:
        self._scheduler = scheduler
        self._logger = logger
        self._seeds = tuple(seeds)
        self._state_restorer = state_restorer

    async def prepare(self) -> CrawlerSeedEnqueueResult:
        restored_tasks = await self._state_restorer.restore_checkpoint()
        requeued_dead_letters = (
            await self._state_restorer.requeue_dead_letters()
        )

        accepted_seeds = 0
        rejected_seeds = 0

        decisions = await self._scheduler.enqueue_many(self._seeds)
        for decision in decisions:
            if not decision.accepted:
                rejected_seeds += 1
                self._logger.warning(
                    "seed_rejected",
                    url=decision.normalized_url,
                    reason=decision.reason,
                )
                continue

            accepted_seeds += 1

        return CrawlerSeedEnqueueResult(
            total_seeds=len(self._seeds),
            seed_source=_seed_source_label(self._seeds),
            accepted_seeds=accepted_seeds,
            rejected_seeds=rejected_seeds,
            restored_tasks=restored_tasks,
            requeued_dead_letters=requeued_dead_letters,
        )


def _seed_source_label(seeds: Sequence[CrawlTask]) -> str:
    source_types = {
        str(seed.source_type).strip()
        for seed in seeds
        if str(seed.source_type).strip()
    }

    if not source_types:
        return "unknown"

    if len(source_types) == 1:
        return next(iter(source_types))

    return "mixed"
