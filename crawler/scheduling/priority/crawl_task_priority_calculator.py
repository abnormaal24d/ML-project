"""Calculate crawl task priority from task and host signals.

Convention: lower numeric priority score = higher scheduling priority.
Boosts lower the score; penalties raise it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.discovery import UrlPrioritySettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.scheduling.host_control.host_budget_tracker import (
        HostBudgetTracker,
    )

_DEFAULT_SCORE = 0.2


class CrawlTaskPriorityCalculator:
    def __init__(
        self,
        config: UrlPrioritySettings,
        logger: ProjectLogger,
        host_extractor: HostExtractor,
        host_budget_tracker: HostBudgetTracker,
    ) -> None:
        self._config = config
        self._logger = logger
        self._host_extractor = host_extractor
        self._host_budget_tracker = host_budget_tracker

    def resolve(
        self,
        *,
        url: str,
        depth: int,
        source_type: str,
        kind: MediaKind = MediaKind.PAGE,
    ) -> int:
        # 1. Startpunt op basis van config
        priority = (
            self._config.seed_priority
            if source_type == "seed"
            else self._config.discovered_priority
        )

        # 2. Toepassen van Depth Penalty
        priority += depth * self._config.depth_penalty

        if kind is MediaKind.FEED:
            priority += {
                "high": -3,
                "medium": 0,
                "low": 3,
            }[self._config.feed_priority]

        host = self._host_extractor.extract(url)
        host_quality = self._host_quality(host)
        info_gain = self._host_info_gain(host)

        # 3. Bereken Boosts en Penalties (gebruikmakend van jouw 12.0 en 6.0
        # schalen)
        quality_boost = int(
            round(host_quality * self._config.host_quality_boost_scale)
        )
        info_gain_boost = int(
            round(info_gain * self._config.info_gain_boost_scale)
        )
        noise_penalty = int(
            round((1.0 - host_quality) * self._config.host_noise_penalty_scale)
        )

        # Berekening: Boosts verlagen het getal (hogere prioriteit), penalties
        # verhogen het.
        priority -= quality_boost + info_gain_boost
        priority += noise_penalty

        # 4. Externe host penalty
        if source_type != "seed" and host and self._host_budget_tracker:
            if not self._host_budget_tracker.is_seed_host(host):
                priority += self._config.external_host_exploration_penalty

        # 5. Deprioritize hosts with persistently low observed information gain.
        if (
            self._config.low_info_gain_penalty_enabled
            and info_gain < self._config.low_info_gain_penalty_threshold
        ):
            priority += self._config.low_info_gain_penalty

        # 6. Clamping
        final_priority = max(
            self._config.min_priority, min(priority, self._config.max_priority)
        )

        # 7. Schone logger die de 'misleidende' waarden uitlegt
        self._logger.debug(
            "priority_resolved",
            url=url,
            priority=final_priority,
            host=host,
            calc_details={
                "quality_boost": -quality_boost,
                "info_boost": -info_gain_boost,
                "noise_penalty": noise_penalty,
                "depth_impact": depth * self._config.depth_penalty,
                "kind": kind.value,
            },
        )
        return final_priority

    def resolve_task(self, task: CrawlTask) -> int:
        return self.resolve(
            url=task.url,
            depth=task.depth,
            source_type=task.source_type,
            kind=task.kind,
        )

    def __call__(self, task: CrawlTask) -> int:
        return self.resolve_task(task)

    def _host_quality(self, host: str | None) -> float:
        if not host or not self._host_budget_tracker:
            return _DEFAULT_SCORE
        return max(0.0, min(1.0, self._host_budget_tracker.host_quality(host)))

    def _host_info_gain(self, host: str | None) -> float:
        if not host or not self._host_budget_tracker:
            return _DEFAULT_SCORE
        return max(
            0.0, min(1.0, self._host_budget_tracker.host_info_gain(host))
        )
