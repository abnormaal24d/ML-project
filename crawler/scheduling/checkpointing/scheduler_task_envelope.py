"""Scheduled task plus queue metadata used for checkpoint persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask


@dataclass(frozen=True, slots=True)
class SchedulerTaskEnvelope:
    """Scheduled task plus queue metadata used for checkpoint persistence."""

    task: CrawlTask
    host: str | None
    priority: int
    sequence: int
