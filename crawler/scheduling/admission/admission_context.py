"""Admission context for scheduler task admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask


@dataclass(frozen=True, slots=True)
class AdmissionContext:
    """Complete task-admission input for one scheduler enqueue decision."""

    task: CrawlTask
    host: str | None
    source: str | None
    now: float
    queue_size: int
    host_pending: int
    closed: bool
    kind_host_pending: int = 0
