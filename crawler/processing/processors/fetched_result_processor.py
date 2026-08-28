"""Fetched result processor schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from crawler.processing.outcomes.processor_outcome import ProcessorOutcome

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.fetching.results.result import FetchResult


class FetchedResultProcessor(ABC):
    """Abstract processor schema for already-fetched crawl results."""

    @abstractmethod
    async def process_fetched(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
    ) -> ProcessorOutcome:
        """Process one fetched crawl result."""
        raise NotImplementedError
