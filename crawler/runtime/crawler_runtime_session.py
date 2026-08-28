"""Crawler runtime session container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.analysis.enrichment.lanes.analysis_router import (
        AnalysisRouter,
    )
    from crawler.runtime.actions.crawl_runtime_actions import (
        CrawlRuntimeActions,
    )
    from crawler.runtime.state.crawl_state_writer import CrawlStateWriter


@dataclass(frozen=True, slots=True)
class CrawlerRuntimeSession:
    """Concrete collaborators for one crawl loop run."""

    runtime_actions: CrawlRuntimeActions
    state_writer: CrawlStateWriter
    analysis_router: AnalysisRouter | None = None
