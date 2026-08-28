"""Crawl seed planning and expansion.

This module owns the logic for building and expanding crawl seed tasks.
It is a pure domain component with no runtime execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.discovery.feed_alternate_resolver import (
    expand_seed_tasks_with_feed_alternates,
)
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.source_scope.source_scope_registry import (
    SourceScopeRegistry,
)
from shared.runtime_primitives import IdGenerator

if TYPE_CHECKING:
    from config.source_catalog.catalog_settings import SourceProfileSettings


@dataclass(frozen=True, slots=True)
class CrawlSeedPlan:
    """Immutable seed plan for a crawl run.

    Contains the fully expanded seed tasks ready for enqueueing.
    """

    tasks: tuple[CrawlTask, ...]
    primary_count: int


class CrawlSeedPlanBuilder:
    """Build a CrawlSeedPlan from configuration and governance services."""

    def __init__(
        self,
        *,
        seed_entries: "SourceProfileSettings",
        seed_source_type: str,
        feed_alternates_by_primary: dict[str, list[str]],
        url_normalizer: UrlNormalizer,
        host_normalizer: HostNormalizer,
        source_scope_registry: SourceScopeRegistry,
        id_generator: IdGenerator,
    ) -> None:
        self._seed_entries = seed_entries
        self._seed_source_type = seed_source_type
        self._feed_alternates_by_primary = feed_alternates_by_primary
        self._url_normalizer = url_normalizer
        self._host_normalizer = host_normalizer
        self._source_scope_registry = source_scope_registry
        self._id_generator = id_generator

    def build(self) -> CrawlSeedPlan:
        """Build the complete seed plan with feed alternates expanded."""
        # Build primary seed tasks
        seed_tasks = CrawlTask.build_seeds(
            seed_entries=self._seed_entries.seed_entries,
            seed_source_type=self._seed_source_type,
            id_generator=self._id_generator,
        )

        primary_seed_count = len(seed_tasks)

        # Expand with feed alternates
        expanded_seed_tasks = expand_seed_tasks_with_feed_alternates(
            seed_tasks=seed_tasks,
            alternates_by_primary=self._feed_alternates_by_primary,
            url_normalizer=self._url_normalizer,
            host_normalizer=self._host_normalizer,
            source_scope_registry=self._source_scope_registry,
            id_generator=self._id_generator,
        )

        return CrawlSeedPlan(
            tasks=tuple(expanded_seed_tasks),
            primary_count=primary_seed_count,
        )
