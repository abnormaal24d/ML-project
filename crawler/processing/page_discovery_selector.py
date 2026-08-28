"""Page-discovery candidate extraction and bounded selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from crawler.discovery.processing.page_discovery_selection import (
    PageDiscoverySelectionRequest,
    PageDiscoverySelectionResult,
    select_page_discovery_tasks,
)

if TYPE_CHECKING:
    from config.collection.processors import PageProcessorSettings
    from crawler.coverage.discovery_budget import PageDiscoveryBudget
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.discovery.discovery_task_builder import DiscoveryTaskBuilder
    from crawler.extraction.modalities.page_content_extractor import (
        PageExtractionResult,
    )
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.fetching.results.result import FetchResult
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.url_filter.url_admission_filter import (
        UrlAdmissionFilter,
    )
    from crawler.scheduling.url_scheduler import UrlScheduler


class PageDiscoverySelector:
    def __init__(
        self,
        *,
        settings: PageProcessorSettings,
        discovery_task_builder: DiscoveryTaskBuilder,
        url_filter: UrlAdmissionFilter,
        url_normalizer: UrlNormalizer,
        scheduler: UrlScheduler,
        focus_asset_boost: float,
        host_normalizer: HostNormalizer,
    ) -> None:
        self._discovery_task_builder = discovery_task_builder
        self._url_filter = url_filter
        self._url_normalizer = url_normalizer
        self._scheduler = scheduler
        self._focus_asset_boost = focus_asset_boost
        self._host_normalizer = host_normalizer
        self._discovery_ranking = settings.discovery_ranking

    def extract_task_candidates(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: PageExtractionResult,
        discovery_scan_budget: int,
        focus_kinds: tuple[str, ...] = (),
    ) -> tuple[CrawlTask, ...]:
        return self._discovery_task_builder.build_page_tasks(
            source_name=task.source_name,
            parent_url=result.final_url,
            parent_depth=task.depth,
            links=analysis.links,
            assets=analysis.asset_discovery,
            max_tasks=discovery_scan_budget,
            focus_kinds=focus_kinds,
            base_url=result.final_url,
        )

    async def fetch_scope_eligibility(
        self,
        *,
        tasks: tuple[CrawlTask, ...],
    ) -> Mapping[str, bool] | None:
        """Return the scheduler-owned crawl-scope preflight keyed by full discovery identity.

        Fail-open when the scheduler does not offer the read-only preflight;
        final scheduler admission re-checks crawl scope and stays authoritative.
        """

        from crawler.discovery.task_identity import discovered_task_identity

        preflight = getattr(self._scheduler, "discovery_scope_decisions", None)
        if preflight is None:
            return None
        decisions = await preflight(tasks)
        return {
            discovered_task_identity(
                task=task, normalized_url=str(decision.normalized_url)
            ): bool(decision.allowed)
            for task, decision in zip(tasks, decisions, strict=True)
        }

    def select_tasks(
        self,
        *,
        task_candidates: tuple[CrawlTask, ...],
        budget: PageDiscoveryBudget,
        scope_eligibility: Mapping[str, bool] | None = None,
    ) -> PageDiscoverySelectionResult:
        active_focus_kinds = tuple(
            str(kind).strip().lower()
            for kind, missing in budget.coverage_missing_by_kind.items()
            if int(missing) > 0
        )
        return select_page_discovery_tasks(
            request=PageDiscoverySelectionRequest(
                task_stream=task_candidates,
                max_total=budget.max_total,
                max_pages=budget.max_pages,
                max_embedded_assets=budget.max_embedded_assets,
                max_non_page_media=budget.max_non_page_media,
                kind_quotas=budget.kind_quotas,
                coverage_missing_by_kind=budget.coverage_missing_by_kind,
                active_focus_kinds=active_focus_kinds,
                focus_asset_boost=self._focus_asset_boost,
                ranking=self._discovery_ranking,
                url_filter=self._url_filter,
                url_normalizer=self._url_normalizer,
                already_seen=getattr(
                    self._scheduler,
                    "already_seen_for_discovery",
                    None,
                ),
                host_normalizer=self._host_normalizer,
                scheduler_capacity=budget.scheduler_capacity,
                scope_eligibility=scope_eligibility,
            ),
        )
