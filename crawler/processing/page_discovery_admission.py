"""Page discovery orchestration over selection, scheduler admission, and reporting."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from crawler.processing.page_discovery_reporting import (
    PageDiscoveryReporting,
)
from crawler.processing.page_discovery_scheduler_admission import (
    PageDiscoverySchedulerAdmission,
)
from crawler.processing.page_discovery_selector import PageDiscoverySelector
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.collection.processors import PageProcessorSettings
    from crawler.coverage.discovery_budget import PageDiscoveryCapResolver
    from crawler.coverage.snapshot import CoverageSnapshotProvider
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
    from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
    from shared.runtime_primitives import IdGenerator


class PageDiscoveryAdmission:
    """Coordinate page discovery without owning selection or reporting policy."""

    def __init__(
        self,
        *,
        settings: PageProcessorSettings,
        dataset_writer: DatasetWriter,
        logger: ProjectLogger,
        discovery_task_builder: DiscoveryTaskBuilder,
        url_filter: UrlAdmissionFilter,
        url_normalizer: UrlNormalizer,
        scheduler: UrlScheduler,
        cap_resolver: PageDiscoveryCapResolver,
        coverage_tracker: CoverageSnapshotProvider,
        focus_asset_boost: float,
        host_normalizer: HostNormalizer,
        id_generator: IdGenerator,
        rejected_discovery_reporter: Any | None = None,
    ) -> None:
        self._settings = settings
        self._dataset_writer = dataset_writer
        self._logger = logger
        self._scheduler = scheduler
        self._cap_resolver = cap_resolver
        self._coverage_tracker = coverage_tracker
        self._selector = PageDiscoverySelector(
            settings=settings,
            discovery_task_builder=discovery_task_builder,
            url_filter=url_filter,
            url_normalizer=url_normalizer,
            scheduler=scheduler,
            focus_asset_boost=focus_asset_boost,
            host_normalizer=host_normalizer,
        )
        self._scheduler_admission = PageDiscoverySchedulerAdmission(
            scheduler=scheduler,
            logger=logger,
            id_generator=id_generator,
        )
        self._reporting = PageDiscoveryReporting(
            logger=logger,
            host_normalizer=host_normalizer,
            rejected_discovery_reporter=rejected_discovery_reporter,
        )

    async def discover_and_enqueue(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: PageExtractionResult,
    ) -> dict[str, int]:
        """Discover, select, admit, persist, and report linked assets."""

        parent_url = result.final_url
        meta_robots = analysis.metadata.meta_robots
        if "nofollow" in meta_robots or "none" in meta_robots:
            self._logger.debug(
                "page_discovery_skipped",
                url=parent_url,
                reason="meta_robots_nofollow",
                meta_robots=list(meta_robots),
            )
            return self._empty_discovery_result()

        budget = await self._cap_resolver.resolve_budget(
            settings=self._settings,
            scheduler=self._scheduler,
            coverage_tracker=self._coverage_tracker,
            page_url=parent_url,
            source_name=task.source_name,
        )
        if budget.max_total <= 0:
            self._logger.debug(
                "page_discovery_skipped",
                url=parent_url,
                reason="empty_budget",
                pressure_state=budget.pressure_state,
                discovery_scan_budget=budget.discovery_scan_budget,
            )
            return self._empty_discovery_result(
                discovery_scan_budget=budget.discovery_scan_budget,
            )

        live_focus = tuple(
            str(kind).strip().lower()
            for kind, missing in getattr(
                budget,
                "coverage_missing_by_kind",
                {},
            ).items()
            if int(missing or 0) > 0
        )
        task_candidates = self._selector.extract_task_candidates(
            task=task,
            result=result,
            analysis=analysis,
            discovery_scan_budget=budget.discovery_scan_budget,
            focus_kinds=live_focus,
        )
        await self._dataset_writer.awrite_discovered_assets(
            parent_url=parent_url,
            tasks=task_candidates,
        )
        scope_eligibility = await self._selector.fetch_scope_eligibility(
            tasks=task_candidates,
        )
        selection = self._selector.select_tasks(
            task_candidates=task_candidates,
            budget=budget,
            scope_eligibility=scope_eligibility,
        )
        metrics = selection.metrics or {}
        rejected_by_reason, rejected_by_kind = (
            self._reporting.build_discovery_rejected_counters(metrics)
        )
        self._reporting.log_discovery_summary(
            parent_url=parent_url,
            selection=selection,
            metrics=metrics,
            rejected_by_reason=rejected_by_reason,
        )
        self._reporting.log_discovery_details(
            parent_url=parent_url,
            selection=selection,
            budget=budget,
            metrics=metrics,
            rejected_by_reason=rejected_by_reason,
            rejected_by_kind=rejected_by_kind,
        )

        (
            decisions,
            submitted_tasks,
        ) = await self._scheduler_admission.admit_selected_tasks(
            parent_url=parent_url,
            selected_tasks=selection.tasks,
        )
        rejected_assets = self._reporting.collect_rejected(
            parent_url=parent_url,
            selection=selection,
            decisions=list(decisions),
        )
        if rejected_assets:
            await self._dataset_writer.awrite_rejected_assets(
                parent_url=parent_url,
                rejected=tuple(rejected_assets),
            )

        admission = self._reporting.count_admission_decisions(
            tasks=selection.tasks,
            decisions=decisions,
        )
        merged_rejected_by_kind = Counter(rejected_by_kind)
        merged_rejected_by_kind.update(admission.rejected_by_kind)
        merged_rejected_by_reason = Counter(rejected_by_reason)
        merged_rejected_by_reason.update(admission.rejected_by_reason)
        self._reporting.log_admission_summary(
            parent_url=parent_url,
            selection=selection,
            budget=budget,
            submitted_count=len(submitted_tasks),
            admission=admission,
            discovery_rejected_by_kind=rejected_by_kind,
            discovery_rejected_by_reason=rejected_by_reason,
            merged_rejected_by_kind=merged_rejected_by_kind,
            merged_rejected_by_reason=merged_rejected_by_reason,
        )
        return self._reporting.build_result_metrics(
            selection=selection,
            budget=budget,
            selection_metrics=metrics,
            admission=admission,
        )

    @staticmethod
    def _empty_discovery_result(
        *,
        discovery_scan_budget: int = 0,
    ) -> dict[str, int]:
        return {
            "discovered": 0,
            "scheduled": 0,
            "filtered": 0,
            "rejected": 0,
            "scope_blocked": 0,
            "capacity_skipped": 0,
            "truncated": 0,
            "duplicates": 0,
            "discovery_scan_budget": discovery_scan_budget,
        }


__all__ = ["PageDiscoveryAdmission"]
