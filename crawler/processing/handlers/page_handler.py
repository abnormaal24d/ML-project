"""Page persisting processor."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from config.collection.processors import PageProcessorSettings
from config.environment.default_values import (
    ENRICHMENT_PREVIEW_MAX_CHARACTERS,
)
from crawler.extraction.modalities.page_content_extractor import (
    PageExtractionResult,
)
from crawler.processing.page_discovery_admission import PageDiscoveryAdmission
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.coverage.discovery_budget import PageDiscoveryCapResolver
    from crawler.coverage.snapshot import CoverageSnapshotProvider
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.discovery.discovery_task_builder import DiscoveryTaskBuilder
    from crawler.extraction.modalities.page_content_extractor import (
        PageContentExtractor,
    )
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.fetching.results.result import FetchResult
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.url_filter.url_admission_filter import (
        UrlAdmissionFilter,
    )
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )
    from crawler.scheduling.url_scheduler import UrlScheduler
    from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
    from shared.runtime_primitives import IdGenerator


class PageHandler(
    PersistingProcessor[PageProcessorSettings, PageExtractionResult]
):
    """Persisting processor for HTML page fetch results."""

    def __init__(
        self,
        *,
        settings: PageProcessorSettings,
        dataset_writer: DatasetWriter,
        logger: ProjectLogger,
        failure_handler: ProcessorFailureHandler,
        page_content_extractor: PageContentExtractor,
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
        super().__init__(
            settings=settings,
            dataset_writer=dataset_writer,
            logger=logger,
            failure_handler=failure_handler,
        )
        self._page_content_extractor = page_content_extractor
        self._scheduler = scheduler
        self._discovery_admission = PageDiscoveryAdmission(
            settings=settings,
            dataset_writer=dataset_writer,
            logger=logger,
            discovery_task_builder=discovery_task_builder,
            url_filter=url_filter,
            url_normalizer=url_normalizer,
            scheduler=scheduler,
            cap_resolver=cap_resolver,
            coverage_tracker=coverage_tracker,
            focus_asset_boost=focus_asset_boost,
            host_normalizer=host_normalizer,
            id_generator=id_generator,
            rejected_discovery_reporter=rejected_discovery_reporter,
        )
        self._canonical_url_lock = asyncio.Lock()

    async def prepare_analysis(
        self,
        *,
        result: FetchResult,
    ) -> PageExtractionResult:
        """Extract page content without DOM leakage."""
        return await asyncio.to_thread(
            self._page_content_extractor.extract,
            fetch_result=result,
            max_assets=self._settings.max_media_assets_per_page,
        )

    async def validate_result(
        self,
        *,
        result: FetchResult,
        analysis: PageExtractionResult | None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Validate extracted page quality before persistence."""

        if analysis is None:
            raise ValueError("Page extraction is required for validation")

        meta_robots = analysis.metadata.meta_robots
        if set(meta_robots).intersection({"none", "noindex"}):
            return (
                False,
                "meta_robots_blocked",
                {"meta_robots": list(meta_robots)},
            )

        return self._evaluate_quality(analysis=analysis)

    async def build_enrichment(
        self,
        *,
        result: FetchResult,
        analysis: PageExtractionResult | None,
    ) -> dict[str, object]:
        """Build persisted enrichment fields for the extracted page."""

        if analysis is None:
            raise ValueError("Page extraction is required for enrichment")

        return self._build_page_enrichment_fields(analysis=analysis)

    async def after_persist(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: PageExtractionResult | None,
    ) -> dict[str, int]:
        """Schedule links discovered from a persisted page."""
        if analysis is None:
            raise ValueError("Page extraction is required after persistence")

        await self._register_canonical_url(
            task=task,
            requested_url=result.final_url,
            canonical_url=analysis.metadata.canonical_url,
        )

        return await self._discovery_admission.discover_and_enqueue(
            task=task,
            result=result,
            analysis=analysis,
        )

    def _evaluate_quality(
        self,
        *,
        analysis: PageExtractionResult,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        char_count = analysis.text_content.char_count
        if char_count < self._settings.min_html_chars:
            return (
                False,
                "page_text_too_short",
                {
                    "page_char_count": char_count,
                    "quality_score": 0.2,
                },
            )
        return (
            True,
            None,
            {"page_char_count": char_count, "quality_score": 0.85},
        )

    @staticmethod
    def _build_page_enrichment_fields(
        *,
        analysis: PageExtractionResult,
    ) -> dict[str, object]:
        from crawler.storage.datasets.extraction.page_extraction_artifact import (
            build_page_extraction_artifact_from_analysis,
            enrichment_artifact_key,
        )

        text = analysis.text_content
        metadata = analysis.metadata
        artifact = build_page_extraction_artifact_from_analysis(
            analysis=analysis
        )
        return {
            "page_text_preview": text.text_preview[
                :ENRICHMENT_PREVIEW_MAX_CHARACTERS
            ],
            "page_char_count": text.char_count,
            "canonical_url": metadata.canonical_url,
            "meta_robots": list(metadata.meta_robots),
            "meta_refresh_url": metadata.meta_refresh_url,
            # Transient bulk payload consumed by DatasetWritePipeline.
            enrichment_artifact_key(): artifact.to_payload(),
        }

    async def _register_canonical_url(
        self,
        *,
        task: CrawlTask,
        requested_url: str,
        canonical_url: str | None,
    ) -> None:
        if not canonical_url:
            return

        async with self._canonical_url_lock:
            await self._scheduler.register_final_url(
                task=task,
                requested_url=requested_url,
                final_url=urljoin(requested_url, canonical_url),
            )
