"""Feed persisting processor."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

from config.collection.processors import FeedProcessorSettings
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.discovery.processing.discovered_url_normalization import (
    dedupe_url_key,
    infer_discovered_kind,
)
from crawler.discovery.task_identity import discovered_task_identity_from_parts
from crawler.extraction.modalities.feed_content_extractor import (
    FeedContentExtractionResult,
)
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecisionReason,
)
from logger.project_logger import ProjectLogger
from shared.runtime_primitives import IdGenerator


@dataclass(frozen=True, slots=True)
class FeedDiscoverySelection:
    """Feed-discovered task selection summary."""

    tasks: list[CrawlTask]
    capped_count: int
    duplicate_count: int
    audio_count: int
    host_count: int


if TYPE_CHECKING:
    from crawler.extraction.modalities.feed_content_extractor import (
        FeedContentExtractor,
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


class FeedHandler(
    PersistingProcessor[FeedProcessorSettings, FeedContentExtractionResult]
):
    """Persisting processor for feed fetch results."""

    def __init__(
        self,
        *,
        settings: FeedProcessorSettings,
        url_filter: UrlAdmissionFilter | None,
        url_normalizer: UrlNormalizer | None,
        scheduler: UrlScheduler,
        dataset_writer: DatasetWriter,
        logger: ProjectLogger,
        failure_handler: ProcessorFailureHandler,
        feed_content_extractor: FeedContentExtractor,
        id_generator: IdGenerator,
        host_normalizer: HostNormalizer,
    ) -> None:
        if id_generator is None:
            raise ValueError("id_generator is required")

        super().__init__(
            settings=settings,
            dataset_writer=dataset_writer,
            logger=logger,
            failure_handler=failure_handler,
        )
        self._url_normalizer = url_normalizer
        self._url_filter = url_filter
        self._scheduler = scheduler
        self._id_generator = id_generator
        self._feed_content_extractor = feed_content_extractor
        self._host_normalizer = host_normalizer

    async def prepare_analysis(
        self,
        *,
        result: FetchResult,
    ) -> FeedContentExtractionResult:
        """Extract feed structure without analyzer-layer reparse."""
        return await asyncio.to_thread(
            self._feed_content_extractor.extract,
            fetch_result=result,
        )

    async def validate_result(
        self,
        *,
        result: FetchResult,
        analysis: FeedContentExtractionResult | None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Validate extracted feed quality before persistence."""

        if analysis is None:
            raise ValueError("Feed extraction is required for validation")

        return self._evaluate_quality(
            analysis=analysis,
        )

    async def build_enrichment(
        self,
        *,
        result: FetchResult,
        analysis: FeedContentExtractionResult | None,
    ) -> dict[str, object]:
        """Build persisted enrichment fields for the extracted feed."""

        if analysis is None:
            raise ValueError("Feed extraction is required for enrichment")

        return self._build_feed_enrichment_fields(analysis=analysis)

    async def after_persist(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        analysis: FeedContentExtractionResult | None,
    ) -> dict[str, int]:
        """Schedule links discovered from a persisted feed."""
        if analysis is None:
            raise ValueError("Feed extraction is required after persistence")

        if not bool(self._settings.schedule_entry_links):
            return {}

        entry_links = tuple(analysis.entry_links)
        media_enclosures = tuple(analysis.media_enclosures)
        queue_size = (await self._scheduler.snapshot()).total_queued
        total_limit, per_host_limit, audio_limit = self._discovery_limits(
            queue_size=queue_size,
        )
        if total_limit <= 0:
            return self._empty_discovery_result(
                capped_count=len(entry_links) + len(media_enclosures),
                media_enclosure_count=len(media_enclosures),
            )

        selection = await asyncio.wait_for(
            asyncio.to_thread(
                self._select_discovered_tasks,
                task=task,
                result=result,
                entry_links=entry_links,
                media_enclosures=media_enclosures,
                total_limit=total_limit,
                per_host_limit=per_host_limit,
                audio_limit=audio_limit,
            ),
            timeout=30.0,
        )
        filtered_tasks, local_filtered_count = self._filter_discovered_tasks(
            tasks=selection.tasks,
        )

        decisions = await self._scheduler.enqueue_many(filtered_tasks)

        accepted_count = sum(1 for decision in decisions if decision.accepted)
        scheduler_filtered_count = sum(
            1
            for decision in decisions
            if (
                not decision.accepted
                and decision.reason == ScheduleDecisionReason.URL_FILTERED
            )
        )
        scheduler_rejected_count = sum(
            1
            for decision in decisions
            if (
                not decision.accepted
                and decision.reason != ScheduleDecisionReason.URL_FILTERED
            )
        )

        return {
            "feed_entries": len(selection.tasks),
            "scheduled": accepted_count,
            "filtered": local_filtered_count + scheduler_filtered_count,
            "rejected": scheduler_rejected_count,
            "capped": selection.capped_count,
            "duplicates": selection.duplicate_count,
            "audio_entries": selection.audio_count,
            "hosts": selection.host_count,
            "media_enclosures": len(analysis.media_enclosure_links),
        }

    def _select_discovered_tasks(
        self,
        *,
        task: CrawlTask,
        result: FetchResult,
        entry_links: tuple[str, ...] | list[str],
        media_enclosures: tuple[dict[str, object], ...]
        | list[dict[str, object]] = (),
        total_limit: int | None = None,
        per_host_limit: int | None = None,
        audio_limit: int | None = None,
    ) -> FeedDiscoverySelection:
        if total_limit is None:
            total_limit = self._settings.max_feed_items_discovered
        if per_host_limit is None:
            per_host_limit = self._settings.max_discovered_links_per_host
        if audio_limit is None:
            audio_limit = self._settings.max_audio_links

        selected_urls: set[str] = set()
        selected_per_host: dict[str, int] = defaultdict(int)
        selected_audio_count = 0
        capped_count = 0
        duplicate_count = 0
        discovered_tasks: list[CrawlTask] = []

        for link in entry_links:
            absolute_url = urljoin(result.final_url, link)
            discovered_kind = infer_discovered_kind(url=absolute_url)
            normalized_url = self._dedupe_url_key(
                absolute_url,
                kind=discovered_kind,
                source_type="feed_item",
            )

            if normalized_url in selected_urls:
                duplicate_count += 1
                continue

            host = self._host_normalizer.normalize(
                urlparse(absolute_url).hostname
            )

            if total_limit and len(discovered_tasks) >= total_limit:
                capped_count += 1
                continue

            if (
                per_host_limit
                and host
                and selected_per_host[host] >= per_host_limit
            ):
                capped_count += 1
                continue

            if (
                discovered_kind == "audio"
                and audio_limit
                and selected_audio_count >= audio_limit
            ):
                capped_count += 1
                continue

            context: dict[str, object] = {}
            if discovered_kind not in {"page", "feed"}:
                context["source_page_depth"] = task.depth

            discovered_tasks.append(
                CrawlTask.build_discovered(
                    source_name=task.source_name,
                    url=absolute_url,
                    kind=discovered_kind,
                    parent_depth=task.depth,
                    source_type="feed_item",
                    parent_url=result.final_url,
                    context=context,
                    id_generator=self._id_generator,
                )
            )
            selected_urls.add(normalized_url)

            if host:
                selected_per_host[host] += 1

            if discovered_kind == "audio":
                selected_audio_count += 1

        for enclosure in media_enclosures:
            absolute_url = urljoin(
                result.final_url, str(enclosure.get("url") or "")
            )
            discovered_kind = str(
                enclosure.get("kind")
                or infer_discovered_kind(url=absolute_url)
            )
            normalized_url = self._dedupe_url_key(
                absolute_url,
                kind=discovered_kind,
                source_type="feed_enclosure",
            )

            if normalized_url in selected_urls:
                duplicate_count += 1
                continue

            host = self._host_normalizer.normalize(
                urlparse(absolute_url).hostname
            )

            if total_limit and len(discovered_tasks) >= total_limit:
                capped_count += 1
                continue

            if (
                per_host_limit
                and host
                and selected_per_host[host] >= per_host_limit
            ):
                capped_count += 1
                continue

            if (
                discovered_kind == "audio"
                and audio_limit
                and selected_audio_count >= audio_limit
            ):
                capped_count += 1
                continue

            context = {
                "source_tag": "enclosure",
                "source_attribute": "url",
                "mime_hint": enclosure.get("mime_type"),
                "asset_discovery_stage": "rss_enclosure",
                "asset_fetch_mode": "full_payload",
                "source_page_depth": task.depth,
            }
            discovered_tasks.append(
                CrawlTask.build_discovered(
                    source_name=task.source_name,
                    url=absolute_url,
                    kind=discovered_kind,
                    parent_depth=task.depth,
                    source_type="feed_enclosure",
                    parent_url=result.final_url,
                    context=context,
                    id_generator=self._id_generator,
                )
            )
            selected_urls.add(normalized_url)

            if host:
                selected_per_host[host] += 1

            if discovered_kind == "audio":
                selected_audio_count += 1

        return FeedDiscoverySelection(
            tasks=discovered_tasks,
            capped_count=capped_count,
            duplicate_count=duplicate_count,
            audio_count=selected_audio_count,
            host_count=len(selected_per_host),
        )

    def _filter_discovered_tasks(
        self,
        *,
        tasks: list[CrawlTask],
    ) -> tuple[list[CrawlTask], int]:
        if self._url_filter is None:
            return tasks, 0

        accepted_tasks: list[CrawlTask] = []
        filtered_count = 0

        for discovered_task in tasks:
            if self._url_filter.evaluate_task(discovered_task).allowed:
                accepted_tasks.append(discovered_task)
            else:
                filtered_count += 1

        return accepted_tasks, filtered_count

    def _dedupe_url_key(
        self,
        url: str,
        *,
        kind: str = "page",
        source_type: str = "feed_item",
    ) -> str:
        if self._url_normalizer is None:
            normalized_url = dedupe_url_key(url)
        else:
            normalized_url = self._url_normalizer.normalize(url)
        return discovered_task_identity_from_parts(
            url=normalized_url,
            kind=kind,
            source_type=source_type,
        )

    def _discovery_limits(self, *, queue_size: int) -> tuple[int, int, int]:
        total_limit = self._settings.max_feed_items_discovered
        per_host_limit = self._settings.max_discovered_links_per_host
        audio_limit = self._settings.max_audio_links

        high = self._settings.discovery_queue_high_watermark
        critical = self._settings.discovery_queue_critical_watermark
        if queue_size >= critical > 0:
            return (
                self._settings.max_feed_items_discovered_critical,
                0,
                0,
            )
        if queue_size >= high > 0:
            pressure_total = (
                self._settings.max_feed_items_discovered_under_pressure
            )
            return (
                min(total_limit, pressure_total),
                min(per_host_limit, max(1, pressure_total)),
                min(audio_limit, 1 if pressure_total > 0 else 0),
            )

        return total_limit, per_host_limit, audio_limit

    @staticmethod
    def _evaluate_quality(
        *,
        analysis: FeedContentExtractionResult,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        return (
            True,
            None,
            {
                "feed_entry_count": len(analysis.entry_links),
                "feed_media_enclosure_count": len(
                    analysis.media_enclosure_links
                ),
                "quality_score": (
                    0.8
                    if analysis.media_enclosure_links
                    else 0.75
                    if analysis.entry_links
                    else 0.5
                ),
            },
        )

    @staticmethod
    def _build_feed_enrichment_fields(
        *,
        analysis: FeedContentExtractionResult,
    ) -> dict[str, object]:
        return {
            "feed_title": analysis.title,
            "feed_entry_count": len(analysis.entry_links),
            "feed_media_enclosure_count": len(analysis.media_enclosure_links),
        }

    @staticmethod
    def _empty_discovery_result(
        *,
        capped_count: int,
        media_enclosure_count: int,
    ) -> dict[str, int]:
        return {
            "feed_entries": 0,
            "scheduled": 0,
            "filtered": 0,
            "rejected": 0,
            "capped": capped_count,
            "duplicates": 0,
            "audio_entries": 0,
            "hosts": 0,
            "media_enclosures": media_enclosure_count,
        }
