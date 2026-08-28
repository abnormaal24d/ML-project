"""Typed contracts for the crawler execution subgraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from config.collection.processors import PageProcessorSettings
from crawler.governance.processing_activity import ProcessingActivityRegistry
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)
from crawler.runtime.crawler_runtime_session import CrawlerRuntimeSession
from crawler.runtime.feedback.crawler_task_feedback import CrawlerTaskFeedback
from crawler.runtime.loop.crawl_seed_enqueuer import CrawlerSeedEnqueuer
from crawler.scheduling.url_scheduler import UrlScheduler
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from crawler.worker.pool.worker_pool import WorkerPool
from crawler.worker.worker_scaler import WorkerScaler

if TYPE_CHECKING:
    from datachecker.manifests.crawl_state_manifest_writer import (
        CrawlStateManifestWriter,
    )


@dataclass(frozen=True, slots=True)
class CrawlerExecutionServices:
    """Execution services consumed by the crawler graph assembly.

    Contains only services that downstream assembly actually reads; the
    runtime state reader/writer pair stays internal to this subgraph.
    """

    scheduler: UrlScheduler
    worker_pool: WorkerPool
    worker_scaler: WorkerScaler
    dataset_writer: DatasetWriter
    seed_enqueuer: CrawlerSeedEnqueuer
    build_runtime_session: Callable[[], CrawlerRuntimeSession]
    control_directory: CrawlerControlDirectory
    task_feedback: CrawlerTaskFeedback


@dataclass(frozen=True, slots=True)
class CrawlerExecutionOverrides:
    """Optional overrides for execution subgraph construction."""

    crawl_attempt_id: str | None = None
    crawl_state_manifest_writer: CrawlStateManifestWriter | None = None
    processing_activity_registry: ProcessingActivityRegistry | None = None
    page_settings: PageProcessorSettings | None = None


__all__ = [
    "CrawlerExecutionServices",
    "CrawlerExecutionOverrides",
]
