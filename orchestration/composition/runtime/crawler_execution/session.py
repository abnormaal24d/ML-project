"""Runtime session factory composition."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from config.settings.root import Settings
from crawler.metrics.prometheus_exporter import PrometheusExporter
from crawler.runtime.control.crawler_control_directory import (
    CrawlerControlDirectory,
)
from crawler.runtime.crawler_runtime_session import CrawlerRuntimeSession
from crawler.runtime.metrics.collection_metrics import CollectionMetrics
from crawler.runtime.state.crawl_state_writer import CrawlStateWriter
from crawler.analysis.enrichment.lanes.analysis_router import AnalysisRouter
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from crawler.worker.pool.worker_pool import WorkerPool
from crawler.worker.worker_scaler import WorkerScaler
from logger.factory import ProjectLoggerFactory


def build_runtime_session_factory(
    *,
    settings: Settings,
    logger_factory: ProjectLoggerFactory,
    worker_pool: WorkerPool,
    worker_scaler: WorkerScaler,
    state_writer: CrawlStateWriter,
    control_directory: CrawlerControlDirectory,
    metrics: CollectionMetrics | None,
    prometheus_exporter: PrometheusExporter | None,
    dataset_writer: DatasetWriter,
    analysis_router: AnalysisRouter | None,
) -> Callable[[], CrawlerRuntimeSession]:
    """Build the lazy CrawlerRuntimeSession factory used by the crawler."""
    from crawler.runtime.actions.crawl_runtime_actions import CrawlRuntimeActions
    from crawler.runtime.crawler_runtime_session import CrawlerRuntimeSession
    from crawler.runtime.metrics.crawler_metrics_reporter import (
        CrawlerMetricsReporter,
    )

    logger = logger_factory.get_logger_for(CrawlerRuntimeSession)
    metrics_enabled = metrics is not None and metrics.enabled

    metrics_reporter = CrawlerMetricsReporter(
        metrics=metrics,
        worker_pool=worker_pool,
        prometheus_exporter=prometheus_exporter,
        logger=logger,
    )

    def _build_runtime_session() -> CrawlerRuntimeSession:
        checkpoint_interval_seconds = state_writer.checkpoint_interval_seconds()

        metrics_interval = None
        metrics_log_interval_seconds = max(
            0.0,
            float(settings.collection.metrics.emit_snapshot_log_interval_seconds),
        )
        if metrics_enabled and metrics_log_interval_seconds > 0.0:
            metrics_interval = metrics_log_interval_seconds

        progress_interval = max(settings.crawler.progress_log_interval_seconds, 0.1)
        now = asyncio.get_running_loop().time()

        def emit_progress() -> None:
            return None

        runtime_actions = CrawlRuntimeActions(
            shutdown_poll_interval_seconds=settings.crawler.shutdown_poll_interval_seconds,
            crawl_output_gate=settings.crawl_output_gate,
            worker_scaler=worker_scaler,
            worker_pool=worker_pool,
            control_directory=control_directory,
            logger=logger,
            min_workers=settings.collection.autoscaler.min_workers,
            now=now,
            progress_interval=progress_interval,
            checkpoint_interval=checkpoint_interval_seconds,
            metrics_interval=metrics_interval,
            emit_progress=emit_progress,
            emit_metrics=metrics_reporter.emit,
            state_writer=state_writer,
            dataset_snapshot=dataset_writer.snapshot,
            metrics=metrics,
        )

        return CrawlerRuntimeSession(
            runtime_actions=runtime_actions,
            state_writer=state_writer,
            analysis_router=analysis_router,
        )

    return _build_runtime_session
