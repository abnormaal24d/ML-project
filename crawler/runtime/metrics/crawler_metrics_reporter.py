"""Crawler metrics reporter.

Computes and emits crawler metrics snapshots to logging and Prometheus.
This is a runtime service, not a composition concern.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.metrics.prometheus_exporter import PrometheusExporter
    from crawler.runtime.metrics.collection_metrics import CollectionMetrics
    from crawler.worker.pool.worker_pool import WorkerPool


@dataclass(frozen=True, slots=True)
class CrawlerMetricsReporter:
    """Computes and emits crawler metrics snapshot."""

    metrics: CollectionMetrics | None
    worker_pool: WorkerPool
    prometheus_exporter: PrometheusExporter | None
    logger: object

    def emit(self) -> None:
        """Compute and emit crawler metrics snapshot."""
        if self.metrics is None or not self.metrics.enabled:
            return

        from crawler.runtime.metrics.metrics_snapshot import (
            build_runtime_metrics_fields,
        )

        worker_snapshot = self.worker_pool.snapshot(
            now=asyncio.get_running_loop().time()
        )
        collection_snapshot = self.metrics.snapshot(host_limit=3)
        fields = build_runtime_metrics_fields(
            collection_snapshot=collection_snapshot,
            worker_snapshot=worker_snapshot,
        )

        self.logger.info("crawler_metrics", **fields)
        if self.prometheus_exporter is not None:
            self.prometheus_exporter.publish(fields)
