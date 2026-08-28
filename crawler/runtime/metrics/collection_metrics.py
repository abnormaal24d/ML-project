"""Collect crawl runtime metrics and expose immutable snapshots."""

from __future__ import annotations

from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.numeric import coerce_finite_float
from crawler.runtime.metrics import fetch_metrics
from crawler.runtime.metrics.fetch_metrics import FetchMetricsTotals
from crawler.runtime.metrics.host_metrics_registry import HostMetricsRegistry
from crawler.runtime.metrics.metrics_snapshot import (
    DEFAULT_QUALITY_SCORE,
    MetricsSnapshot,
    increment_counter,
    normalize_reason,
    normalize_stage,
    sorted_items,
)
from logger.project_logger import ProjectLogger


class CollectionMetrics:
    """Collect fetch and blacklist-denial metrics."""

    def __init__(
        self,
        *,
        enabled: bool,
        logger: ProjectLogger,
        host_normalizer: HostNormalizer,
    ) -> None:
        self._enabled = enabled
        self._logger = logger
        self._registry = HostMetricsRegistry(
            host_normalizer=host_normalizer,
        )
        self._totals = FetchMetricsTotals(
            quality_ewma=DEFAULT_QUALITY_SCORE,
        )
        self._blacklist_total = 0
        self._blacklist_by_stage: dict[str, int] = {}
        self._blacklist_by_reason: dict[str, int] = {}
        self._skipped_by_reason: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        """Return whether metrics collection is active."""
        return self._enabled

    def record_fetch(
        self,
        *,
        host: str | None,
        status_code: int,
        latency_seconds: float,
        bytes_downloaded: int,
        quality_score: float | None,
    ) -> None:
        """Record a completed host fetch with status, latency, and size."""
        if not self._enabled:
            return

        fetch_metrics.record_fetch(
            registry=self._registry,
            totals=self._totals,
            host=host,
            status_code=status_code,
            latency_seconds=latency_seconds,
            bytes_downloaded=bytes_downloaded,
            quality_score=quality_score,
        )

    def record_fetch_skipped(
        self,
        *,
        host: str | None,
        reason: str,
        latency_seconds: float = 0.0,
        bytes_downloaded: int = 0,
    ) -> None:
        """Record a rules-controlled fetch skip separately from failures."""
        if not self._enabled:
            return

        fetch_metrics.record_fetch_skipped(
            registry=self._registry,
            totals=self._totals,
            host=host,
            reason=reason,
            skipped_by_reason=self._skipped_by_reason,
            normalize_reason=normalize_reason,
            latency_seconds=latency_seconds,
            bytes_downloaded=bytes_downloaded,
        )

    def record_blacklist_block(
        self,
        *,
        url: str,
        host: str | None,
        stage: str | None,
        reason: str,
    ) -> None:
        """Record a blacklist denial without changing fetch counters."""
        if not self._enabled:
            return

        normalized_stage = normalize_stage(stage)
        normalized_reason = normalize_reason(reason)
        self._blacklist_total += 1
        increment_counter(self._blacklist_by_stage, normalized_stage)
        increment_counter(self._blacklist_by_reason, normalized_reason)
        self._logger.debug(
            "blacklist_block_recorded",
            url=url,
            host=self._registry.resolve_metric_host_key(host=host),
            stage=normalized_stage,
            reason=normalized_reason,
        )

    def snapshot(
        self,
        *,
        host_limit: int | None = None,
    ) -> MetricsSnapshot:
        """Return totals and at most ``host_limit`` ranked host snapshots.

        ``None`` includes every known host, while zero omits hosts entirely.
        A finite positive limit ranks hosts by request count and only
        materializes the selected snapshots.
        """
        if not self._enabled:
            return MetricsSnapshot.empty()

        latency_ewma = self._totals.latency_ewma
        return MetricsSnapshot(
            requests_total=self._totals.requests_total,
            successes_total=self._totals.successes_total,
            failures_total=self._totals.failures_total,
            skipped_total=self._totals.skipped_total,
            bytes_total=self._totals.bytes_total,
            average_latency_seconds=(
                None
                if latency_ewma is None
                else round(
                    coerce_finite_float(latency_ewma, default=0.0),
                    6,
                )
            ),
            quality_score=round(
                coerce_finite_float(
                    self._totals.quality_ewma,
                    default=DEFAULT_QUALITY_SCORE,
                    minimum=0.0,
                    maximum=1.0,
                ),
                6,
            ),
            blacklist_total=self._blacklist_total,
            blacklist_by_stage=sorted_items(self._blacklist_by_stage),
            blacklist_by_reason=sorted_items(self._blacklist_by_reason),
            skipped_by_reason=sorted_items(self._skipped_by_reason),
            hosts=self._registry.snapshots(host_limit=host_limit),
        )

    def quality_score(self) -> float:
        """Return the current global quality signal."""
        if not self._enabled:
            return 0.0

        return coerce_finite_float(
            self._totals.quality_ewma,
            default=DEFAULT_QUALITY_SCORE,
            minimum=0.0,
            maximum=1.0,
        )

    def average_latency_seconds(self) -> float | None:
        """Return latency EWMA, or ``None`` before the first observation."""
        if not self._enabled:
            return None

        if self._totals.latency_ewma is None:
            return None
        return coerce_finite_float(self._totals.latency_ewma, default=0.0)
