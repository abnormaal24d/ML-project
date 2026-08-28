"""Fetch-related runtime metrics recording."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable

from crawler.numeric import coerce_finite_float
from crawler.runtime.metrics.host_metrics_registry import (
    HostMetricsRegistry,
    HostMetricsState,
)
from crawler.runtime.metrics.metrics_snapshot import clamp, ewma


@dataclass(slots=True)
class FetchMetricsTotals:
    """Mutable aggregate counters updated by fetch recording."""

    requests_total: int = 0
    successes_total: int = 0
    failures_total: int = 0
    skipped_total: int = 0
    bytes_total: int = 0
    latency_ewma: float | None = None
    quality_ewma: float = 0.5


def record_fetch(
    *,
    registry: HostMetricsRegistry,
    totals: FetchMetricsTotals,
    host: str | None,
    status_code: int,
    latency_seconds: float,
    bytes_downloaded: int,
    quality_score: float | None,
) -> None:
    """Record a completed host fetch with status, latency, and size."""
    state = registry.state_for_host(host=host)
    state.requests += 1
    totals.requests_total += 1

    if 200 <= status_code < 400:
        state.successes += 1
        totals.successes_total += 1
    else:
        state.failures += 1
        totals.failures_total += 1

    bounded_bytes = max(0, int(bytes_downloaded))
    state.bytes_downloaded += bounded_bytes
    totals.bytes_total += bounded_bytes

    record_latency(
        state=state,
        totals=totals,
        latency_seconds=coerce_finite_float(
            latency_seconds,
            default=0.0,
            minimum=0.0,
        ),
    )

    state.last_status_code = int(status_code)
    state.last_seen_monotonic = monotonic()

    if quality_score is not None:
        record_quality(
            state=state,
            totals=totals,
            quality_score=quality_score,
        )


def record_fetch_skipped(
    *,
    registry: HostMetricsRegistry,
    totals: FetchMetricsTotals,
    host: str | None,
    reason: str,
    skipped_by_reason: dict[str, int],
    normalize_reason: Callable[[str | None], str],
    latency_seconds: float = 0.0,
    bytes_downloaded: int = 0,
) -> None:
    """Record a rules-controlled fetch skip separately from failures."""
    state = registry.state_for_host(host=host)
    state.skipped += 1
    totals.skipped_total += 1

    normalized_reason = normalize_reason(reason)
    skipped_by_reason[normalized_reason] = (
        skipped_by_reason.get(normalized_reason, 0) + 1
    )

    bounded_bytes = max(0, int(bytes_downloaded))
    state.bytes_downloaded += bounded_bytes
    totals.bytes_total += bounded_bytes

    bounded_latency = coerce_finite_float(
        latency_seconds,
        default=0.0,
        minimum=0.0,
    )
    if bounded_latency > 0.0:
        record_latency(
            state=state,
            totals=totals,
            latency_seconds=bounded_latency,
        )

    state.last_seen_monotonic = monotonic()


def record_latency(
    *,
    state: HostMetricsState,
    totals: FetchMetricsTotals,
    latency_seconds: float,
) -> None:
    """Record one real latency observation in host and global EWMAs."""
    state.latency_ewma = (
        latency_seconds
        if state.latency_ewma is None
        else ewma(state.latency_ewma, latency_seconds)
    )
    totals.latency_ewma = (
        latency_seconds
        if totals.latency_ewma is None
        else ewma(totals.latency_ewma, latency_seconds)
    )


def record_quality(
    *,
    state: HostMetricsState,
    totals: FetchMetricsTotals,
    quality_score: float,
) -> None:
    """Record one bounded quality observation."""
    bounded_quality = clamp(
        float(quality_score),
        minimum=0.0,
        maximum=1.0,
    )
    state.quality_ewma = ewma(state.quality_ewma, bounded_quality)
    totals.quality_ewma = ewma(totals.quality_ewma, bounded_quality)
