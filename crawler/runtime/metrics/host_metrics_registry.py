"""Registry and mutable state for per-host runtime metrics."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import nsmallest
from typing import TYPE_CHECKING

from crawler.numeric import coerce_finite_float
from crawler.runtime.metrics.metrics_snapshot import (
    DEFAULT_HOST,
    DEFAULT_QUALITY_SCORE,
    HostMetricSnapshot,
)

if TYPE_CHECKING:
    from crawler.governance.domains.host_normalizer import HostNormalizer


@dataclass(slots=True)
class HostMetricsState:
    """Mutable metrics for one canonical host."""

    requests: int = 0
    successes: int = 0
    failures: int = 0
    skipped: int = 0
    latency_ewma: float | None = None
    bytes_downloaded: int = 0
    quality_ewma: float = DEFAULT_QUALITY_SCORE
    last_status_code: int | None = None
    last_seen_monotonic: float = 0.0


class HostMetricsRegistry:
    """Track mutable per-host metrics keyed by canonical host."""

    def __init__(
        self,
        *,
        host_normalizer: HostNormalizer,
    ) -> None:
        self._host_normalizer = host_normalizer
        self._hosts: dict[str, HostMetricsState] = {}

    def resolve_metric_host_key(self, *, host: str | None) -> str:
        canonical_host = self._host_normalizer.normalize(host)
        return canonical_host or DEFAULT_HOST

    def state_for_host(self, *, host: str | None) -> HostMetricsState:
        host_key = self.resolve_metric_host_key(host=host)
        existing = self._hosts.get(host_key)
        if existing is not None:
            return existing

        state = HostMetricsState()
        self._hosts[host_key] = state
        return state

    def snapshots(
        self,
        *,
        host_limit: int | None,
    ) -> tuple[HostMetricSnapshot, ...]:
        """Materialize all hosts or only the requested top-ranked hosts."""
        if host_limit is None:
            ranked_items = sorted(
                self._hosts.items(),
                key=lambda item: item[0],
            )
        else:
            normalized_limit = max(0, int(host_limit))
            if normalized_limit == 0 or not self._hosts:
                return ()
            ranked_items = nsmallest(
                normalized_limit,
                self._hosts.items(),
                key=lambda item: (-item[1].requests, item[0]),
            )

        return tuple(
            build_host_snapshot(host=host, state=state)
            for host, state in ranked_items
        )


def build_host_snapshot(
    *,
    host: str,
    state: HostMetricsState,
) -> HostMetricSnapshot:
    """Convert mutable host state into an immutable snapshot."""
    return HostMetricSnapshot(
        host=host,
        requests=state.requests,
        successes=state.successes,
        failures=state.failures,
        skipped=state.skipped,
        average_latency_seconds=(
            None
            if state.latency_ewma is None
            else round(
                coerce_finite_float(state.latency_ewma, default=0.0),
                6,
            )
        ),
        bytes_downloaded=state.bytes_downloaded,
        quality_score=round(
            coerce_finite_float(
                state.quality_ewma,
                default=DEFAULT_QUALITY_SCORE,
                minimum=0.0,
                maximum=1.0,
            ),
            6,
        ),
        last_status_code=state.last_status_code,
        last_seen_monotonic=round(
            coerce_finite_float(state.last_seen_monotonic, default=0.0),
            6,
        ),
    )
