"""Immutable runtime metrics snapshot models and shared helpers."""

from __future__ import annotations

from dataclasses import dataclass

from crawler.numeric import coerce_finite_float

DEFAULT_HOST = "unknown"
DEFAULT_QUALITY_SCORE = 0.5


@dataclass(frozen=True, slots=True)
class HostMetricSnapshot:
    """Immutable per-host metric snapshot."""

    host: str
    requests: int
    successes: int
    failures: int
    skipped: int
    average_latency_seconds: float | None
    bytes_downloaded: int
    quality_score: float
    last_status_code: int | None
    last_seen_monotonic: float


def ewma(previous: float, current: float, *, alpha: float = 0.2) -> float:
    """Return the exponentially weighted moving average."""

    safe_alpha = coerce_finite_float(
        alpha,
        default=0.2,
        minimum=0.0,
        maximum=1.0,
    )
    safe_previous = coerce_finite_float(previous, default=0.0)
    safe_current = coerce_finite_float(current, default=safe_previous)
    return coerce_finite_float(
        (safe_alpha * safe_current) + ((1.0 - safe_alpha) * safe_previous),
        default=safe_current,
    )


def clamp(value: float, *, minimum: float, maximum: float) -> float:
    """Clamp a float value to an inclusive range."""
    return coerce_finite_float(
        value,
        default=minimum,
        minimum=minimum,
        maximum=maximum,
    )


def normalize_stage(stage: str | None) -> str:
    """Normalize a blacklist stage label."""
    normalized = (stage or "unspecified").strip().lower()
    return normalized or "unspecified"


def normalize_reason(reason: str | None) -> str:
    """Normalize a reason label."""
    normalized = (reason or "unspecified").strip().lower()
    return normalized or "unspecified"


def increment_counter(counter: dict[str, int], key: str) -> None:
    """Increment one named counter."""
    counter[key] = counter.get(key, 0) + 1


def sorted_items(counter: dict[str, int]) -> tuple[tuple[str, int], ...]:
    """Return deterministic immutable counter items."""
    return tuple(sorted(counter.items()))


def build_runtime_metrics_fields(
    *,
    collection_snapshot: MetricsSnapshot,
    worker_snapshot: object,
) -> dict[str, object]:
    """Compute derived runtime metrics fields from collection and worker snapshots.

    Pure transformation: no logging, no side effects, no external calls.
    """
    processed_total = (
        collection_snapshot.requests_total + collection_snapshot.skipped_total
    )

    tasks_per_100_fetches = 0.0
    if collection_snapshot.requests_total > 0:
        tasks_per_100_fetches = (
            worker_snapshot.completed_task_count
            / collection_snapshot.requests_total
        ) * 100.0

    fields: dict[str, object] = {
        "fetch_attempts_total": collection_snapshot.requests_total,
        "fetch_success_total": collection_snapshot.successes_total,
        "fetch_failures_total": collection_snapshot.failures_total,
        "task_failures_total": worker_snapshot.failure_count,
        "non_fatal_timeouts_total": worker_snapshot.non_fatal_timeout_count,
        "retry_exhausted_total": worker_snapshot.retry_exhausted_count,
        "skipped_before_fetch_total": collection_snapshot.skipped_total,
        "skip_rate": round(
            (collection_snapshot.skipped_total / processed_total) * 100.0, 1
        )
        if processed_total > 0
        else 0.0,
        "downloaded_bytes_total": collection_snapshot.bytes_total,
        "avg_latency_seconds": round(
            collection_snapshot.average_latency_seconds or 0.0, 2
        ),
        "quality_score": round(collection_snapshot.quality_score, 4),
        "attempted_success_rate": round(
            (
                collection_snapshot.successes_total
                / collection_snapshot.requests_total
            )
            * 100.0,
            1,
        )
        if collection_snapshot.requests_total > 0
        else 0.0,
        "tasks_per_100_fetches": round(tasks_per_100_fetches, 1),
        "top_hosts": [
            {
                "host": host.host,
                "requests": host.requests,
                "successes": host.successes,
                "failures": host.failures,
                "quality_score": round(host.quality_score, 4),
                "avg_latency_seconds": round(
                    host.average_latency_seconds or 0.0, 4
                ),
            }
            for host in collection_snapshot.hosts
        ],
    }

    if collection_snapshot.blacklist_total > 0:
        fields["blacklist_total"] = collection_snapshot.blacklist_total
        fields["blacklist_by_stage"] = dict(
            collection_snapshot.blacklist_by_stage
        )
        fields["blacklist_by_reason"] = dict(
            collection_snapshot.blacklist_by_reason
        )

    if collection_snapshot.skipped_by_reason:
        fields["skipped_by_reason"] = dict(
            collection_snapshot.skipped_by_reason
        )

    return fields


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """Immutable global and optionally per-host metrics snapshot."""

    requests_total: int
    successes_total: int
    failures_total: int
    skipped_total: int
    bytes_total: int
    average_latency_seconds: float | None
    quality_score: float
    blacklist_total: int
    blacklist_by_stage: tuple[tuple[str, int], ...]
    blacklist_by_reason: tuple[tuple[str, int], ...]
    skipped_by_reason: tuple[tuple[str, int], ...]
    hosts: tuple[HostMetricSnapshot, ...]

    @classmethod
    def empty(cls) -> MetricsSnapshot:
        """Return a zero-valued snapshot."""
        return cls(
            requests_total=0,
            successes_total=0,
            failures_total=0,
            skipped_total=0,
            bytes_total=0,
            average_latency_seconds=None,
            quality_score=0.0,
            blacklist_total=0,
            blacklist_by_stage=(),
            blacklist_by_reason=(),
            skipped_by_reason=(),
            hosts=(),
        )
