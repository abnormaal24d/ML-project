"""Tests for the pure build_runtime_metrics_fields transformation."""

from __future__ import annotations

from types import SimpleNamespace

from crawler.runtime.metrics.metrics_snapshot import (
    MetricsSnapshot,
    build_runtime_metrics_fields,
)


def _snapshot(**overrides) -> MetricsSnapshot:
    defaults = dict(
        requests_total=100,
        successes_total=80,
        failures_total=20,
        skipped_total=10,
        bytes_total=50000,
        average_latency_seconds=1.5,
        quality_score=0.75,
        blacklist_total=0,
        blacklist_by_stage=(),
        blacklist_by_reason=(),
        skipped_by_reason=(),
        hosts=(),
    )
    defaults.update(overrides)
    return MetricsSnapshot(**defaults)


def _worker_snapshot(**overrides) -> SimpleNamespace:
    defaults = dict(
        completed_task_count=75,
        failure_count=5,
        non_fatal_timeout_count=2,
        retry_exhausted_count=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_normal_values() -> None:
    fields = build_runtime_metrics_fields(
        collection_snapshot=_snapshot(),
        worker_snapshot=_worker_snapshot(),
    )
    assert fields["fetch_attempts_total"] == 100
    assert fields["fetch_success_total"] == 80
    assert fields["fetch_failures_total"] == 20
    assert fields["task_failures_total"] == 5
    assert fields["skip_rate"] == 9.1
    assert fields["attempted_success_rate"] == 80.0
    assert fields["tasks_per_100_fetches"] == 75.0
    assert fields["avg_latency_seconds"] == 1.5
    assert fields["quality_score"] == 0.75


def test_zero_fetch_attempts() -> None:
    fields = build_runtime_metrics_fields(
        collection_snapshot=_snapshot(requests_total=0),
        worker_snapshot=_worker_snapshot(),
    )
    assert fields["skip_rate"] == 100.0
    assert fields["attempted_success_rate"] == 0.0
    assert fields["tasks_per_100_fetches"] == 0.0


def test_zero_processed_total() -> None:
    fields = build_runtime_metrics_fields(
        collection_snapshot=_snapshot(requests_total=0, skipped_total=0),
        worker_snapshot=_worker_snapshot(),
    )
    assert fields["skip_rate"] == 0.0


def test_empty_hosts() -> None:
    fields = build_runtime_metrics_fields(
        collection_snapshot=_snapshot(hosts=()),
        worker_snapshot=_worker_snapshot(),
    )
    assert fields["top_hosts"] == []


def test_blacklist_included_when_present() -> None:
    fields = build_runtime_metrics_fields(
        collection_snapshot=_snapshot(
            blacklist_total=5,
            blacklist_by_stage=(("robots", 3), ("policy", 2)),
            blacklist_by_reason=(("blocked", 5),),
        ),
        worker_snapshot=_worker_snapshot(),
    )
    assert fields["blacklist_total"] == 5
    assert fields["blacklist_by_stage"] == {"robots": 3, "policy": 2}
    assert fields["blacklist_by_reason"] == {"blocked": 5}


def test_blacklist_excluded_when_zero() -> None:
    fields = build_runtime_metrics_fields(
        collection_snapshot=_snapshot(blacklist_total=0),
        worker_snapshot=_worker_snapshot(),
    )
    assert "blacklist_total" not in fields
    assert "blacklist_by_stage" not in fields
    assert "blacklist_by_reason" not in fields


def test_skipped_by_reason_included_when_present() -> None:
    fields = build_runtime_metrics_fields(
        collection_snapshot=_snapshot(
            skipped_by_reason=(("governance", 3), ("duplicate", 2)),
        ),
        worker_snapshot=_worker_snapshot(),
    )
    assert fields["skipped_by_reason"] == {"governance": 3, "duplicate": 2}


def test_skipped_by_reason_excluded_when_empty() -> None:
    fields = build_runtime_metrics_fields(
        collection_snapshot=_snapshot(skipped_by_reason=()),
        worker_snapshot=_worker_snapshot(),
    )
    assert "skipped_by_reason" not in fields


def test_null_latency_treated_as_zero() -> None:
    fields = build_runtime_metrics_fields(
        collection_snapshot=_snapshot(average_latency_seconds=None),
        worker_snapshot=_worker_snapshot(),
    )
    assert fields["avg_latency_seconds"] == 0.0


def test_output_keys_are_complete() -> None:
    fields = build_runtime_metrics_fields(
        collection_snapshot=_snapshot(),
        worker_snapshot=_worker_snapshot(),
    )
    expected_keys = {
        "fetch_attempts_total",
        "fetch_success_total",
        "fetch_failures_total",
        "task_failures_total",
        "non_fatal_timeouts_total",
        "retry_exhausted_total",
        "skipped_before_fetch_total",
        "skip_rate",
        "downloaded_bytes_total",
        "avg_latency_seconds",
        "quality_score",
        "attempted_success_rate",
        "tasks_per_100_fetches",
        "top_hosts",
    }
    assert set(fields.keys()) == expected_keys
