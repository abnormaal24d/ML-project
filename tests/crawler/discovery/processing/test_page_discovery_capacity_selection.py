"""Regression tests: host-cap-aware selection in the discovery pipeline.

A candidate on a full host must not consume a selection slot; the next
ranked candidate gets its chance. Capacity skips are reported separately
and are never counted as selection truncation.
"""

from __future__ import annotations

from types import SimpleNamespace

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.discovery.processing.page_discovery_limits import (
    limit_ranked_tasks,
)
from crawler.metrics.media_discovery_metrics import MediaDiscoveryMetrics
from crawler.scheduling.progress.scheduler_snapshot_reader import (
    DiscoveryCapacitySnapshot,
    build_discovery_capacity_snapshot,
)


def _task(
    *, host: str, index: int, kind: MediaKind = MediaKind.PAGE
) -> CrawlTask:
    return CrawlTask(
        url=f"https://{host}/{kind.value}-{index}",
        source_name="test",
        kind=kind,
        source_type="discovered_link",
    )


def _host_of(task: CrawlTask) -> str | None:
    return task.url.split("//")[1].split("/")[0]


def test_full_host_candidate_consumes_no_selection_slot() -> None:
    host_a_candidates = [_task(host="a.example", index=i) for i in range(10)]
    host_b_candidates = [_task(host="b.example", index=i) for i in range(16)]
    ranked = host_a_candidates + host_b_candidates

    selected, capacity_skipped, truncated_count = limit_ranked_tasks(
        ranked_tasks=ranked,
        max_total=16,
        max_pages=16,
        max_embedded_assets=16,
        max_non_page_media=16,
        remaining_capacity={
            ("a.example", MediaKind.PAGE): 2,
            ("b.example", MediaKind.PAGE): 14,
        },
        host_of=_host_of,
    )

    selected_from_a = sum(1 for task in selected if "a.example" in task.url)
    selected_from_b = sum(1 for task in selected if "b.example" in task.url)

    assert selected_from_a == 2
    assert selected_from_b == 14
    assert len(selected) == 16
    assert len(capacity_skipped) == 8
    assert truncated_count == 2
    assert all("a.example" in task.url for task in capacity_skipped)


def test_capacity_unlimited_when_no_scheduler_capacity_provided() -> None:
    ranked = [_task(host="a.example", index=i) for i in range(20)]

    selected, capacity_skipped, truncated_count = limit_ranked_tasks(
        ranked_tasks=ranked,
        max_total=16,
        max_pages=16,
        max_embedded_assets=16,
        max_non_page_media=16,
    )

    assert len(selected) == 16
    assert capacity_skipped == []
    assert truncated_count == 4


def test_capacity_skipped_not_counted_as_truncated() -> None:
    ranked = [_task(host="a.example", index=i) for i in range(20)]

    selected, capacity_skipped, truncated_count = limit_ranked_tasks(
        ranked_tasks=ranked,
        max_total=16,
        max_pages=16,
        max_embedded_assets=16,
        max_non_page_media=16,
        remaining_capacity={("a.example", MediaKind.PAGE): 0},
        host_of=_host_of,
    )

    assert len(selected) == 0
    assert len(capacity_skipped) == 20
    assert truncated_count == 0


def test_capacity_respected_together_with_page_and_kind_quotas() -> None:
    ranked = [_task(host="a.example", index=i) for i in range(10)]

    selected, capacity_skipped, truncated_count = limit_ranked_tasks(
        ranked_tasks=ranked,
        max_total=16,
        max_pages=3,
        max_embedded_assets=16,
        max_non_page_media=16,
        remaining_capacity={("a.example", MediaKind.PAGE): 1},
        host_of=_host_of,
    )

    assert len(selected) == 1
    assert len(capacity_skipped) == 9
    assert truncated_count == 0


def test_metrics_record_capacity_skipped_separately() -> None:
    metrics = MediaDiscoveryMetrics(host_normalizer=_NoopNormalizer())
    skipped = [
        _task(host="a.example", index=0),
        _task(host="a.example", index=1, kind=MediaKind.IMAGE),
    ]

    metrics.record_capacity_skipped_many(tasks=skipped)

    payload = metrics.as_payload()
    assert payload["capacity_skipped_by_kind"] == {
        "page": 1,
        "image": 1,
    }
    assert payload["capacity_skipped_by_reason"] == {
        "page:frontier_capacity": 1,
        "image:frontier_capacity": 1,
    }


def test_build_discovery_capacity_snapshot_uses_injected_ready_state() -> None:
    snapshot = build_discovery_capacity_snapshot(
        ready_pending={"a.example": 3, "b.example": 0},
        host_limit_fn=lambda kind, host: 4 if host == "a.example" else 8,
        kind_host_pending_fn=lambda host, kind: (
            3 if host == "a.example" and kind is MediaKind.PAGE else 0
        ),
        queue_size=10,
        now=1.0,
    )

    assert snapshot.remaining(host="a.example", kind=MediaKind.PAGE) == 1
    assert snapshot.remaining(host="a.example", kind=MediaKind.IMAGE) == 4
    assert snapshot.remaining(host="b.example", kind=MediaKind.PAGE) == 8
    assert (
        snapshot.remaining(host="unknown.example", kind=MediaKind.PAGE) is None
    )
    assert snapshot.queue_size == 10


def test_scheduler_capacity_map_handles_empty_and_missing_sources() -> None:
    from crawler.discovery.processing.page_discovery_selection import (
        _scheduler_capacity_map,
    )

    assert _scheduler_capacity_map(None) is None
    assert _scheduler_capacity_map(SimpleNamespace()) is None

    empty = DiscoveryCapacitySnapshot(
        by_host_kind={},
        queue_size=0,
        captured_at_monotonic=1.0,
    )
    assert _scheduler_capacity_map(empty) == {}

    exhausted_key = ("a.example", MediaKind.PAGE)
    exhausted = DiscoveryCapacitySnapshot(
        by_host_kind={exhausted_key: 0},
        queue_size=1,
        captured_at_monotonic=2.0,
    )
    assert _scheduler_capacity_map(exhausted) == {exhausted_key: 0}


class _NoopNormalizer:
    def normalize(self, host: str | None) -> str | None:
        return host
