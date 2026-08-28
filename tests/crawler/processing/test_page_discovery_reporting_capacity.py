"""Regression tests: capacity misses are reported, never rejected.

When the per-host frontier cap rejects during a concurrency race, reporting
must classify the outcome as ``capacity_skipped`` — not as a permanent
rejection — and must not write the asset to the rejected-assets output.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.processing.page_discovery_reporting import PageDiscoveryReporting
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecision,
    ScheduleDecisionReason,
)


class _RecordingLogger:
    def info(self, _event: str, **_fields: object) -> None:
        return None

    def debug(self, _event: str, **_fields: object) -> None:
        return None


class _RecordingReporter:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def record(self, **fields: object) -> None:
        self.records.append(fields)


def _task(*, kind: MediaKind = MediaKind.IMAGE) -> CrawlTask:
    return CrawlTask(
        url="https://example.com/asset.jpg",
        source_name="test",
        kind=kind,
        source_type="embedded_asset",
    )


def _selection(*, tasks: tuple[CrawlTask, ...]) -> Any:
    return SimpleNamespace(
        tasks=tasks,
        filtered_tasks=(),
        discovered_count=len(tasks),
        duplicate_count=0,
        filtered_count=0,
        truncated_count=0,
        capacity_skipped_count=0,
    )


def test_max_pending_race_is_capacity_skipped_not_rejected() -> None:
    task = _task()
    decisions = (
        ScheduleDecision.reject(
            reason=ScheduleDecisionReason.MAX_PENDING_PER_HOST_REACHED,
            normalized_url=task.url,
        ),
    )
    reporting = PageDiscoveryReporting(
        logger=_RecordingLogger(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rejected_discovery_reporter=None,
    )

    counts = reporting.count_admission_decisions(
        tasks=(task,),
        decisions=decisions,
    )

    assert counts.accepted == 0
    assert counts.scheduler_rejected == 0
    assert counts.capacity_skipped == 1
    assert counts.capacity_skipped_by_kind == {"image": 1}
    assert counts.capacity_skipped_by_reason == {
        "max_pending_per_host_reached": 1
    }
    assert counts.rejected_by_reason == {}


def test_scheduler_backpressure_is_capacity_skipped_not_rejected() -> None:
    task = _task()
    decisions = (
        ScheduleDecision.reject(
            reason=ScheduleDecisionReason.SCHEDULER_BACKPRESSURE,
            normalized_url=task.url,
        ),
    )
    reporting = PageDiscoveryReporting(
        logger=_RecordingLogger(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rejected_discovery_reporter=None,
    )

    counts = reporting.count_admission_decisions(
        tasks=(task,),
        decisions=decisions,
    )

    assert counts.capacity_skipped == 1
    assert counts.scheduler_rejected == 0


def test_capacity_miss_not_written_to_rejected_assets() -> None:
    task = _task()
    decisions = [
        ScheduleDecision.reject(
            reason=ScheduleDecisionReason.MAX_PENDING_PER_HOST_REACHED,
            normalized_url=task.url,
        )
    ]
    reporter = _RecordingReporter()
    reporting = PageDiscoveryReporting(
        logger=_RecordingLogger(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rejected_discovery_reporter=reporter,
    )

    rejected_assets = reporting.collect_rejected(
        parent_url="https://example.com/page",
        selection=_selection(tasks=(task,)),
        decisions=decisions,
    )

    assert rejected_assets == []
    assert reporter.records == []


def test_real_rejection_still_written_to_rejected_assets() -> None:
    task = _task()
    decisions = [
        ScheduleDecision.reject(
            reason=ScheduleDecisionReason.CRAWL_BUDGET_EXHAUSTED,
            normalized_url=task.url,
        )
    ]
    reporter = _RecordingReporter()
    reporting = PageDiscoveryReporting(
        logger=_RecordingLogger(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rejected_discovery_reporter=reporter,
    )

    rejected_assets = reporting.collect_rejected(
        parent_url="https://example.com/page",
        selection=_selection(tasks=(task,)),
        decisions=decisions,
    )

    assert rejected_assets == [(task, "crawl_budget_exhausted")]
    assert reporter.records == [
        {
            "url": task.url,
            "kind": "image",
            "reason": "crawl_budget_exhausted",
            "parent_url": "https://example.com/page",
            "context": {"selection_reason": None},
        }
    ]


def test_result_metrics_include_capacity_skipped() -> None:
    task = _task()
    decisions = (
        ScheduleDecision.reject(
            reason=ScheduleDecisionReason.MAX_PENDING_PER_HOST_REACHED,
            normalized_url=task.url,
        ),
    )
    reporting = PageDiscoveryReporting(
        logger=_RecordingLogger(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rejected_discovery_reporter=None,
    )
    counts = reporting.count_admission_decisions(
        tasks=(task,),
        decisions=decisions,
    )

    result = reporting.build_result_metrics(
        selection=_selection(tasks=(task,)),
        budget=SimpleNamespace(
            coverage_missing_by_kind={},
            discovery_scan_budget=8,
        ),
        selection_metrics={},
        admission=counts,
    )

    assert result["capacity_skipped"] == 1
    assert result["rejected"] == 0


def test_crawl_scope_blocked_is_scope_not_rejected_not_filtered() -> None:
    task = _task(kind=MediaKind.PAGE)
    decisions = (
        ScheduleDecision.reject(
            reason=ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED,
            normalized_url=task.url,
        ),
    )
    reporting = PageDiscoveryReporting(
        logger=_RecordingLogger(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rejected_discovery_reporter=None,
    )

    counts = reporting.count_admission_decisions(
        tasks=(task,),
        decisions=decisions,
    )

    assert counts.accepted == 0
    assert counts.scheduler_filtered == 0
    assert counts.scheduler_rejected == 0
    assert counts.capacity_skipped == 0
    assert counts.scope_blocked == 1
    assert counts.scope_blocked_by_kind == {"page": 1}
    assert counts.scope_blocked_by_reason == {"crawl_scope_blocked": 1}
    assert counts.rejected_by_reason == {}

    result = reporting.build_result_metrics(
        selection=_selection(tasks=(task,)),
        budget=SimpleNamespace(
            coverage_missing_by_kind={},
            discovery_scan_budget=8,
        ),
        selection_metrics={},
        admission=counts,
    )
    assert result["scope_blocked"] == 1
    assert result["filtered"] == 0
    assert result["rejected"] == 0


def test_scope_blocked_not_written_to_rejected_assets() -> None:
    task = _task()
    decisions = [
        ScheduleDecision.reject(
            reason=ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED,
            normalized_url=task.url,
        )
    ]
    reporter = _RecordingReporter()
    reporting = PageDiscoveryReporting(
        logger=_RecordingLogger(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rejected_discovery_reporter=reporter,
    )

    rejected_assets = reporting.collect_rejected(
        parent_url="https://example.com/page",
        selection=_selection(tasks=(task,)),
        decisions=decisions,
    )

    assert rejected_assets == []
    assert reporter.records == []


def test_result_metrics_merge_selection_and_scheduler_scope_blocked() -> None:
    task = _task(kind=MediaKind.PAGE)
    decisions = (
        ScheduleDecision.reject(
            reason=ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED,
            normalized_url=task.url,
        ),
    )
    reporting = PageDiscoveryReporting(
        logger=_RecordingLogger(),  # type: ignore[arg-type]
        host_normalizer=HostNormalizer(),
        rejected_discovery_reporter=None,
    )
    counts = reporting.count_admission_decisions(
        tasks=(task,),
        decisions=decisions,
    )
    selection = _selection(tasks=(task,))
    selection.scope_blocked_count = 2

    result = reporting.build_result_metrics(
        selection=selection,
        budget=SimpleNamespace(
            coverage_missing_by_kind={},
            discovery_scan_budget=8,
        ),
        selection_metrics={},
        admission=counts,
    )

    assert result["scope_blocked"] == 3
