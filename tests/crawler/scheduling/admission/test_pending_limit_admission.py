"""Regression tests: pending limits admit ready-only and act as safety net.

Invariants (docs/architecture/scheduler_retry_semantics.md):

- ready=11, delayed=50, inflight=3, max_pending=12 -> one ready slot remains;
  the task is admitted. Delayed and in-flight work are not pending capacity.
- max_inflight=3, inflight=3, ready=5, max_pending=12 -> seven pending slots
  remain; execution concurrency and the pending frontier are independent.
- At pending == limit the admitter still rejects: the per-host cap remains the
  final concurrency safety net after capacity-aware selection.
"""

from __future__ import annotations

from types import SimpleNamespace

from config.collection.discovery import SchedulingSettings
from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.source_scope.source_scope_registry import (
    SourceScope,
    SourceScopeRegistry,
)
from crawler.scheduling.admission.admission_context import AdmissionContext
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecisionReason,
)
from crawler.scheduling.admission.scheduler_task_admitter import (
    SchedulerTaskAdmitter,
)
from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry
from crawler.scheduling.host_control.host_advice_tracker import (
    HostAdviceTracker,
)


def _settings(
    *,
    max_pending_per_host: int = 12,
    kind_limits: dict[str, int] | None = None,
) -> SchedulingSettings:
    return SchedulingSettings(
        max_pending_per_host=max_pending_per_host,
        max_pending_per_host_by_kind=kind_limits or {},
        max_pending_per_host_under_pressure=None,
        max_pending_per_host_critical=None,
        max_pending_per_host_by_kind_under_pressure={},
        max_pending_per_host_by_kind_critical={},
        queue_high_watermark=1000,
        queue_critical_watermark=2000,
        dynamic_crawl_budget_enabled=False,
    )


def _admitter(*, settings: SchedulingSettings) -> SchedulerTaskAdmitter:
    return SchedulerTaskAdmitter(
        settings=settings,
        seen_urls=SeenUrlRegistry(max_seen=10_000, ttl_seconds=None),
        host_advice_tracker=HostAdviceTracker(
            ttl_seconds=None,
            max_hosts=32,
            host_normalizer=HostNormalizer(),
        ),
        url_filter=_OpenUrlFilter(),
        source_scope_registry=SourceScopeRegistry(
            scopes=[
                SourceScope(
                    source_name="test",
                    page_hosts={"example.com", "example.test"},
                    asset_hosts=set(),
                    redirect_hosts=set(),
                ),
            ],
        ),
    )


class _OpenUrlFilter:
    """Admission filter stub that disables seed-scope restriction."""

    restrict_to_seed_hosts_enabled = False

    def evaluate_task(self, _task: object) -> object:
        return SimpleNamespace(allowed=True)


def _decision(
    *,
    admitter: SchedulerTaskAdmitter,
    host_pending: int,
    kind_host_pending: int = 0,
    kind: MediaKind = MediaKind.PAGE,
) -> object:
    task = CrawlTask(
        url="https://example.com/discovered",
        source_name="test",
        kind=kind,
        source_type="discovered_link",
    )
    return admitter.evaluate(
        ctx=AdmissionContext(
            task=task,
            host="example.com",
            source=task.source_type,
            now=0.0,
            queue_size=10,
            host_pending=host_pending,
            kind_host_pending=kind_host_pending,
            closed=False,
        )
    )


def test_admitted_when_ready_pending_below_limit_despite_delayed_and_inflight() -> (
    None
):
    admitter = _admitter(settings=_settings(max_pending_per_host=12))

    decision = _decision(admitter=admitter, host_pending=11)

    assert decision.accepted is True


def test_rejected_at_pending_limit_is_concurrency_safety_net() -> None:
    admitter = _admitter(settings=_settings(max_pending_per_host=12))

    decision = _decision(admitter=admitter, host_pending=12)

    assert decision.accepted is False
    assert (
        decision.reason == ScheduleDecisionReason.MAX_PENDING_PER_HOST_REACHED
    )


def test_inflight_does_not_consume_pending_frontier_slots() -> None:
    admitter = _admitter(settings=_settings(max_pending_per_host=12))

    decision = _decision(admitter=admitter, host_pending=5)

    assert decision.accepted is True


def test_kind_pending_limit_still_enforced_with_ready_semantics() -> None:
    settings = _settings(
        max_pending_per_host=12,
        kind_limits={"image": 4},
    )
    admitter = _admitter(settings=settings)

    allowed = _decision(
        admitter=admitter,
        host_pending=4,
        kind_host_pending=3,
        kind=MediaKind.IMAGE,
    )
    blocked = _decision(
        admitter=admitter,
        host_pending=4,
        kind_host_pending=4,
        kind=MediaKind.IMAGE,
    )

    assert allowed.accepted is True
    assert blocked.accepted is False
    assert (
        blocked.reason == ScheduleDecisionReason.MAX_PENDING_PER_HOST_REACHED
    )


def test_host_pending_limit_view_resolves_kind_and_pressure_policy() -> None:
    settings = _settings(
        max_pending_per_host=12,
        kind_limits={"page": 8},
    )
    admitter = _admitter(settings=settings)

    page_limit = admitter.host_pending_limit(
        kind=MediaKind.PAGE,
        host="example.com",
        queue_size=10,
    )
    audio_limit = admitter.host_pending_limit(
        kind=MediaKind.AUDIO,
        host="example.com",
        queue_size=10,
    )

    assert page_limit == 8
    assert audio_limit == 12
