"""Regression tests: crawl-scope taxonomy and discovery preflight contract.

Invariants (crawl-scope governance):

- AdmissionScopeChecker classifies out-of-scope work as CRAWL_SCOPE_BLOCKED,
  never as URL_FILTERED: static URL eligibility and dynamic crawl scope are
  different policies with different owners.
- Source authorization is checked before seed/expansion exemptions.
- A host outside the source scope is scope-blocked; after dynamic
  expansion the same URL is eligible. Scope policy must stay with the
  scheduler's AdmissionScopeChecker, not move into the static URL filter.
- The scheduler-owned scope preflight is a read-only selection optimization;
  final admission re-checks scope and remains the authority.
- The scheduler always applies its static URL filter during final admission;
  discovery preflight cannot bypass scheduler policy.
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
from crawler.scheduling.admission.admission_scope_checker import (
    AdmissionScopeChecker,
)
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


class _RestrictingFilter:
    restrict_to_seed_hosts_enabled = True

    def evaluate_task(self, _task: object) -> object:
        return SimpleNamespace(allowed=True)


class _OpenFilter:
    restrict_to_seed_hosts_enabled = False

    def evaluate_task(self, _task: object) -> object:
        return SimpleNamespace(allowed=True)


class _TrackingFilter(_RestrictingFilter):
    def __init__(self) -> None:
        self.evaluate_calls = 0
        self.evaluated_tasks: list[object] = []

    def evaluate_task(self, task: object) -> object:
        self.evaluate_calls += 1
        self.evaluated_tasks.append(task)
        return SimpleNamespace(allowed=True)


class _HostBudgetTracker:
    def __init__(self) -> None:
        self._seed_hosts: set[str] = set()
        self._expanded_hosts: set[str] = set()

    def add_seed_host(self, host: str) -> None:
        self._seed_hosts.add(host)

    def expand_host(self, host: str) -> None:
        self._expanded_hosts.add(host)

    def remove_expansion(self, host: str) -> None:
        self._expanded_hosts.discard(host)

    def is_seed_host(self, host: str) -> bool:
        return host in self._seed_hosts

    def is_expanded_host(self, source_name: str, host: str) -> bool:
        return host in self._expanded_hosts


def _registry(*, page_hosts: set[str] | None = None) -> SourceScopeRegistry:
    return SourceScopeRegistry(
        scopes=[
            SourceScope(
                source_name="test",
                page_hosts=page_hosts or {"external.example"},
                asset_hosts=set(),
                redirect_hosts=set(),
            ),
        ],
    )


def _page(*, host: str = "external.example") -> CrawlTask:
    return CrawlTask(
        url=f"https://{host}/page",
        source_name="test",
        kind=MediaKind.PAGE,
        source_type="discovered_link",
    )


def _asset(*, host: str = "external.example") -> CrawlTask:
    return CrawlTask(
        url=f"https://{host}/image.jpg",
        source_name="test",
        kind=MediaKind.IMAGE,
        source_type="embedded_asset",
    )


def test_scope_checker_returns_crawl_scope_blocked_not_url_filtered() -> None:
    tracker = _HostBudgetTracker()
    tracker.add_seed_host("seed.example")
    checker = AdmissionScopeChecker(
        url_filter=_RestrictingFilter(),
        host_budget_tracker=tracker,
        source_scope_registry=_registry(
            page_hosts={"seed.example", "external.example"},
        ),
    )

    out_of_scope = checker.rejection_reason(
        task=_page(),
        host="external.example",
    )
    seed = checker.rejection_reason(
        task=_page(host="seed.example"),
        host="seed.example",
    )
    embedded_asset = checker.rejection_reason(
        task=_asset(),
        host="external.example",
    )

    assert out_of_scope == ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED
    assert out_of_scope is not ScheduleDecisionReason.URL_FILTERED
    assert seed is None
    assert embedded_asset is None


def test_dynamic_expansion_makes_same_url_scope_eligible() -> None:
    tracker = _HostBudgetTracker()
    tracker.add_seed_host("seed.example")
    checker = AdmissionScopeChecker(
        url_filter=_RestrictingFilter(),
        host_budget_tracker=tracker,
        source_scope_registry=_registry(),
    )
    task = _page()

    blocked = checker.rejection_reason(task=task, host="external.example")
    tracker.expand_host("external.example")
    expanded = checker.rejection_reason(task=task, host="external.example")

    assert blocked == ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED
    assert expanded is None


def test_source_unauthorized_host_is_blocked_regardless_of_expansion() -> None:
    tracker = _HostBudgetTracker()
    tracker.add_seed_host("seed.example")
    tracker.expand_host("external.example")
    checker = AdmissionScopeChecker(
        url_filter=_RestrictingFilter(),
        host_budget_tracker=tracker,
        source_scope_registry=_registry(
            page_hosts={"other.example"},
        ),
    )

    blocked = checker.rejection_reason(
        task=_page(),
        host="external.example",
    )
    assert blocked == ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED


def test_source_authorized_asset_host_is_allowed() -> None:
    tracker = _HostBudgetTracker()
    checker = AdmissionScopeChecker(
        url_filter=_RestrictingFilter(),
        host_budget_tracker=tracker,
        source_scope_registry=SourceScopeRegistry(
            scopes=[
                SourceScope(
                    source_name="test",
                    page_hosts={"page.example"},
                    asset_hosts={"cdn.example"},
                    redirect_hosts=set(),
                ),
            ],
        ),
    )

    result = checker.rejection_reason(
        task=_asset(host="cdn.example"),
        host="cdn.example",
    )
    assert result is None


def test_source_authorized_page_host_is_allowed() -> None:
    tracker = _HostBudgetTracker()
    tracker.add_seed_host("external.example")
    checker = AdmissionScopeChecker(
        url_filter=_RestrictingFilter(),
        host_budget_tracker=tracker,
        source_scope_registry=_registry(
            page_hosts={"external.example"},
        ),
    )

    result = checker.rejection_reason(
        task=_page(),
        host="external.example",
    )
    assert result is None


def _settings() -> SchedulingSettings:
    return SchedulingSettings(
        max_pending_per_host=12,
        max_pending_per_host_by_kind={},
        max_pending_per_host_under_pressure=None,
        max_pending_per_host_critical=None,
        max_pending_per_host_by_kind_under_pressure={},
        max_pending_per_host_by_kind_critical={},
        queue_high_watermark=1000,
        queue_critical_watermark=2000,
        dynamic_crawl_budget_enabled=False,
    )


def _admitter(
    *,
    url_filter: object | None,
    tracker: _HostBudgetTracker,
    registry: SourceScopeRegistry | None = None,
) -> SchedulerTaskAdmitter:
    return SchedulerTaskAdmitter(
        settings=_settings(),
        seen_urls=SeenUrlRegistry(max_seen=10_000, ttl_seconds=None),
        host_advice_tracker=HostAdviceTracker(
            ttl_seconds=None,
            max_hosts=32,
            host_normalizer=HostNormalizer(),
        ),
        url_filter=url_filter,  # type: ignore[arg-type]
        host_budget_tracker=tracker,  # type: ignore[arg-type]
        source_scope_registry=registry or _registry(),
    )


def _context(*, task: CrawlTask) -> object:
    return AdmissionContext(
        task=task,
        host="external.example",
        source=task.source_type,
        now=0.0,
        queue_size=10,
        host_pending=0,
        kind_host_pending=0,
        closed=False,
    )


def test_preview_allowed_but_final_check_blocked_after_scope_change() -> None:
    tracker = _HostBudgetTracker()
    tracker.add_seed_host("seed.example")
    tracker.expand_host("external.example")
    admitter = _admitter(
        url_filter=_RestrictingFilter(),
        tracker=tracker,
    )
    task = _page()

    preview = admitter.scope_rejection_reason(
        task=task,
        host="external.example",
    )
    assert preview is None

    tracker.remove_expansion("external.example")
    final = admitter.evaluate(ctx=_context(task=task))

    assert final.accepted is False
    assert final.reason == ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED


def test_final_admission_always_runs_static_filter_and_scope() -> None:
    tracker = _HostBudgetTracker()
    tracker.add_seed_host("seed.example")
    tracking_filter = _TrackingFilter()
    admitter = _admitter(
        url_filter=tracking_filter,
        tracker=tracker,
    )
    out_of_scope_task = _page()

    decision = admitter.evaluate(ctx=_context(task=out_of_scope_task))

    assert tracking_filter.evaluate_calls == 1
    assert tracking_filter.evaluated_tasks == [out_of_scope_task]
    assert decision.accepted is False
    assert decision.reason == ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED


def test_static_filter_and_scope_share_canonical_task_contract() -> None:
    tracker = _HostBudgetTracker()
    tracker.add_seed_host("seed.example")
    admitter = _admitter(
        url_filter=_RestrictingFilter(),
        tracker=tracker,
    )
    task = _page().clone(url="https://external.example/canonical")

    decision = admitter.evaluate(ctx=_context(task=task))

    assert decision.reason == ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED
    assert decision.normalized_url == task.url


def test_frontier_scope_preflight_is_read_only_batch() -> None:
    from crawler.scheduling.admission.scheduler_frontier import (
        SchedulerFrontier,
    )

    tracker = _HostBudgetTracker()
    tracker.add_seed_host("seed.example")
    tracker.expand_host("expanded.example")
    admitter = _admitter(
        url_filter=_RestrictingFilter(),
        tracker=tracker,
        registry=SourceScopeRegistry(
            scopes=[
                SourceScope(
                    source_name="test",
                    page_hosts={"external.example", "expanded.example"},
                    asset_hosts=set(),
                    redirect_hosts=set(),
                ),
            ],
        ),
    )
    frontier = object.__new__(SchedulerFrontier)
    frontier._url_normalizer = SimpleNamespace(
        normalize=lambda url: url.strip()
    )
    frontier._canonical_host_from_url = lambda url: url.split("//")[1].split(
        "/"
    )[0]
    frontier._task_admitter = admitter

    candidates = (
        _page(host="expanded.example"),
        _page(host="external.example"),
    )
    verdicts = frontier.discovery_scope_decisions_locked(tasks=candidates)

    assert len(verdicts) == 2
    assert verdicts[0].normalized_url == "https://expanded.example/page"
    assert verdicts[0].allowed is True
    assert verdicts[1].normalized_url == "https://external.example/page"
    assert verdicts[1].allowed is False
    assert tracker.is_expanded_host("test", "expanded.example") is True
