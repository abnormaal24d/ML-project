"""Discovery metrics, logging, and rejection reporting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from crawler.metrics.media_discovery_metrics import MediaDiscoveryMetrics
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecisionReason,
)
from crawler.scheduling.reason_key import reason_key
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.governance.domains.host_normalizer import HostNormalizer


_CAPACITY_REASONS = frozenset(
    {
        ScheduleDecisionReason.MAX_PENDING_PER_HOST_REACHED,
        ScheduleDecisionReason.SCHEDULER_BACKPRESSURE,
    }
)

_SCOPE_REASONS = frozenset(
    {
        ScheduleDecisionReason.CRAWL_SCOPE_BLOCKED,
    }
)


@dataclass(frozen=True, slots=True)
class AdmissionCounts:
    accepted: int
    scheduler_filtered: int
    scheduler_rejected: int
    accepted_by_kind: Counter[str]
    rejected_by_kind: Counter[str]
    rejected_by_reason: Counter[str]
    metrics: MediaDiscoveryMetrics
    capacity_skipped: int = 0
    capacity_skipped_by_kind: Counter[str] = field(default_factory=Counter)
    capacity_skipped_by_reason: Counter[str] = field(default_factory=Counter)
    scope_blocked: int = 0
    scope_blocked_by_kind: Counter[str] = field(default_factory=Counter)
    scope_blocked_by_reason: Counter[str] = field(default_factory=Counter)


class PageDiscoveryReporting:
    def __init__(
        self,
        *,
        logger: ProjectLogger,
        host_normalizer: HostNormalizer,
        rejected_discovery_reporter: Any | None,
    ) -> None:
        self._logger = logger
        self._host_normalizer = host_normalizer
        self._rejected_discovery_reporter = rejected_discovery_reporter

    def build_discovery_rejected_counters(
        self, metrics: Mapping[str, object]
    ) -> tuple[Counter[str], Counter[str]]:
        """Pure extraction of rejected counters from selection metrics."""
        by_reason: Counter[str] = Counter()
        for metric_key in (
            "duplicate_by_reason",
            "filtered_by_reason",
            "truncated_by_reason",
        ):
            values = metrics.get(metric_key, {}) or {}
            if not isinstance(values, dict):
                continue
            by_reason.update(
                {
                    str(k): int(v)
                    for k, v in values.items()
                    if isinstance(v, int)
                }
            )

        by_kind: Counter[str] = Counter()
        for metric_key in (
            "filtered_by_kind",
            "duplicate_by_kind",
            "truncated_by_kind",
        ):
            values = metrics.get(metric_key, {}) or {}
            if not isinstance(values, dict):
                continue
            by_kind.update(
                {
                    str(k): int(v)
                    for k, v in values.items()
                    if isinstance(v, int)
                }
            )

        return by_reason, by_kind

    def count_admission_decisions(
        self, *, tasks: Any, decisions: Any
    ) -> AdmissionCounts:
        accepted = 0
        scheduler_filtered = 0
        scheduler_rejected = 0
        capacity_skipped = 0
        scope_blocked = 0
        accepted_by_kind: Counter[str] = Counter()
        rejected_by_kind: Counter[str] = Counter()
        rejected_by_reason: Counter[str] = Counter()
        capacity_skipped_by_kind: Counter[str] = Counter()
        capacity_skipped_by_reason: Counter[str] = Counter()
        scope_blocked_by_kind: Counter[str] = Counter()
        scope_blocked_by_reason: Counter[str] = Counter()
        metrics = MediaDiscoveryMetrics(
            host_normalizer=self._host_normalizer,
        )
        for selected_task, decision in zip(tasks, decisions, strict=False):
            is_accepted = bool(getattr(decision, "accepted", False))
            reason = getattr(decision, "reason", None)

            is_capacity_skipped = (
                not is_accepted and reason in _CAPACITY_REASONS
            )
            is_scope_blocked = not is_accepted and reason in _SCOPE_REASONS

            metric_reason = str(
                getattr(reason, "value", None)
                or reason
                or "scheduler_rejected"
            )

            metrics.record_admission_decision(
                task=selected_task,
                accepted=is_accepted,
                reason=metric_reason,
                capacity_skipped=is_capacity_skipped,
                scope_blocked=is_scope_blocked,
            )

            if is_accepted:
                accepted += 1
                accepted_task = getattr(decision, "task", None)
                accepted_kind = str(
                    getattr(accepted_task, "kind", "unknown") or "unknown"
                )
                accepted_by_kind[accepted_kind] += 1
                continue

            rejected_kind = str(
                getattr(selected_task, "kind", "unknown") or "unknown"
            )

            if is_capacity_skipped:
                capacity_skipped += 1
                capacity_skipped_by_kind[rejected_kind] += 1
                capacity_skipped_by_reason[reason_key(reason)] += 1
                continue

            if is_scope_blocked:
                scope_blocked += 1
                scope_blocked_by_kind[rejected_kind] += 1
                scope_blocked_by_reason[reason_key(reason)] += 1
                continue

            rejected_by_reason[reason_key(reason)] += 1
            rejected_by_kind[rejected_kind] += 1
            if reason == ScheduleDecisionReason.URL_FILTERED:
                scheduler_filtered += 1
            else:
                scheduler_rejected += 1
        return AdmissionCounts(
            accepted=accepted,
            scheduler_filtered=scheduler_filtered,
            scheduler_rejected=scheduler_rejected,
            accepted_by_kind=accepted_by_kind,
            rejected_by_kind=rejected_by_kind,
            rejected_by_reason=rejected_by_reason,
            metrics=metrics,
            capacity_skipped=capacity_skipped,
            capacity_skipped_by_kind=capacity_skipped_by_kind,
            capacity_skipped_by_reason=capacity_skipped_by_reason,
            scope_blocked=scope_blocked,
            scope_blocked_by_kind=scope_blocked_by_kind,
            scope_blocked_by_reason=scope_blocked_by_reason,
        )

    def log_admission_summary(
        self,
        *,
        parent_url: str,
        selection: Any,
        budget: Any,
        submitted_count: int,
        admission: AdmissionCounts,
        discovery_rejected_by_kind: Counter[str],
        discovery_rejected_by_reason: Counter[str],
        merged_rejected_by_kind: Counter[str],
        merged_rejected_by_reason: Counter[str],
    ) -> None:
        target_kinds = {
            str(kind).strip().lower()
            for kind, missing in budget.coverage_missing_by_kind.items()
            if int(missing) > 0
            and str(kind).strip().lower()
            in {"image", "audio", "video", "document"}
        }
        target_kind_admitted = {
            kind: admission.accepted_by_kind.get(kind, 0)
            for kind in sorted(target_kinds)
        }
        self._logger.info(
            "asset_admission_summary",
            url=parent_url,
            selected_before_admission=len(selection.tasks),
            submitted_to_scheduler=submitted_count,
            accepted_by_kind=dict(sorted(admission.accepted_by_kind.items())),
            target_kind_admitted=target_kind_admitted,
            rejected_by_kind=dict(sorted(merged_rejected_by_kind.items())),
            discovery_rejected_by_kind=dict(
                sorted(discovery_rejected_by_kind.items())
            ),
            scheduler_rejected_by_kind=dict(
                sorted(admission.rejected_by_kind.items())
            ),
            capacity_skipped_by_kind=dict(
                sorted(admission.capacity_skipped_by_kind.items())
            ),
            capacity_skipped_by_reason=dict(
                sorted(admission.capacity_skipped_by_reason.items())
            ),
            scope_blocked_by_kind=dict(
                sorted(admission.scope_blocked_by_kind.items())
            ),
            scope_blocked_by_reason=dict(
                sorted(admission.scope_blocked_by_reason.items())
            ),
            rejected_by_reason=dict(sorted(merged_rejected_by_reason.items())),
            discovery_rejected_by_reason=dict(
                sorted(discovery_rejected_by_reason.items())
            ),
            scheduler_rejected_by_reason=dict(
                sorted(admission.rejected_by_reason.items())
            ),
        )

    @staticmethod
    def build_result_metrics(
        *,
        selection: Any,
        budget: Any,
        selection_metrics: Mapping[str, object],
        admission: AdmissionCounts,
    ) -> dict[str, int]:
        result = {
            "discovered": selection.discovered_count,
            "scheduled": admission.accepted,
            "filtered": selection.filtered_count
            + admission.scheduler_filtered,
            "rejected": admission.scheduler_rejected,
            "scope_blocked": (
                getattr(selection, "scope_blocked_count", 0)
                + admission.scope_blocked
            ),
            "capacity_skipped": (
                getattr(selection, "capacity_skipped_count", 0)
                + admission.capacity_skipped
            ),
            "truncated": selection.truncated_count,
            "duplicates": selection.duplicate_count,
            "discovery_scan_budget": budget.discovery_scan_budget,
        }
        metric_groups = (
            selection_metrics,
            admission.metrics.as_payload(),
        )
        for metric_group in metric_groups:
            for group_name, counts in metric_group.items():
                if not isinstance(counts, dict):
                    continue
                for key, count in counts.items():
                    if isinstance(count, int):
                        result[f"{group_name}_{key}"] = count
        return result

    def log_discovery_summary(
        self,
        *,
        parent_url: str,
        selection: Any,
        metrics: Mapping[str, object],
        rejected_by_reason: Counter[str],
    ) -> None:
        self._logger.info(
            "page_asset_discovery_summary",
            url=parent_url,
            discovered=selection.discovered_count,
            selected=len(selection.tasks),
            filtered=selection.filtered_count,
            truncated=selection.truncated_count,
            assets_discovered_by_kind=metrics.get("discovered_by_kind", {}),
            assets_selected_by_kind=metrics.get("selected_by_kind", {}),
            assets_truncated_by_kind=metrics.get("truncated_by_kind", {}),
            assets_rejected_by_reason=dict(sorted(rejected_by_reason.items())),
        )

    def log_discovery_details(
        self,
        *,
        parent_url: str,
        selection: Any,
        budget: Any,
        metrics: Mapping[str, object],
        rejected_by_reason: Counter[str],
        rejected_by_kind: Counter[str],
    ) -> None:
        self._logger.debug(
            "page_asset_discovery_details",
            url=parent_url,
            selection_budget_missing_by_kind=dict(
                budget.coverage_missing_by_kind
            ),
            coverage_snapshot_version=(budget.coverage_snapshot_version),
            coverage_snapshot_captured_at_monotonic=(
                budget.coverage_snapshot_captured_at_monotonic
            ),
            coverage_snapshot_source=budget.coverage_snapshot_source,
            discovered=selection.discovered_count,
            selected=len(selection.tasks),
            filtered=selection.filtered_count,
            truncated=selection.truncated_count,
            assets_discovered_by_kind=metrics.get("discovered_by_kind", {}),
            assets_selected_by_kind=metrics.get("selected_by_kind", {}),
            assets_truncated_by_kind=metrics.get("truncated_by_kind", {}),
            assets_selected_by_reason=metrics.get("selected_by_reason", {}),
            assets_target_kind_discovered=metrics.get(
                "target_kind_discovered", {}
            ),
            assets_target_kind_selected=metrics.get(
                "target_kind_selected", {}
            ),
            assets_selected_fallback_page=metrics.get(
                "selected_fallback_page", {}
            ),
            assets_selection_ratios=metrics.get("selection_ratios", {}),
            assets_duplicate_by_reason=metrics.get("duplicate_by_reason", {}),
            assets_filtered_by_reason=metrics.get("filtered_by_reason", {}),
            assets_truncated_by_reason=metrics.get("truncated_by_reason", {}),
            assets_rejected_by_reason=dict(sorted(rejected_by_reason.items())),
            assets_rejected_by_kind=dict(sorted(rejected_by_kind.items())),
            metrics=metrics,
        )

    def collect_rejected(
        self,
        *,
        parent_url: str,
        selection: Any,
        decisions: list[Any],
    ) -> list[tuple[CrawlTask, str]]:
        """Collect rejected tasks and emit optional reporter records.

        Capacity misses (a temporarily full host/frontier) and crawl-scope
        misses (host outside the current frontier scope) are not content
        rejections and are excluded from the rejected-assets output.
        """
        rejected_assets: list[tuple[CrawlTask, str]] = [
            (filtered_task, "url_filter_rejected")
            for filtered_task in getattr(selection, "filtered_tasks", ()) or ()
        ]

        rejected_assets.extend(
            (selected_task, reason_key(decision.reason))
            for selected_task, decision in zip(
                selection.tasks, decisions, strict=False
            )
            if not getattr(decision, "accepted", False)
            and getattr(decision, "reason", None) not in _CAPACITY_REASONS
            and getattr(decision, "reason", None) not in _SCOPE_REASONS
        )

        if self._rejected_discovery_reporter is not None:
            for filtered_task in (
                getattr(selection, "filtered_tasks", ()) or ()
            ):
                self._rejected_discovery_reporter.record(
                    url=filtered_task.url,
                    kind=str(filtered_task.kind),
                    reason="url_filter_rejected",
                    parent_url=parent_url,
                )

            for selected_task, decision in zip(
                selection.tasks, decisions, strict=False
            ):
                if getattr(decision, "accepted", False):
                    continue
                if getattr(decision, "reason", None) in _CAPACITY_REASONS:
                    continue
                if getattr(decision, "reason", None) in _SCOPE_REASONS:
                    continue
                self._rejected_discovery_reporter.record(
                    url=selected_task.url,
                    kind=str(selected_task.kind),
                    reason=reason_key(getattr(decision, "reason", None)),
                    parent_url=parent_url,
                    context={
                        "selection_reason": (
                            getattr(
                                selected_task.context, "selection_reason", None
                            )
                            if selected_task.context
                            else None
                        ),
                    },
                )

        return rejected_assets
