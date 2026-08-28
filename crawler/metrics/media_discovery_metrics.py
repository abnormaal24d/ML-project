"""Per-kind media discovery counters for page discovery decisions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from crawler.classification.media_kind import MediaKind

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.governance.domains.host_normalizer import HostNormalizer


@dataclass(slots=True)
class MediaDiscoveryMetrics:
    """Counters that show where discovered media leaves the pipeline."""

    host_normalizer: HostNormalizer
    discovered_by_kind: dict[str, int] = field(default_factory=dict)
    duplicate_by_kind: dict[str, int] = field(default_factory=dict)
    duplicate_by_reason: dict[str, int] = field(default_factory=dict)
    filtered_by_kind: dict[str, int] = field(default_factory=dict)
    filtered_by_reason: dict[str, int] = field(default_factory=dict)
    selected_by_kind: dict[str, int] = field(default_factory=dict)
    selected_by_reason: dict[str, int] = field(default_factory=dict)
    truncated_by_kind: dict[str, int] = field(default_factory=dict)
    truncated_by_reason: dict[str, int] = field(default_factory=dict)
    accepted_by_kind: dict[str, int] = field(default_factory=dict)
    scheduler_rejected_by_kind: dict[str, int] = field(default_factory=dict)
    capacity_skipped_by_kind: dict[str, int] = field(default_factory=dict)
    capacity_skipped_by_reason: dict[str, int] = field(default_factory=dict)
    scope_blocked_by_kind: dict[str, int] = field(default_factory=dict)
    scope_blocked_by_reason: dict[str, int] = field(default_factory=dict)
    accepted_by_source_type: dict[str, int] = field(default_factory=dict)
    scheduler_rejected_by_source_type: dict[str, int] = field(
        default_factory=dict
    )
    accepted_by_domain: dict[str, int] = field(default_factory=dict)
    scheduler_rejected_by_domain: dict[str, int] = field(default_factory=dict)
    target_kind_discovered: dict[str, int] = field(default_factory=dict)
    target_kind_selected: dict[str, int] = field(default_factory=dict)
    selected_fallback_page: dict[str, int] = field(default_factory=dict)
    selected_duplicate_free: dict[str, int] = field(default_factory=dict)
    selection_ratios: dict[str, float] = field(default_factory=dict)

    def record_discovered(self, *, task: CrawlTask) -> None:
        self._increment(self.discovered_by_kind, kind=task.kind)

    def record_duplicate(
        self,
        *,
        task: CrawlTask,
        reason: str = "same_page_duplicate",
    ) -> None:
        self._increment(self.duplicate_by_kind, kind=task.kind)
        self._increment_reason(
            self.duplicate_by_reason, task=task, reason=reason
        )

    def record_filtered(
        self,
        *,
        task: CrawlTask,
        reason: str = "url_filter",
    ) -> None:
        self._increment(self.filtered_by_kind, kind=task.kind)
        self._increment_reason(
            self.filtered_by_reason, task=task, reason=reason
        )

    def record_selected(
        self,
        *,
        task: CrawlTask,
        reason: str = "selected",
    ) -> None:
        self._increment(self.selected_by_kind, kind=task.kind)
        self._increment_reason(
            self.selected_by_reason, task=task, reason=reason
        )

    def record_selected_many(
        self,
        *,
        tasks: Iterable[CrawlTask],
        reason: str = "selected",
    ) -> None:
        for task in tasks:
            self.record_selected(task=task, reason=reason)

    def record_truncated_many(self, *, tasks: Iterable[CrawlTask]) -> None:
        for task in tasks:
            self._increment(self.truncated_by_kind, kind=task.kind)
            self._increment_reason(
                self.truncated_by_reason,
                task=task,
                reason="selection_truncated",
            )

    def record_capacity_skipped_many(
        self,
        *,
        tasks: Iterable[CrawlTask],
    ) -> None:
        for task in tasks:
            self.record_capacity_skipped(task=task)

    def record_capacity_skipped(
        self,
        *,
        task: CrawlTask,
        reason: str = "frontier_capacity",
    ) -> None:
        self._increment(self.capacity_skipped_by_kind, kind=task.kind)
        self._increment_reason(
            self.capacity_skipped_by_reason,
            task=task,
            reason=reason,
        )

    def record_scope_blocked(
        self,
        *,
        task: CrawlTask,
        reason: str = "crawl_scope_blocked",
    ) -> None:
        self._increment(self.scope_blocked_by_kind, kind=task.kind)
        self._increment_reason(
            self.scope_blocked_by_reason,
            task=task,
            reason=reason,
        )

    def record_admission_decision(
        self,
        *,
        task: CrawlTask,
        accepted: bool,
        reason: str = "scheduler_rejected",
        capacity_skipped: bool = False,
        scope_blocked: bool = False,
    ) -> None:
        domain = (
            self.host_normalizer.normalize(urlparse(task.url).hostname)
            or "unknown"
        )
        source_type = (
            str(task.source_type or "unknown").strip().lower() or "unknown"
        )
        if accepted:
            self._increment(self.accepted_by_kind, kind=task.kind)
            self._increment_key(self.accepted_by_source_type, key=source_type)
            self._increment_key(self.accepted_by_domain, key=domain)
            return

        if capacity_skipped:
            self._increment(self.capacity_skipped_by_kind, kind=task.kind)
            self._increment_reason(
                self.capacity_skipped_by_reason,
                task=task,
                reason=reason,
            )
            return

        if scope_blocked:
            self._increment(self.scope_blocked_by_kind, kind=task.kind)
            self._increment_reason(
                self.scope_blocked_by_reason,
                task=task,
                reason=reason,
            )
            return

        self._increment(self.scheduler_rejected_by_kind, kind=task.kind)
        self._increment_key(
            self.scheduler_rejected_by_source_type,
            key=source_type,
        )
        self._increment_key(self.scheduler_rejected_by_domain, key=domain)
        self._increment_reason(
            self.truncated_by_reason,
            task=task,
            reason=f"scheduler:{reason}",
        )

    def mark_focus_targets(
        self,
        *,
        focus_kinds: Iterable[str | MediaKind],
    ) -> None:
        targets: set[str] = set()
        for kind in focus_kinds:
            try:
                parsed_kind = MediaKind.parse(kind)
            except (TypeError, ValueError):
                continue
            if parsed_kind is not MediaKind.PAGE:
                targets.add(parsed_kind.value)
        self.target_kind_discovered = {
            kind: self.discovered_by_kind.get(kind, 0)
            for kind in sorted(targets)
        }
        self.target_kind_selected = {
            kind: self.selected_by_kind.get(kind, 0)
            for kind in sorted(targets)
        }
        fallback_count = self.selected_by_reason.get("page:fallback_page", 0)
        self.selected_fallback_page = {"page": fallback_count}
        self.selected_duplicate_free = dict(
            sorted(self.selected_by_kind.items())
        )
        selected_total = sum(self.selected_by_kind.values())
        self.selection_ratios = {
            "fallback_page": (
                round(fallback_count / selected_total, 4)
                if selected_total
                else 0.0
            )
        }

    def as_flat_fields(self) -> dict[str, int]:
        fields: dict[str, int] = {}
        for prefix, values in (
            ("discovered", self.discovered_by_kind),
            ("duplicate", self.duplicate_by_kind),
            ("filtered", self.filtered_by_kind),
            ("selected", self.selected_by_kind),
            ("truncated", self.truncated_by_kind),
            ("accepted", self.accepted_by_kind),
            ("scheduler_rejected", self.scheduler_rejected_by_kind),
            ("capacity_skipped", self.capacity_skipped_by_kind),
            ("scope_blocked", self.scope_blocked_by_kind),
        ):
            for kind, count in sorted(values.items()):
                fields[f"{prefix}_{kind}"] = count
        return fields

    def as_payload(self) -> dict[str, dict[str, int] | dict[str, float]]:
        return {
            "discovered_by_kind": dict(
                sorted(self.discovered_by_kind.items())
            ),
            "duplicate_by_kind": dict(sorted(self.duplicate_by_kind.items())),
            "duplicate_by_reason": dict(
                sorted(self.duplicate_by_reason.items())
            ),
            "filtered_by_kind": dict(sorted(self.filtered_by_kind.items())),
            "filtered_by_reason": dict(
                sorted(self.filtered_by_reason.items())
            ),
            "selected_by_kind": dict(sorted(self.selected_by_kind.items())),
            "selected_by_reason": dict(
                sorted(self.selected_by_reason.items())
            ),
            "truncated_by_kind": dict(sorted(self.truncated_by_kind.items())),
            "truncated_by_reason": dict(
                sorted(self.truncated_by_reason.items())
            ),
            "accepted_by_kind": dict(sorted(self.accepted_by_kind.items())),
            "scheduler_rejected_by_kind": dict(
                sorted(self.scheduler_rejected_by_kind.items())
            ),
            "capacity_skipped_by_kind": dict(
                sorted(self.capacity_skipped_by_kind.items())
            ),
            "capacity_skipped_by_reason": dict(
                sorted(self.capacity_skipped_by_reason.items())
            ),
            "scope_blocked_by_kind": dict(
                sorted(self.scope_blocked_by_kind.items())
            ),
            "scope_blocked_by_reason": dict(
                sorted(self.scope_blocked_by_reason.items())
            ),
            "accepted_by_source_type": dict(
                sorted(self.accepted_by_source_type.items())
            ),
            "scheduler_rejected_by_source_type": dict(
                sorted(self.scheduler_rejected_by_source_type.items())
            ),
            "accepted_by_domain": dict(
                sorted(self.accepted_by_domain.items())
            ),
            "scheduler_rejected_by_domain": dict(
                sorted(self.scheduler_rejected_by_domain.items())
            ),
            "target_kind_discovered": dict(
                sorted(self.target_kind_discovered.items())
            ),
            "target_kind_selected": dict(
                sorted(self.target_kind_selected.items())
            ),
            "selected_fallback_page": dict(
                sorted(self.selected_fallback_page.items())
            ),
            "selected_duplicate_free": dict(
                sorted(self.selected_duplicate_free.items())
            ),
            "selection_ratios": dict(sorted(self.selection_ratios.items())),
        }

    @classmethod
    def empty_payload(
        cls,
        *,
        host_normalizer: HostNormalizer,
    ) -> dict[str, dict[str, int] | dict[str, float]]:
        return cls(host_normalizer=host_normalizer).as_payload()

    @staticmethod
    def _increment(counter: dict[str, int], *, kind: MediaKind) -> None:
        key = kind.value
        counter[key] = counter.get(key, 0) + 1

    @staticmethod
    def _increment_key(counter: dict[str, int], *, key: str) -> None:
        normalized = str(key or "unknown").strip().lower() or "unknown"
        counter[normalized] = counter.get(normalized, 0) + 1

    @staticmethod
    def _increment_reason(
        counter: dict[str, int],
        *,
        task: CrawlTask,
        reason: str,
    ) -> None:
        kind = task.kind.value
        reason_key = str(reason or "unknown").strip().lower() or "unknown"
        counter[f"{kind}:{reason_key}"] = (
            counter.get(f"{kind}:{reason_key}", 0) + 1
        )


def merge_discovery_metric_fields(
    *,
    base: Mapping[str, int],
    metrics: MediaDiscoveryMetrics,
) -> dict[str, int]:
    """Merge base counters with explicit per-kind discovery counters."""

    merged = dict(base)
    merged.update(metrics.as_flat_fields())
    return merged
