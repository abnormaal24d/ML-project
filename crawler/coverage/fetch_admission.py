"""Coverage-aware fetch admission gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from config.coverage.settings import CoverageSettings
from config.validation.coverage_settings import nonnegative_int, normalize_kind
from crawler.processing.outcomes.processor_outcome import ProcessorOutcome

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask


class CoverageTargetStatus(Protocol):
    """Coverage API needed for pre-fetch target checks."""

    targets: dict[str, int]

    def target_met(self, *, kind: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class CoverageFetchGate:
    """Drop fetch tasks for configured kinds whose target is already met."""

    settings: CoverageSettings
    coverage_tracker: CoverageTargetStatus | None

    def outcome_for(self, *, task: CrawlTask) -> ProcessorOutcome | None:
        tracker = self.coverage_tracker
        if tracker is None:
            return None

        requested_kind = normalize_kind(task.kind)
        if requested_kind not in set(self.settings.kinds.gated_fetch_kinds):
            return None

        target_kind = self.settings.kinds.modality_to_media_kind.get(
            requested_kind,
            requested_kind,
        )
        if target_kind not in set(self.settings.kinds.media_kinds):
            return None

        target_count = nonnegative_int(tracker.targets.get(target_kind, 0))
        if target_count <= 0:
            return None

        if not tracker.target_met(kind=target_kind):
            return None

        return ProcessorOutcome.dropped(
            stage="fetch_admission",
            reason="coverage_target_already_met",
            counts_toward_task_retry_budget=False,
            terminal_eligible=True,
        )
