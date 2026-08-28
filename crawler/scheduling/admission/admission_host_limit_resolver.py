"""Per-host pending limits for scheduler task admission."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from crawler.scheduling.admission.admission_context import AdmissionContext
from crawler.scheduling.admission.admission_pressure_rules import (
    AdmissionPressureRules,
)
from crawler.scheduling.admission.schedule_decision import (
    ScheduleDecisionReason,
)
from crawler.scheduling.scheduling_value_parser import coerce_float, coerce_int

if TYPE_CHECKING:
    from config.collection.discovery import SchedulingSettings
    from crawler.classification.media_kind import MediaKind
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.scheduling.host_control.host_advice_tracker import (
        HostAdviceTracker,
    )
    from crawler.scheduling.host_control.host_budget_tracker import (
        HostBudgetTracker,
    )


class AdmissionHostLimitResolver:
    """Resolve effective per-host pending limits and crawl budgets."""

    def __init__(
        self,
        *,
        settings: SchedulingSettings,
        host_advice_tracker: HostAdviceTracker,
        host_budget_tracker: HostBudgetTracker | None,
        pressure_rules: AdmissionPressureRules,
    ) -> None:
        self._settings = settings
        self._host_advice_tracker = host_advice_tracker
        self._host_budget_tracker = host_budget_tracker
        self._pressure_rules = pressure_rules

    def pending_for_limit(self, *, ctx: AdmissionContext) -> int:
        if any(
            self._kind_pending_limit(kind=ctx.task.kind, limits=limits)
            is not None
            for limits in (
                self._settings.max_pending_per_host_by_kind,
                self._settings.max_pending_per_host_by_kind_under_pressure,
                self._settings.max_pending_per_host_by_kind_critical,
            )
        ):
            return ctx.kind_host_pending
        return ctx.host_pending

    def effective_max_pending_per_host(
        self,
        *,
        task: CrawlTask,
        host: str | None,
        queue_size: int,
        use_host_advice: bool = True,
    ) -> int | None:
        if task.source_type == "seed":
            return None
        return self._effective_max_pending_limit(
            kind=task.kind,
            host=host,
            queue_size=queue_size,
            use_host_advice=use_host_advice,
        )

    def effective_max_pending_for_kind(
        self,
        *,
        kind: MediaKind,
        host: str | None,
        queue_size: int,
        use_host_advice: bool = True,
    ) -> int | None:
        """Return the pending limit for a kind without a concrete task.

        Used by scheduler-owned discovery capacity views; discovery never
        interprets policies itself, it only consumes the resolved limit.
        """

        return self._effective_max_pending_limit(
            kind=kind,
            host=host,
            queue_size=queue_size,
            use_host_advice=use_host_advice,
        )

    def _effective_max_pending_limit(
        self,
        *,
        kind: MediaKind,
        host: str | None,
        queue_size: int,
        use_host_advice: bool,
    ) -> int | None:
        kind_limit = self._kind_pending_limit(
            kind=kind,
            limits=self._settings.max_pending_per_host_by_kind,
        )
        if kind_limit is not None:
            base_limit: int | None = kind_limit
        else:
            configured_limit = coerce_int(self._settings.max_pending_per_host)
            if configured_limit is None or configured_limit < 0:
                base_limit = None
            else:
                base_limit = int(configured_limit)

        pressure_limit = self._host_limit(kind=kind, queue_size=queue_size)
        if pressure_limit is not None:
            base_limit = (
                pressure_limit
                if base_limit is None
                else min(base_limit, pressure_limit)
            )

        if host is None or not use_host_advice:
            return base_limit

        advice_entry = self._host_advice_tracker.get(host)
        if advice_entry is None:
            return base_limit

        discovery_factor = self._coerce_discovery_factor(
            advice_entry.advice.discovery_factor,
        )

        if discovery_factor == 0.0:
            return 0

        if base_limit is None:
            return None

        if base_limit == 0:
            return 0

        return max(1, int(round(base_limit * discovery_factor)))

    def crawl_budget_rejection_reason(
        self,
        *,
        task: CrawlTask,
        host: str | None,
    ) -> ScheduleDecisionReason | None:
        if task.source_type == "seed" or host is None:
            return None

        if self._host_budget_tracker is None:
            return None

        if not self._host_budget_tracker.host_budget_exhausted(host):
            return None

        return ScheduleDecisionReason.CRAWL_BUDGET_EXHAUSTED

    @staticmethod
    def _kind_pending_limit(
        *,
        kind: MediaKind,
        limits: dict[str, int],
    ) -> int | None:
        raw_limit = limits.get(kind.value)
        limit = coerce_int(raw_limit)
        if limit is None or limit < 0:
            return None

        return int(limit)

    def _host_limit(
        self,
        *,
        kind: MediaKind,
        queue_size: int,
    ) -> int | None:
        if self._pressure_rules.is_critical_backpressure(
            queue_size=queue_size
        ):
            kind_limit = self._kind_pending_limit(
                kind=kind,
                limits=self._settings.max_pending_per_host_by_kind_critical,
            )
            if kind_limit is not None:
                return kind_limit
            return coerce_int(self._settings.max_pending_per_host_critical)

        if self._pressure_rules.is_high_backpressure(queue_size=queue_size):
            kind_limit = self._kind_pending_limit(
                kind=kind,
                limits=(
                    self._settings.max_pending_per_host_by_kind_under_pressure
                ),
            )
            if kind_limit is not None:
                return kind_limit
            return coerce_int(
                self._settings.max_pending_per_host_under_pressure,
            )

        return None

    @staticmethod
    def _coerce_discovery_factor(value: object) -> float:
        discovery_factor = coerce_float(value, allow_bool=True)
        if discovery_factor is None or not isfinite(discovery_factor):
            return 1.0
        return max(0.0, min(1.0, discovery_factor))
