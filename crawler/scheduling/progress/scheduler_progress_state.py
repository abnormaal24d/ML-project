"""Mutable scheduler progress counters."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..admission.schedule_decision import (
    ScheduleDecision,
    ScheduleDecisionReason,
)

CounterState = dict[str, int]


@dataclass(frozen=True, slots=True)
class SchedulerProgressRestoreState:
    """Validated progress counters ready for an atomic restore commit."""

    accepted_total: int
    filtered_total: int
    rejected_total: int
    duplicate_total: int
    rejected_by_reason: tuple[tuple[str, int], ...]
    completed_by_outcome: tuple[tuple[str, int], ...]
    deferrals_by_reason: tuple[tuple[str, int], ...] = ()


@dataclass(slots=True)
class SchedulerProgressState:
    """Track lifetime scheduler counters independent of snapshot rendering."""

    accepted_total: int = 0
    filtered_total: int = 0
    rejected_total: int = 0
    duplicate_total: int = 0
    rejected_by_reason: CounterState = field(default_factory=dict)
    completed_by_outcome: CounterState = field(default_factory=dict)
    deferrals_by_reason: CounterState = field(default_factory=dict)
    drain_completed_baseline: int | None = None
    drain_accepted_baseline: int | None = None
    drain_reserved: int = 0

    def record_accepted_task(self) -> None:
        """Record one accepted scheduler task."""
        self.accepted_total += 1

    def record_rejected_decision(self, *, decision: ScheduleDecision) -> None:
        """Record a rejected scheduling decision."""
        if decision.accepted:
            return

        reason_key = decision.reason.value
        self.rejected_by_reason[reason_key] = (
            self.rejected_by_reason.get(reason_key, 0) + 1
        )

        if decision.reason == ScheduleDecisionReason.URL_FILTERED:
            self.filtered_total += 1
            return

        self.rejected_total += 1

        if decision.reason == ScheduleDecisionReason.DUPLICATE_URL:
            self.duplicate_total += 1

    def record_completed_outcome(self, *, outcome: str) -> None:
        """Record one completed task outcome."""
        outcome_key = _normalize_key(outcome)
        self.completed_by_outcome[outcome_key] = (
            self.completed_by_outcome.get(outcome_key, 0) + 1
        )

    def record_task_deferral(self, *, reason: str | None) -> None:
        """Record one deferred task outcome keyed by its reason."""
        reason_key = _normalize_key(reason or "")
        self.deferrals_by_reason[reason_key] = (
            self.deferrals_by_reason.get(reason_key, 0) + 1
        )

    @property
    def completed_total(self) -> int:
        """Return the total number of completed scheduler multimodal."""
        return sum(self.completed_by_outcome.values())

    def ensure_drain_budget_window(self) -> None:
        """Start the high-pressure drain accounting window if needed."""
        if (
            self.drain_completed_baseline is not None
            and self.drain_accepted_baseline is not None
        ):
            return

        self.drain_completed_baseline = self.completed_total
        self.drain_accepted_baseline = self.accepted_total
        self.drain_reserved = 0

    def reset_drain_budget_window(self) -> None:
        """Clear high-pressure drain accounting."""
        self.drain_completed_baseline = None
        self.drain_accepted_baseline = None
        self.drain_reserved = 0

    def available_drain_budget(self) -> int:
        """Return how many non-seed tasks may be admitted while draining."""
        self.ensure_drain_budget_window()
        completed_since_window = self.completed_total - int(
            self.drain_completed_baseline or 0
        )
        accepted_since_window = self.accepted_total - int(
            self.drain_accepted_baseline or 0
        )
        return max(
            0,
            completed_since_window
            - accepted_since_window
            - max(0, int(self.drain_reserved)),
        )

    def reserve_drain_budget(self, *, configured_cap: int) -> int:
        """Reserve a high-pressure drain budget slot for pending admission."""
        normalized_cap = max(0, int(configured_cap))
        if normalized_cap <= 0:
            return 0

        allowed = min(normalized_cap, self.available_drain_budget())
        self.drain_reserved += allowed
        return allowed

    def release_drain_budget_reservation(self, *, reserved: int) -> None:
        """Release a previously reserved drain budget slot."""
        self.drain_reserved = max(
            0,
            self.drain_reserved - max(0, int(reserved)),
        )

    def export_state(self) -> dict[str, object]:
        """Return a serializable progress-counter payload."""
        return {
            "accepted_total": self.accepted_total,
            "filtered_total": self.filtered_total,
            "rejected_total": self.rejected_total,
            "duplicate_total": self.duplicate_total,
            "rejected_by_reason": dict(
                sorted_counter_items(self.rejected_by_reason),
            ),
            "completed_by_outcome": dict(
                sorted_counter_items(self.completed_by_outcome),
            ),
            "deferrals_by_reason": dict(
                sorted_counter_items(self.deferrals_by_reason),
            ),
        }

    def restore_state(self, payload: dict[str, object]) -> None:
        """Restore progress counters from a checkpoint payload."""
        self.apply_restore_state(self.parse_restore_state(payload))

    @staticmethod
    def parse_restore_state(
        payload: dict[str, object],
    ) -> SchedulerProgressRestoreState:
        """Validate every persisted counter without mutating live state."""

        return SchedulerProgressRestoreState(
            accepted_total=_positive_int_from_mapping(
                payload,
                "accepted_total",
            ),
            filtered_total=_positive_int_from_mapping(
                payload,
                "filtered_total",
            ),
            rejected_total=_positive_int_from_mapping(
                payload,
                "rejected_total",
            ),
            duplicate_total=_positive_int_from_mapping(
                payload,
                "duplicate_total",
            ),
            rejected_by_reason=tuple(
                sorted(
                    _counter_from_mapping(
                        payload,
                        "rejected_by_reason",
                    ).items()
                )
            ),
            completed_by_outcome=tuple(
                sorted(
                    _counter_from_mapping(
                        payload,
                        "completed_by_outcome",
                    ).items()
                )
            ),
            deferrals_by_reason=tuple(
                sorted(
                    _counter_from_mapping(
                        payload,
                        "deferrals_by_reason",
                        default={},
                    ).items()
                )
            ),
        )

    def apply_restore_state(
        self,
        state: SchedulerProgressRestoreState,
    ) -> None:
        """Commit a previously validated progress-counter state."""

        self.accepted_total = state.accepted_total
        self.filtered_total = state.filtered_total
        self.rejected_total = state.rejected_total
        self.duplicate_total = state.duplicate_total
        self.rejected_by_reason = dict(state.rejected_by_reason)
        self.completed_by_outcome = dict(state.completed_by_outcome)
        self.deferrals_by_reason = dict(state.deferrals_by_reason)
        self.reset_drain_budget_window()

    def reset(self) -> None:
        """Reset all counters to their initial empty state."""
        self.accepted_total = 0
        self.filtered_total = 0
        self.rejected_total = 0
        self.duplicate_total = 0
        self.rejected_by_reason.clear()
        self.completed_by_outcome.clear()
        self.deferrals_by_reason.clear()
        self.reset_drain_budget_window()


def sorted_counter_items(counter: CounterState) -> tuple[tuple[str, int], ...]:
    """Return deterministic counter items for snapshots and checkpoints."""
    return tuple(sorted(counter.items()))


def _normalize_key(value: str) -> str:
    return value.strip().lower() or "unknown"


def _positive_int_from_mapping(payload: dict[str, object], key: str) -> int:
    if key not in payload:
        raise ValueError(f"progress counters missing {key}")
    raw_value = payload[key]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise TypeError(f"{key} must be an integer")
    if raw_value < 0:
        raise ValueError(f"{key} must be non-negative")
    return raw_value


def _counter_from_mapping(
    payload: dict[str, object],
    key: str,
    *,
    default: CounterState | None = None,
) -> CounterState:
    if key not in payload:
        if default is None:
            raise ValueError(f"progress counters missing {key}")
        return dict(default)
    raw_value = payload[key]
    if not isinstance(raw_value, dict):
        raise TypeError(f"{key} must be a dictionary")

    restored: CounterState = {}
    for raw_key, raw_count in raw_value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(f"{key} contains an invalid counter name")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise TypeError(f"{key}.{raw_key} must be an integer")
        if raw_count < 0:
            raise ValueError(f"{key}.{raw_key} must be non-negative")
        if raw_count > 0:
            restored[raw_key] = raw_count

    return restored
