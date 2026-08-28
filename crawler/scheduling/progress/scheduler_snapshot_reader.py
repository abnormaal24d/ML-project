"""Scheduler snapshot and queue-state query operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind

from ..queueing.host_eligibility_queue import coerce_monotonic_timestamp
from ..queueing.host_task_queue import PendingMap, max_pending_count

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class DiscoveryCapacitySnapshot:
    """Immutable scheduler-owned remaining frontier capacity for discovery.

    Keyed by ``(host, kind)`` with the number of ready tasks that may still be
    admitted per the effective pending limits at snapshot time. Hosts absent
    from the snapshot have no scheduler capacity knowledge; admission remains
    the authority for them.

    Discovery consumes this view and never interprets scheduler policy.
    """

    by_host_kind: Mapping[tuple[str | None, MediaKind], int]
    queue_size: int
    captured_at_monotonic: float

    def remaining(self, *, host: str | None, kind: MediaKind) -> int | None:
        """Return remaining ready capacity for one host and kind."""
        return self.by_host_kind.get((host, kind))


_DISCOVERY_CAPACITY_KINDS = (
    MediaKind.PAGE,
    MediaKind.FEED,
    MediaKind.IMAGE,
    MediaKind.AUDIO,
    MediaKind.VIDEO,
    MediaKind.DOCUMENT,
)


def build_discovery_capacity_snapshot(
    *,
    ready_pending: PendingMap,
    host_limit_fn: Callable[[MediaKind, str | None], int | None],
    kind_host_pending_fn: Callable[[str | None, MediaKind], int],
    queue_size: int,
    now: float,
) -> DiscoveryCapacitySnapshot:
    """Build the discovery capacity view from scheduler-owned state.

    ``host_limit_fn(kind, host)`` resolves the effective pending limit and
    ``kind_host_pending_fn(host, kind)`` the ready-queue pending count; both
    are injected so this builder never interprets scheduler policy itself.
    """

    by_host_kind: dict[tuple[str | None, MediaKind], int] = {}
    for host, _pending in (ready_pending or {}).items():
        for kind in _DISCOVERY_CAPACITY_KINDS:
            limit = host_limit_fn(kind, host)
            if limit is None:
                continue
            kind_pending = kind_host_pending_fn(host, kind)
            by_host_kind[(host, kind)] = max(
                0,
                int(limit) - int(kind_pending),
            )
    return DiscoveryCapacitySnapshot(
        by_host_kind=by_host_kind,
        queue_size=queue_size,
        captured_at_monotonic=now,
    )


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    """Immutable scheduler queue, delayed queue, and inflight state."""

    queued: int
    pending_hosts: int
    max_pending_per_host: int
    inflight: int

    delayed_queued: int = 0
    delayed_pending_hosts: int = 0
    delayed_max_pending_per_host: int = 0

    total_queued: int = 0
    total_pending_hosts: int = 0
    total_max_pending_per_host: int = 0

    next_delayed_ready_at: float | None = None
    next_delayed_ready_in_seconds: float | None = None
    delayed_wait_seconds_total: float = 0.0
    average_delayed_wait_seconds: float | None = None
    delayed_ready_count: int = 0

    max_inflight_per_host: int = 0
    runnable_slots: int = 0

    def __post_init__(self) -> None:
        """Normalize derived totals and validate snapshot counters."""
        self._validate_non_negative_int("queued", self.queued)
        self._validate_non_negative_int("pending_hosts", self.pending_hosts)
        self._validate_non_negative_int(
            "max_pending_per_host",
            self.max_pending_per_host,
        )
        self._validate_non_negative_int("inflight", self.inflight)
        self._validate_non_negative_int("delayed_queued", self.delayed_queued)
        self._validate_non_negative_int(
            "delayed_pending_hosts",
            self.delayed_pending_hosts,
        )
        self._validate_non_negative_int(
            "delayed_max_pending_per_host",
            self.delayed_max_pending_per_host,
        )
        self._validate_non_negative_int(
            "delayed_ready_count",
            self.delayed_ready_count,
        )
        self._validate_non_negative_int(
            "max_inflight_per_host",
            self.max_inflight_per_host,
        )
        self._validate_non_negative_int(
            "runnable_slots",
            self.runnable_slots,
        )

        normalized_total_queued = self.queued + self.delayed_queued
        normalized_total_pending_hosts = self.total_pending_hosts or (
            self.pending_hosts + self.delayed_pending_hosts
        )
        normalized_total_max_pending_per_host = (
            self.total_max_pending_per_host
            or max(
                self.max_pending_per_host,
                self.delayed_max_pending_per_host,
            )
        )

        self._validate_non_negative_int(
            "total_queued",
            normalized_total_queued,
        )
        self._validate_non_negative_int(
            "total_pending_hosts",
            normalized_total_pending_hosts,
        )
        self._validate_non_negative_int(
            "total_max_pending_per_host",
            normalized_total_max_pending_per_host,
        )

        if self.next_delayed_ready_in_seconds is not None:
            if self.next_delayed_ready_in_seconds < 0:
                raise ValueError(
                    "next_delayed_ready_in_seconds must be >= 0 when provided"
                )
        if self.delayed_wait_seconds_total < 0:
            raise ValueError("delayed_wait_seconds_total must be >= 0")
        if (
            self.average_delayed_wait_seconds is not None
            and self.average_delayed_wait_seconds < 0
        ):
            raise ValueError(
                "average_delayed_wait_seconds must be >= 0 when provided"
            )

        object.__setattr__(self, "total_queued", normalized_total_queued)
        object.__setattr__(
            self,
            "total_pending_hosts",
            normalized_total_pending_hosts,
        )
        object.__setattr__(
            self,
            "total_max_pending_per_host",
            normalized_total_max_pending_per_host,
        )

    @staticmethod
    def _validate_non_negative_int(name: str, value: int) -> None:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value!r}")


if TYPE_CHECKING:
    import asyncio

    from crawler.scheduling.admission.scheduler_task_admitter import (
        SchedulerTaskAdmitter,
    )

    from ..admission.scheduler_frontier import SchedulerFrontier
    from ..progress.active_task_registry import ActiveTaskRegistry
    from ..progress.scheduler_backlog_reader import SchedulerBacklogReader
    from ..progress.scheduler_progress_state import SchedulerProgressState
    from ..queueing.delayed_task_queue import DelayedTaskQueue
    from ..queueing.host_eligibility_queue import HostEligibilityQueue
    from ..queueing.host_task_queue import HostTaskQueue


def compute_runnable_slots(
    *,
    ready_pending: PendingMap,
    inflight_count_by_host: Mapping[str, int],
    max_inflight_per_host: int,
    host_eligibility_queue: HostEligibilityQueue | None = None,
    now: float | None = None,
) -> int:
    """Count ready tasks that can actually start fetching this instant.

    Per host the runnable count is ``min(pending, free_slots)`` where
    ``free_slots = max_inflight_per_host - inflight``. When an eligibility
    queue is provided, hosts whose cached eligibility timestamp is in the
    future (rate limited, suppressed, or inflight-blocked) are excluded, so
    the metric vouches for full governance-level dispatchability instead of
    only the hard inflight cap. Hosts without a cached eligibility entry
    count as runnable until the dispatcher computes them.
    """

    if max_inflight_per_host <= 0:
        return 0

    current_time = (
        coerce_monotonic_timestamp(now)
        if host_eligibility_queue is not None
        else None
    )

    total = 0
    for host, pending in ready_pending.items():
        if pending <= 0:
            continue
        inflight = 0 if host is None else inflight_count_by_host.get(host, 0)
        free_slots = max(0, max_inflight_per_host - inflight)
        if free_slots <= 0:
            continue

        if host_eligibility_queue is not None and current_time is not None:
            eligible_at = host_eligibility_queue.next_eligible_at(host)
            if eligible_at is not None and eligible_at > current_time:
                continue

        total += min(pending, free_slots)
    return total


def build_scheduler_snapshot(
    *,
    host_queue: HostTaskQueue,
    delayed_queue: DelayedTaskQueue,
    active_registry: ActiveTaskRegistry,
    ready_pending: PendingMap,
    delayed_pending: PendingMap,
    total_pending: PendingMap,
    now: float,
    inflight_count_by_host: Mapping[str, int],
    max_inflight_per_host: int,
    host_eligibility_queue: HostEligibilityQueue | None = None,
) -> SchedulerSnapshot:
    ready_queued = host_queue.queue_size
    delayed_queued = delayed_queue.queue_size

    return SchedulerSnapshot(
        queued=ready_queued,
        pending_hosts=len(ready_pending),
        max_pending_per_host=max_pending_count(ready_pending),
        inflight=active_registry.count,
        delayed_queued=delayed_queued,
        delayed_pending_hosts=len(delayed_pending),
        delayed_max_pending_per_host=max_pending_count(delayed_pending),
        total_queued=ready_queued + delayed_queued,
        total_pending_hosts=len(total_pending),
        total_max_pending_per_host=max_pending_count(total_pending),
        next_delayed_ready_at=delayed_queue.peek_next_ready_at(),
        next_delayed_ready_in_seconds=(
            delayed_queue.next_ready_in_seconds(now=now)
        ),
        delayed_wait_seconds_total=delayed_queue.wait_seconds_total(now=now),
        average_delayed_wait_seconds=delayed_queue.average_wait_seconds(
            now=now
        ),
        delayed_ready_count=delayed_queue.delayed_ready_count(now=now),
        max_inflight_per_host=max_inflight_per_host,
        runnable_slots=compute_runnable_slots(
            ready_pending=ready_pending,
            inflight_count_by_host=inflight_count_by_host,
            max_inflight_per_host=max_inflight_per_host,
            host_eligibility_queue=host_eligibility_queue,
            now=now,
        ),
    )


class SchedulerSnapshotReader:
    """Expose scheduler backlog snapshots and queue metrics."""

    def __init__(
        self,
        *,
        condition: asyncio.Condition,
        host_queue: HostTaskQueue,
        delayed_queue: DelayedTaskQueue,
        active_registry: ActiveTaskRegistry,
        backlog_reader: SchedulerBacklogReader,
        frontier_service: SchedulerFrontier,
        task_admitter: SchedulerTaskAdmitter,
        progress_state: SchedulerProgressState,
        max_inflight_per_host: int,
        host_eligibility_queue: HostEligibilityQueue | None = None,
    ) -> None:
        self._condition = condition
        self._host_queue = host_queue
        self._delayed_queue = delayed_queue
        self._active_registry = active_registry
        self._backlog_reader = backlog_reader
        self._frontier_service = frontier_service
        self._task_admitter = task_admitter
        self._progress_state = progress_state
        self._max_inflight_per_host = max(0, int(max_inflight_per_host))
        self._host_eligibility_queue = host_eligibility_queue

    def snapshot(self) -> SchedulerSnapshot:
        now = monotonic()
        ready_pending, delayed_pending, total_pending = (
            self._backlog_reader.pending_maps()
        )

        return build_scheduler_snapshot(
            host_queue=self._host_queue,
            delayed_queue=self._delayed_queue,
            active_registry=self._active_registry,
            ready_pending=ready_pending,
            delayed_pending=delayed_pending,
            total_pending=total_pending,
            now=now,
            inflight_count_by_host=self._active_registry.inflight_count_by_host,
            max_inflight_per_host=self._max_inflight_per_host,
            host_eligibility_queue=self._host_eligibility_queue,
        )

    def discovery_capacity_snapshot(self) -> DiscoveryCapacitySnapshot:
        """Return scheduler-owned remaining frontier capacity for discovery."""

        now = monotonic()
        ready_pending = self._host_queue.pending_count_by_host()
        queue_size = self._frontier_service.queue_size_total()
        return build_discovery_capacity_snapshot(
            ready_pending=ready_pending,
            host_limit_fn=lambda kind, host: (
                self._task_admitter.host_pending_limit(
                    kind=kind,
                    host=host,
                    queue_size=queue_size,
                )
            ),
            kind_host_pending_fn=lambda host, kind: (
                self._backlog_reader.kind_host_pending(
                    host=host,
                    kind=kind,
                )
            ),
            queue_size=queue_size,
            now=now,
        )

    async def discovery_drain_budget(
        self,
        *,
        configured_cap: int,
        force: bool = False,
    ) -> int:
        """Return the current high-pressure discovery budget."""
        normalized_cap = max(0, int(configured_cap))
        if normalized_cap <= 0:
            return 0

        async with self._condition:
            pressure_state = self._task_admitter.queue_pressure_state(
                queue_size=self._frontier_service.queue_size_total(),
            )
            if pressure_state == "critical":
                self._progress_state.reset_drain_budget_window()
                return 0
            if pressure_state != "high" and not force:
                self._progress_state.reset_drain_budget_window()
                return normalized_cap

            self._progress_state.ensure_drain_budget_window()
            return min(
                normalized_cap,
                self._progress_state.available_drain_budget(),
            )

    def queue_size(self) -> int:
        return self._host_queue.queue_size

    def pending_host_count(self) -> int:
        return len(self._host_queue.pending_count_by_host())

    def max_pending_per_host(self) -> int:
        return max_pending_count(self._host_queue.pending_count_by_host())

    def inflight_count(self) -> int:
        return self._active_registry.count
