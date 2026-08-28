"""Immutable, point-in-time worker-pool observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crawler.worker.pool.worker_task_counters import WorkerTaskCounters


@dataclass(frozen=True, slots=True)
class ActiveWorkerTaskSnapshot:
    """Task identity and elapsed busy time for one busy worker."""

    worker_id: int
    task_id: str | None
    url: str | None
    kind: str | None
    busy_seconds: float
    retiring: bool


@dataclass(frozen=True, slots=True)
class WorkerPoolSnapshot:
    """Consistent live state and cumulative counters for one read moment."""

    size: int
    effective_worker_count: int
    retiring_worker_count: int
    busy_worker_count: int
    idle_worker_count: int
    completed_task_count: int
    failure_count: int
    non_fatal_timeout_count: int
    retry_exhausted_count: int
    average_processing_seconds: float
    longest_busy_seconds: float
    active_tasks: tuple[ActiveWorkerTaskSnapshot, ...]
    root_seeds_total: int = 0
    root_seeds_succeeded: int = 0
    root_seeds_transient_failed: int = 0
    root_seeds_governance_blocked: int = 0


def build_worker_pool_snapshot(
    *,
    workers: Mapping[int, Any],
    task_counters: WorkerTaskCounters,
    now: float,
) -> WorkerPoolSnapshot:
    """Read worker state and counters once for the supplied monotonic time."""

    active_busy = 0
    active_idle = 0
    retiring_busy = 0
    retiring_idle = 0
    longest_busy_seconds = 0.0
    active_tasks: list[ActiveWorkerTaskSnapshot] = []

    for worker_id in sorted(workers):
        state = workers[worker_id].state
        retiring = bool(state.retire_when_idle)
        busy = bool(state.busy)

        if retiring:
            if busy:
                retiring_busy += 1
            else:
                retiring_idle += 1
        elif busy:
            active_busy += 1
        else:
            active_idle += 1

        if not busy:
            continue

        started_at = state.current_started_at
        busy_seconds = (
            0.0
            if started_at is None
            else max(0.0, float(now) - float(started_at))
        )
        longest_busy_seconds = max(longest_busy_seconds, busy_seconds)
        active_tasks.append(
            ActiveWorkerTaskSnapshot(
                worker_id=int(worker_id),
                task_id=state.current_task_id,
                url=state.current_url,
                kind=state.current_kind,
                busy_seconds=round(busy_seconds, 3),
                retiring=retiring,
            )
        )

    size = len(workers)
    effective_worker_count = active_busy + active_idle
    retiring_worker_count = retiring_busy + retiring_idle
    busy_worker_count = active_busy + retiring_busy
    idle_worker_count = active_idle + retiring_idle
    return WorkerPoolSnapshot(
        size=size,
        effective_worker_count=effective_worker_count,
        retiring_worker_count=retiring_worker_count,
        busy_worker_count=busy_worker_count,
        idle_worker_count=idle_worker_count,
        completed_task_count=max(0, int(task_counters.completed_task_count)),
        failure_count=max(0, int(task_counters.failure_count)),
        non_fatal_timeout_count=max(
            0, int(task_counters.non_fatal_timeout_count)
        ),
        retry_exhausted_count=max(0, int(task_counters.retry_exhausted_count)),
        average_processing_seconds=max(
            0.0, float(task_counters.average_processing_seconds)
        ),
        longest_busy_seconds=longest_busy_seconds,
        active_tasks=tuple(active_tasks),
        root_seeds_total=max(0, int(task_counters.root_seeds_total)),
        root_seeds_succeeded=max(0, int(task_counters.root_seeds_succeeded)),
        root_seeds_transient_failed=max(
            0, int(task_counters.root_seeds_transient_failed)
        ),
        root_seeds_governance_blocked=max(
            0, int(task_counters.root_seeds_governance_blocked)
        ),
    )
