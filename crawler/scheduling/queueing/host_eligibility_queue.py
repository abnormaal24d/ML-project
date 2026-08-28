"""Track host-level dispatch eligibility separately from task-level delays.

Host governance waits (rate limiting, suppression, inflight cap) are host
properties, not task properties. This queue orders hosts with queued work by
their earliest dispatch eligibility time so the dispatcher can pop a task
only for a host that is actually ready, instead of popping and re-delaying
tasks while a host is paced.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from time import monotonic

from crawler.scheduling.scheduling_value_parser import coerce_float

from .host_task_queue import HostKey

_BLOCKED_ELIGIBLE_AT = math.inf


@dataclass(slots=True, eq=False)
class HostDispatchState:
    """Per-host dispatch eligibility snapshot.

    ``next_eligible_at`` is a monotonic timestamp; ``math.inf`` marks a host
    that is blocked by the inflight cap until a completion releases it.
    """

    host: HostKey
    next_eligible_at: float
    inflight: int
    generation: int


class HostEligibilityQueue:
    """Order hosts with queued work by earliest dispatch eligibility.

    Entries are best-effort lower bounds on when the next task for a host may
    be dispatched; the authoritative governance state is re-read right before
    a dispatch decision, so stale entries are safe. Only hosts with queued
    tasks are tracked; hosts without tasks are removed lazily or explicitly.
    """

    def __init__(self) -> None:
        self._states: dict[HostKey, HostDispatchState] = {}
        self._heap: list[tuple[float, int, HostKey]] = []
        self._generation = 0

    def __len__(self) -> int:
        return len(self._states)

    @property
    def queue_size(self) -> int:
        """Return the number of tracked hosts."""
        return len(self._states)

    @property
    def pending_host_count(self) -> int:
        """Return the number of tracked hosts."""
        return len(self._states)

    def clear(self) -> None:
        """Drop all tracked hosts."""
        self._states.clear()
        self._heap.clear()

    def next_eligible_at(self, host: HostKey) -> float | None:
        """Return the cached eligibility timestamp for a host, if tracked."""
        state = self._states.get(host)
        if state is None:
            return None
        return state.next_eligible_at

    def upsert(
        self,
        host: HostKey,
        *,
        next_eligible_at: float,
        inflight: int,
    ) -> None:
        """Set or refresh the eligibility timestamp for a host.

        New heap entries are pushed with a fresh generation so earlier
        entries for the same host are skipped lazily.
        """
        self._generation += 1
        state = HostDispatchState(
            host=host,
            next_eligible_at=self._coerce_timestamp(next_eligible_at),
            inflight=max(0, int(inflight)),
            generation=self._generation,
        )
        self._states[host] = state
        heapq.heappush(
            self._heap,
            (state.next_eligible_at, state.generation, host),
        )

    def remove(self, host: HostKey) -> None:
        """Stop tracking a host; heap entries become stale lazily."""
        self._states.pop(host, None)

    def release_blocked(self, host: HostKey) -> None:
        """Mark a blocked host immediately eligible again.

        Called by the completion path when a host's inflight slot frees up.
        """
        state = self._states.get(host)
        if state is None or not math.isinf(state.next_eligible_at):
            return
        self.upsert(
            host=host,
            next_eligible_at=0.0,
            inflight=state.inflight,
        )

    def peek(self) -> tuple[HostKey, float] | None:
        """Return (host, next_eligible_at) of the earliest tracked host."""
        while self._heap:
            next_eligible_at, generation, host = self._heap[0]
            state = self._states.get(host)
            if state is not None and state.generation == generation:
                return host, next_eligible_at
            heapq.heappop(self._heap)
        return None

    def has_ready(self, *, now: float | None = None) -> bool:
        """Return whether the earliest host is due at ``now``."""
        current_time = coerce_monotonic_timestamp(now)
        peeked = self.peek()
        if peeked is None:
            return False
        _host, ready_at = peeked
        return ready_at <= current_time

    def next_ready_in_seconds(
        self,
        *,
        now: float | None = None,
    ) -> float | None:
        """Return seconds until the next non-blocked host becomes eligible."""
        current_time = coerce_monotonic_timestamp(now)
        peeked = self.peek()
        if peeked is None:
            return None
        _host, ready_at = peeked
        if math.isinf(ready_at):
            return None
        return max(0.0, ready_at - current_time)

    def states_items(self) -> tuple[tuple[HostKey, HostDispatchState], ...]:
        """Return a snapshot of all tracked host states."""
        return tuple(self._states.items())

    @staticmethod
    def _coerce_timestamp(value: float) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"next_eligible_at must be a number, got {value!r}"
            ) from exc
        if math.isnan(numeric_value):
            raise ValueError("next_eligible_at must not be NaN")
        return numeric_value


def coerce_monotonic_timestamp(value: float | None) -> float:
    """Return a finite monotonic timestamp, using now when omitted."""
    if value is None:
        return monotonic()

    timestamp = coerce_float(value)
    if timestamp is None:
        raise ValueError(f"monotonic timestamp must be finite, got {value!r}")
    return timestamp
