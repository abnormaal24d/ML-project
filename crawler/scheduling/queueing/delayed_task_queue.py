"""Track delayed scheduler work separately from the ready host queue."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import TYPE_CHECKING, Any

from crawler.scheduling.scheduling_value_parser import coerce_float

from .host_task_queue import HostKey

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.governance.domains.host_normalizer import HostNormalizer

    from .host_task_queue import HostTaskQueue


@dataclass(slots=True, eq=False)
class DelayedTaskEntry:
    """Deferred queue item awaiting a future monotonic dispatch time."""

    ready_at: float
    sequence: int
    host: HostKey
    priority: int
    task: "CrawlTask"

    def __lt__(self, other: object) -> bool:
        """Order delayed entries by ready time, then scheduler sequence."""
        if not isinstance(other, DelayedTaskEntry):
            return NotImplemented
        return (self.ready_at, self.sequence) < (
            other.ready_at,
            other.sequence,
        )


class DelayedTaskQueue:
    """Track delayed scheduler work separately from the ready host queue."""

    def __init__(self, *, host_normalizer: HostNormalizer) -> None:
        self._host_normalizer = host_normalizer
        self._entries: list[DelayedTaskEntry] = []
        self._pending_by_host: dict[HostKey, int] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def queue_size(self) -> int:
        return len(self._entries)

    @property
    def pending_host_count(self) -> int:
        return len(self._pending_by_host)

    @property
    def max_pending_for_any_host(self) -> int:
        if not self._pending_by_host:
            return 0

        return max(self._pending_by_host.values())

    def clear(self) -> None:
        self._entries.clear()
        self._pending_by_host.clear()

    def pending_count_by_host(self) -> dict[HostKey, int]:
        return dict(self._pending_by_host)

    def pending_for_host(self, host: HostKey) -> int:
        return self._pending_by_host.get(
            self._host_normalizer.normalize(host), 0
        )

    def push(
        self,
        *,
        host: HostKey,
        priority: int,
        sequence: int,
        task: CrawlTask,
        wait_seconds: float,
        now: float | None = None,
    ) -> bool:
        clamped_wait_seconds = coerce_wait_seconds(wait_seconds)

        if clamped_wait_seconds <= 0.0:
            return False

        host = self._host_normalizer.normalize(host)
        current_time = coerce_monotonic_timestamp(now)
        ready_at = current_time + clamped_wait_seconds

        heapq.heappush(
            self._entries,
            DelayedTaskEntry(
                ready_at=ready_at,
                sequence=sequence,
                host=host,
                priority=priority,
                task=task,
            ),
        )

        self._increment_host_count(host)

        return True

    def peek_next_ready_at(self) -> float | None:
        if not self._entries:
            return None

        return self._entries[0].ready_at

    def next_ready_in_seconds(
        self,
        *,
        now: float | None = None,
    ) -> float | None:
        next_ready_at = self.peek_next_ready_at()

        if next_ready_at is None:
            return None

        current_time = coerce_monotonic_timestamp(now)

        return max(0.0, next_ready_at - current_time)

    def wait_seconds_total(
        self,
        *,
        now: float | None = None,
    ) -> float:
        current_time = coerce_monotonic_timestamp(now)

        return sum(
            max(0.0, entry.ready_at - current_time) for entry in self._entries
        )

    def average_wait_seconds(
        self,
        *,
        now: float | None = None,
    ) -> float | None:
        if not self._entries:
            return None

        return self.wait_seconds_total(now=now) / len(self._entries)

    def has_ready(
        self,
        *,
        now: float | None = None,
    ) -> bool:
        next_ready_at = self.peek_next_ready_at()

        if next_ready_at is None:
            return False

        current_time = coerce_monotonic_timestamp(now)

        return next_ready_at <= current_time

    def delayed_ready_count(
        self,
        *,
        now: float | None = None,
    ) -> int:
        current_time = coerce_monotonic_timestamp(now)

        return sum(
            1 for entry in self._entries if entry.ready_at <= current_time
        )

    def promote_ready(
        self,
        *,
        host_queue: HostTaskQueue,
        now: float | None = None,
        limit: int | None = None,
    ) -> int:
        if limit is not None and limit <= 0:
            return 0

        promoted = 0
        current_time = coerce_monotonic_timestamp(now)

        while self._entries:
            if limit is not None and promoted >= limit:
                break

            next_entry = self._entries[0]

            if next_entry.ready_at > current_time:
                break

            entry = heapq.heappop(self._entries)

            self._decrement_host_count(entry.host)

            host_queue.push(
                host=entry.host,
                priority=entry.priority,
                sequence=entry.sequence,
                task=entry.task,
            )

            promoted += 1

        return promoted

    def snapshot_items(self) -> tuple[DelayedTaskEntry, ...]:
        return tuple(
            sorted(
                self._entries,
                key=lambda entry: (
                    entry.ready_at,
                    entry.sequence,
                ),
            )
        )

    def discard_host(self, host: HostKey) -> int:
        if not self._entries:
            return 0

        host = self._host_normalizer.normalize(host)

        kept_entries = [entry for entry in self._entries if entry.host != host]

        dropped = len(self._entries) - len(kept_entries)

        if dropped <= 0:
            return 0

        self._entries = kept_entries
        heapq.heapify(self._entries)

        self._pending_by_host.pop(host, None)

        return dropped

    @staticmethod
    def serialize_entry(
        *,
        entry: DelayedTaskEntry,
        now: float,
        serializer: Any,
    ) -> dict[str, object]:
        current_time = coerce_monotonic_timestamp(now)

        serialized = serializer.serialize_queue_item(
            host=entry.host,
            priority=entry.priority,
            sequence=entry.sequence,
            task=entry.task,
        )
        if not isinstance(serialized, dict):
            raise TypeError("queue item serializer must return a dictionary")
        payload: dict[str, object] = {
            str(key): value for key, value in serialized.items()
        }

        payload["delay_remaining_seconds"] = max(
            0.0,
            entry.ready_at - current_time,
        )

        return payload

    @staticmethod
    def deserialize_wait_seconds(item: object) -> float | None:
        if not isinstance(item, dict):
            return None

        wait_seconds = coerce_float(
            item.get("delay_remaining_seconds"),
        )

        if wait_seconds is None:
            return None

        if not isfinite(wait_seconds):
            return None

        return max(0.0, wait_seconds)

    @staticmethod
    def coerce_requeue_wait_seconds(
        *,
        outcome: str,
        fields: dict[str, object] | None,
    ) -> float | None:
        if outcome != "deferred" or not fields:
            return None

        wait_seconds = coerce_float(
            fields.get("retry_after_seconds"),
        )

        if wait_seconds is None:
            return None

        if not isfinite(wait_seconds):
            return None

        if wait_seconds <= 0.0:
            return None

        return wait_seconds

    @staticmethod
    def combine_pending_maps(
        ready_counts: dict[HostKey, int],
        delayed_counts: dict[HostKey, int],
    ) -> dict[HostKey, int]:
        combined = dict(ready_counts)

        for host, delayed_count in delayed_counts.items():
            combined[host] = combined.get(host, 0) + delayed_count

        return combined

    def _increment_host_count(self, host: HostKey) -> None:
        self._pending_by_host[host] = self._pending_by_host.get(host, 0) + 1

    def _decrement_host_count(self, host: HostKey) -> None:
        previous = self._pending_by_host.get(host, 0)

        if previous <= 1:
            self._pending_by_host.pop(host, None)
            return

        self._pending_by_host[host] = previous - 1


def coerce_wait_seconds(wait_seconds: float) -> float:
    """Return a non-negative finite wait duration."""
    value = coerce_float(wait_seconds)
    if value is None or not isfinite(value):
        raise ValueError(
            f"wait_seconds must be a finite number, got {wait_seconds!r}"
        )
    return max(0.0, value)


def coerce_monotonic_timestamp(value: float | None) -> float:
    """Return a finite monotonic timestamp, using now when omitted."""
    if value is None:
        return monotonic()

    timestamp = coerce_float(value)
    if timestamp is None or not isfinite(timestamp):
        raise ValueError(f"monotonic timestamp must be finite, got {value!r}")
    return timestamp
