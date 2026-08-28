"""Maintain fair per-host task queues with round-robin scheduling."""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.governance.domains.host_normalizer import HostNormalizer

HostKey: TypeAlias = str | None
QueuedTaskSnapshot: TypeAlias = tuple[HostKey, int, int, "CrawlTask"]
HostQueueEntrySnapshot: TypeAlias = tuple[int, int, "CrawlTask"]
HostQueueSnapshot: TypeAlias = tuple[
    HostKey, tuple[HostQueueEntrySnapshot, ...]
]
PendingMap: TypeAlias = dict[HostKey, int]


def max_pending_count(pending: Mapping[HostKey, int]) -> int:
    """Return the largest per-host pending count."""

    if not pending:
        return 0
    return max(pending.values())


@dataclass(slots=True, eq=False)
class QueueEntry:
    """Single comparable entry stored in a host-local heap."""

    priority: int
    sequence: int
    task: "CrawlTask"

    def __lt__(self, other: object) -> bool:
        """Order heap entries deterministically by priority and sequence.

        Lower numeric priority score means higher scheduling priority, so the
        minimum priority is popped first. sequence preserves FIFO ordering for
        equal priority scores.
        """
        if not isinstance(other, QueueEntry):
            return NotImplemented
        return (self.priority, self.sequence) < (
            other.priority,
            other.sequence,
        )


@dataclass(frozen=True, slots=True)
class HostTaskQueueState:
    """Serializable queue state for exact checkpoint/restore."""

    host_order: tuple[HostKey, ...]
    host_queues: tuple[HostQueueSnapshot, ...]


class HostTaskQueue:
    """Maintain fair per-host task queues with round-robin scheduling."""

    def __init__(self, *, host_normalizer: HostNormalizer) -> None:
        self._host_normalizer = host_normalizer
        self._queues: dict[HostKey, list[QueueEntry]] = {}
        self._pending_by_host: dict[HostKey, int] = {}
        self._host_order: deque[HostKey] = deque()
        self._hosts_in_rotation: set[HostKey] = set()
        self._queued_count = 0

    @property
    def queue_size(self) -> int:
        """Return the total queued task count."""
        return self._queued_count

    @property
    def host_count(self) -> int:
        """Return the number of hosts that currently have queued work."""
        return len(self._queues)

    @property
    def max_pending_for_any_host(self) -> int:
        """Return the largest queued backlog for a single host."""
        if not self._pending_by_host:
            return 0
        return max(self._pending_by_host.values())

    def pending_for_host(self, host: HostKey) -> int:
        """Return the queued task count for a host without mutating state."""
        return self._pending_by_host.get(
            self._host_normalizer.normalize(host), 0
        )

    def pending_count_by_host(self) -> dict[HostKey, int]:
        """Return a copy of the queued task count mapping per host."""
        return dict(self._pending_by_host)

    def push(
        self,
        *,
        host: HostKey,
        priority: int,
        sequence: int,
        task: CrawlTask,
    ) -> None:
        """Push a task for a host into the fair queue."""
        host = self._host_normalizer.normalize(host)
        queue = self._queues.get(host)
        if queue is None:
            queue = []
            self._queues[host] = queue

        queue_was_empty = not queue
        entry = QueueEntry(priority=priority, sequence=sequence, task=task)
        heapq.heappush(queue, entry)

        self._pending_by_host[host] = self._pending_by_host.get(host, 0) + 1
        self._queued_count += 1

        if queue_was_empty and host not in self._hosts_in_rotation:
            self._host_order.append(host)
            self._hosts_in_rotation.add(host)

    def pop(self) -> CrawlTask:
        """Pop the next fairly scheduled task."""
        return self.pop_item()[-1]

    def pop_item_for_host(
        self,
        host: HostKey,
    ) -> tuple[HostKey, int, int, CrawlTask] | None:
        """Pop the next item for one host, preserving round-robin state.

        The host keeps its rotation position while it still has queued work;
        when its queue empties the host is removed from the rotation.
        """
        host = self._host_normalizer.normalize(host)
        queue = self._queues.get(host)
        if not queue:
            return None

        entry = heapq.heappop(queue)
        remaining = self._pending_by_host.get(host, 0) - 1
        self._queued_count -= 1

        if remaining > 0:
            self._pending_by_host[host] = remaining
            if host not in self._hosts_in_rotation:
                self._host_order.append(host)
                self._hosts_in_rotation.add(host)
        else:
            self._queues.pop(host, None)
            self._pending_by_host.pop(host, None)
            self._hosts_in_rotation.discard(host)
            if self._host_order:
                self._host_order = deque(
                    queued_host
                    for queued_host in self._host_order
                    if queued_host != host
                )

        return host, entry.priority, entry.sequence, entry.task

    def pop_item(self) -> tuple[HostKey, int, int, CrawlTask]:
        """Pop the next fairly scheduled queue item with metadata."""
        if not self._host_order:
            raise LookupError("no queued tasks available")

        host = self._host_order.popleft()
        self._hosts_in_rotation.discard(host)

        queue = self._queues.get(host)
        if not queue:
            raise LookupError(
                "scheduler state corrupted: host rotation contains empty host"
            )

        entry = heapq.heappop(queue)
        remaining = self._pending_by_host.get(host, 0) - 1
        self._queued_count -= 1

        if remaining > 0:
            self._pending_by_host[host] = remaining
            self._host_order.append(host)
            self._hosts_in_rotation.add(host)
        else:
            self._queues.pop(host, None)
            self._pending_by_host.pop(host, None)

        return host, entry.priority, entry.sequence, entry.task

    def discard_host(self, host: HostKey) -> int:
        """Remove all queued work for one host and return the dropped count."""

        host = self._host_normalizer.normalize(host)
        queue = self._queues.pop(host, None)
        if not queue:
            self._pending_by_host.pop(host, None)
            self._hosts_in_rotation.discard(host)
            if self._host_order:
                self._host_order = deque(
                    queued_host
                    for queued_host in self._host_order
                    if queued_host != host
                )
            return 0

        dropped = len(queue)
        self._queued_count = max(0, self._queued_count - dropped)
        self._pending_by_host.pop(host, None)
        self._hosts_in_rotation.discard(host)
        self._host_order = deque(
            queued_host
            for queued_host in self._host_order
            if queued_host != host
        )
        return dropped

    def snapshot_items(self) -> tuple[QueuedTaskSnapshot, ...]:
        """Return queued items in the true future fairness pop order.

        This simulates future pops against copied per-host heaps and a copied
        rotation deque. It preserves the actual current round-robin scheduling
        order instead of imposing a global sort that would distort checkpoint
        and restore behavior.
        """
        if not self._host_order:
            return ()

        queues_copy: dict[HostKey, list[QueueEntry]] = {
            host: list(queue) for host, queue in self._queues.items()
        }
        rotation_copy = deque(self._host_order)
        snapshot: list[QueuedTaskSnapshot] = []

        while rotation_copy:
            host = rotation_copy.popleft()
            queue = queues_copy.get(host)
            if not queue:
                continue

            entry = heapq.heappop(queue)
            snapshot.append((host, entry.priority, entry.sequence, entry.task))

            if queue:
                rotation_copy.append(host)

        return tuple(snapshot)

    def snapshot_state(self) -> HostTaskQueueState:
        """Return full queue state preserving exact fairness rotation.

        Use this for checkpoint and restore when future scheduling behavior
        must
        remain identical across restarts.
        """
        ordered_hosts: list[HostKey] = []
        seen_hosts: set[HostKey] = set()

        for host in self._host_order:
            if host in seen_hosts:
                continue
            queue = self._queues.get(host)
            if not queue:
                continue
            ordered_hosts.append(host)
            seen_hosts.add(host)

        for host, queue in self._queues.items():
            if host in seen_hosts or not queue:
                continue
            ordered_hosts.append(host)
            seen_hosts.add(host)

        host_snapshots: list[HostQueueSnapshot] = []
        for host in ordered_hosts:
            queue = self._queues.get(host)
            if not queue:
                continue

            entries_in_pop_order = tuple(
                (entry.priority, entry.sequence, entry.task)
                for entry in self._iter_heap_in_pop_order(queue)
            )
            host_snapshots.append((host, entries_in_pop_order))

        return HostTaskQueueState(
            host_order=tuple(ordered_hosts),
            host_queues=tuple(host_snapshots),
        )

    def restore_state(self, state: HostTaskQueueState) -> int:
        """Restore queue state from ``snapshot_state()``.

        The restored state preserves:
        - per-host queue contents
        - host round-robin rotation order
        - total queued count
        """
        self.clear()

        active_hosts: list[HostKey] = []

        for raw_host, entries in state.host_queues:
            host = self._host_normalizer.normalize(raw_host)
            queue = self._queues.setdefault(host, [])

            for priority, sequence, task in entries:
                heapq.heappush(
                    queue,
                    QueueEntry(
                        priority=priority,
                        sequence=sequence,
                        task=task,
                    ),
                )

            if not queue:
                continue

            self._pending_by_host[host] = len(queue)
            if host not in active_hosts:
                active_hosts.append(host)

        self._queued_count = sum(len(queue) for queue in self._queues.values())

        active_host_set = set(active_hosts)

        for raw_host in state.host_order:
            host = self._host_normalizer.normalize(raw_host)
            if host in active_host_set and host not in self._hosts_in_rotation:
                self._host_order.append(host)
                self._hosts_in_rotation.add(host)

        for host in active_hosts:
            if host not in self._hosts_in_rotation:
                self._host_order.append(host)
                self._hosts_in_rotation.add(host)

        return self._queued_count

    def clear(self) -> None:
        """Drop all queued tasks and reset host fairness state."""
        self._queues.clear()
        self._pending_by_host.clear()
        self._host_order.clear()
        self._hosts_in_rotation.clear()
        self._queued_count = 0

    @staticmethod
    def _iter_heap_in_pop_order(
        queue: list[QueueEntry],
    ) -> tuple[QueueEntry, ...]:
        """
        Return heap entries in the exact order ``heapq.heappop`` would yield.
        """
        queue_copy = list(queue)
        ordered: list[QueueEntry] = []

        while queue_copy:
            ordered.append(heapq.heappop(queue_copy))

        return tuple(ordered)
