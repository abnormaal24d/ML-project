"""Maintain the collection of currently in-flight tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from crawler.scheduling.admission.admission_task_identity import (
    scheduler_task_identity_key,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.governance.domains.host_normalizer import HostNormalizer


@dataclass(frozen=True, slots=True)
class ActiveTaskRecord:
    """Immutable record for an active in-flight crawl task."""

    host: str | None
    priority: int
    sequence: int
    task: CrawlTask


class ActiveTaskRegistry:
    """Maintain the collection of currently in-flight tasks."""

    def __init__(self, *, host_normalizer: HostNormalizer) -> None:
        self._host_normalizer = host_normalizer
        self._records: dict[str, ActiveTaskRecord] = {}
        self._inflight_count_by_host: dict[str, int] = {}
        self._dispatching_records: dict[str, ActiveTaskRecord] = {}
        self._dead_letter_pending_records: dict[str, ActiveTaskRecord] = {}

    @property
    def inflight_count_by_host(self) -> dict[str, int]:
        """Return mutable per-host in-flight counts for dispatch readers."""

        return self._inflight_count_by_host

    def add(
        self,
        *,
        host: str | None,
        priority: int,
        sequence: int,
        task: CrawlTask,
    ) -> None:
        """Register a task as active."""
        host = self._host_normalizer.normalize(host)
        record = ActiveTaskRecord(
            host=host,
            priority=priority,
            sequence=sequence,
            task=task,
        )
        identity = self._task_identity(task)
        previous = self._records.get(identity)
        if previous is not None:
            self.decrement_inflight_for_host(previous.host)
        self._records[identity] = record
        if host is not None:
            self._inflight_count_by_host[host] = (
                self._inflight_count_by_host.get(host, 0) + 1
            )

    def remove(self, *, task: CrawlTask) -> ActiveTaskRecord | None:
        """Remove and return the active record for the canonical task."""
        record = self._records.pop(self._task_identity(task), None)
        if record is not None:
            self.decrement_inflight_for_host(record.host)
        return record

    def stage_dead_letter(self, *, record: ActiveTaskRecord) -> None:
        """Keep a terminal task tracked until its dead letter is durable."""

        self._dead_letter_pending_records[self._task_identity(record.task)] = (
            record
        )

    def remove_dead_letter_pending(
        self,
        *,
        task: CrawlTask,
    ) -> ActiveTaskRecord | None:
        """Remove and return a task awaiting dead-letter persistence."""

        return self._dead_letter_pending_records.pop(
            self._task_identity(task),
            None,
        )

    @property
    def dead_letter_pending_count(self) -> int:
        """Return terminal tasks still awaiting durable persistence."""

        return len(self._dead_letter_pending_records)

    def dead_letter_pending_values(self) -> Iterable[ActiveTaskRecord]:
        """Iterate terminal records awaiting durable persistence."""

        return self._dead_letter_pending_records.values()

    @property
    def count(self) -> int:
        """Return the number of currently active tasks."""
        return len(self._records)

    def mark_dispatching(
        self,
        *,
        host: str | None,
        priority: int,
        sequence: int,
        task: "CrawlTask",
    ) -> None:
        """Register a task as being dispatched (popped from ready, pre-activation).

        Dispatching tasks do not count toward host inflight concurrency limits.
        They are tracked for visibility (idle, checkpoint, snapshot).
        """
        host = self._host_normalizer.normalize(host)
        record = ActiveTaskRecord(
            host=host,
            priority=priority,
            sequence=sequence,
            task=task,
        )
        self._dispatching_records[self._task_identity(task)] = record

    def activate_dispatching(
        self,
        *,
        task: CrawlTask,
    ) -> ActiveTaskRecord | None:
        """Transition a dispatching task to active state and increment inflight."""
        identity = self._task_identity(task)
        record = self._dispatching_records.pop(identity, None)
        if record is None:
            return None
        previous = self._records.get(identity)
        if previous is not None:
            self.decrement_inflight_for_host(previous.host)
        self._records[identity] = record
        if record.host is not None:
            self._inflight_count_by_host[record.host] = (
                self._inflight_count_by_host.get(record.host, 0) + 1
            )
        return record

    def remove_dispatching(
        self,
        *,
        task: CrawlTask,
    ) -> ActiveTaskRecord | None:
        """Remove a task from dispatching state without activating (delay/skip/requeue)."""
        return self._dispatching_records.pop(self._task_identity(task), None)

    @property
    def dispatching_count(self) -> int:
        """Return the number of tasks currently in dispatching state."""
        return len(self._dispatching_records)

    @property
    def total_tracked_count(self) -> int:
        """Return all tasks that keep scheduler idle/join from completing."""

        return (
            self.count
            + self.dispatching_count
            + self.dead_letter_pending_count
        )

    def dispatching_values(self) -> "Iterable[ActiveTaskRecord]":
        """Iterate dispatching records."""
        return self._dispatching_records.values()

    def dispatching_items(self) -> "Iterable[tuple[str, ActiveTaskRecord]]":
        """Iterate scheduler-identity + dispatching records."""
        return self._dispatching_records.items()

    def clear(self) -> None:
        """Drop all active task records."""
        self._records.clear()
        self._dispatching_records.clear()
        self._dead_letter_pending_records.clear()
        self._inflight_count_by_host.clear()

    def decrement_inflight_for_host(self, host: str | None) -> None:
        """Reduce the in-flight count for a host when work finishes."""

        if host is None:
            return

        host = self._host_normalizer.require(host)

        previous = self._inflight_count_by_host.get(host, 0)
        if previous <= 1:
            self._inflight_count_by_host.pop(host, None)
            return

        self._inflight_count_by_host[host] = previous - 1

    def values(self) -> Iterable[ActiveTaskRecord]:
        """Iterate over all active task records."""
        return self._records.values()

    def items(self) -> Iterable[tuple[str, ActiveTaskRecord]]:
        """Iterate over scheduler identity keys and their records."""
        return self._records.items()

    @staticmethod
    def _task_identity(task: CrawlTask) -> str:
        return scheduler_task_identity_key(task=task)
