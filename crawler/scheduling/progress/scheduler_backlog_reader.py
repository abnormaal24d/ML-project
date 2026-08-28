"""Read scheduler ready-queue pending counts for admission decisions.

``host_pending`` and ``kind_host_pending`` count only tasks in the ready
``HostTaskQueue``. Delayed and in-flight tasks are intentionally excluded:
``max_pending_per_host`` is a ready-queue cap (see
``docs/architecture/scheduler_retry_semantics.md``), and execution
concurrency is governed by the separate ``max_inflight_per_host`` contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.classification.media_kind import MediaKind
from crawler.crawl_tasks.crawl_task import CrawlTask

from ..queueing.delayed_task_queue import DelayedTaskQueue
from ..queueing.host_task_queue import PendingMap

if TYPE_CHECKING:
    from ..queueing.host_task_queue import HostTaskQueue


class SchedulerBacklogReader:
    """Expose ready-queue pending counts for scheduler admission decisions."""

    def __init__(
        self,
        *,
        host_queue: HostTaskQueue,
        delayed_queue: DelayedTaskQueue,
        max_feeds_per_host: int,
    ) -> None:
        self._host_queue = host_queue
        self._delayed_queue = delayed_queue
        self._max_feeds_per_host = max(0, int(max_feeds_per_host))

    def pending_maps(self) -> tuple[PendingMap, PendingMap, PendingMap]:
        """Return ready, delayed, and combined pending counts by host."""

        ready_pending = self._host_queue.pending_count_by_host()
        delayed_pending = self._delayed_queue.pending_count_by_host()
        total_pending = DelayedTaskQueue.combine_pending_maps(
            ready_pending,
            delayed_pending,
        )
        return ready_pending, delayed_pending, total_pending

    def combined_pending_count_by_host(self) -> PendingMap:
        """Return total pending task counts across ready and delayed queues."""

        _, _, total_pending = self.pending_maps()
        return total_pending

    def host_pending(self, *, host: str | None) -> int:
        """Return ready-queue pending tasks for one host.

        Delayed (pacing) tasks and in-flight tasks are not pending frontier
        capacity: they are governed by ``DelayedTaskQueue`` and
        ``max_inflight_per_host`` independently. A ``None`` host has no
        host bucket and therefore no pending count.
        """

        return self._host_queue.pending_for_host(host)

    def kind_host_pending_if_needed(
        self,
        *,
        task: CrawlTask,
        host: str | None,
    ) -> int:
        """Return per-kind ready pending when admission limits require it."""

        if not self._needs_kind_host_pending(task=task):
            return 0
        return self.kind_host_pending(host=host, kind=task.kind)

    def kind_host_pending(
        self,
        *,
        host: str | None,
        kind: MediaKind,
    ) -> int:
        """Return ready-queue pending task count for one host and kind."""

        if host is None:
            return 0
        return sum(
            1
            for queued_host, _priority, _sequence, task in (
                self._host_queue.snapshot_items()
            )
            if queued_host == host and task.kind == kind
        )

    def _needs_kind_host_pending(self, *, task: CrawlTask) -> bool:
        if task.source_type == "seed":
            return False

        if task.kind is MediaKind.FEED:
            return self._max_feeds_per_host > 0

        return task.kind in {
            MediaKind.PAGE,
            MediaKind.IMAGE,
            MediaKind.AUDIO,
            MediaKind.DOCUMENT,
            MediaKind.VIDEO,
        }
