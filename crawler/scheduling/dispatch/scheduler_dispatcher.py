"""Dispatch tasks host-first, applying governance waits at host level."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from time import monotonic
from typing import TYPE_CHECKING

from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.runtime.concurrency import condition_notify_all

if TYPE_CHECKING:
    from logger.project_logger import ProjectLogger

    from ..progress.active_task_registry import ActiveTaskRegistry
    from ..queueing.delayed_task_queue import DelayedTaskQueue
    from ..queueing.host_eligibility_queue import HostEligibilityQueue
    from ..queueing.host_task_queue import HostTaskQueue
    from .host_dispatch_wait_reader import HostDispatchWaitReader

IsClosed = Callable[[], bool]
ClosedErrorFactory = Callable[[], BaseException]
SkipReasonBeforeFetch = Callable[[CrawlTask], str | None]


class SchedulerDispatcher:
    """Apply host wait rules to ready queue items.

    Host governance waits (rate limiting, suppression, inflight cap) are
    tracked per host in a ``HostEligibilityQueue``; task-level retry delays
    remain in the ``DelayedTaskQueue``. A task is only popped from the ready
    queue once its host is actually eligible, so host pacing never requeues
    tasks.
    """

    _MIN_DELAY_WAIT_SECONDS = 0.01
    _MAX_DELAY_WAIT_SECONDS = 0.5
    _MAX_DELAYED_PROMOTIONS_PER_TICK = 512

    def __init__(
        self,
        *,
        dispatch_wait_reader: HostDispatchWaitReader,
        host_queue: HostTaskQueue,
        delayed_queue: DelayedTaskQueue,
        host_eligibility_queue: HostEligibilityQueue,
        inflight_count_by_host: dict[str, int],
        max_inflight_per_host: int | None = None,
        condition: asyncio.Condition,
        active_registry: ActiveTaskRegistry,
        abandon_suppressed_host_threshold_seconds: float | None = None,
        logger: ProjectLogger,
        is_closed: Callable[[], bool] | None = None,
        closed_error_factory: Callable[[], BaseException] | None = None,
        skip_reason_before_fetch: SkipReasonBeforeFetch | None = None,
    ) -> None:
        self._dispatch_wait_reader = dispatch_wait_reader
        self._host_queue = host_queue
        self._delayed_queue = delayed_queue
        self._host_eligibility_queue = host_eligibility_queue
        self._inflight_count_by_host = inflight_count_by_host
        self._max_inflight_per_host = coerce_max_inflight(
            max_inflight_per_host
        )
        self._condition = condition
        self._active_registry = active_registry
        self._abandon_suppressed_host_threshold_seconds = (
            abandon_suppressed_host_threshold_seconds
        )
        self._logger = logger
        self._is_closed = is_closed or (lambda: False)
        self._closed_error_factory = closed_error_factory or (
            lambda: RuntimeError("scheduler is closed")
        )
        self._skip_reason_before_fetch = skip_reason_before_fetch

    async def get(self) -> CrawlTask:
        while True:
            async with self._condition:
                host, poll_delay = self._pick_host_or_delay_locked()

                if host is None and poll_delay is None:
                    await self._condition.wait()
                    continue

            if host is not None:
                task = await self.dispatch_for_host(host=host)
                if task is not None:
                    return task
                continue

            if poll_delay is not None:
                await asyncio.sleep(poll_delay)

    async def dispatch_for_host(
        self,
        *,
        host: str | None,
    ) -> CrawlTask | None:
        """Dispatch the next task for one host.

        Returns the task when one was activated, or None when the host is
        blocked, paced, or the task was skipped before fetch.
        """

        try:
            return await self._dispatch_host_locked(host=host)
        except (RuntimeError, OSError, ValueError):
            await self._requeue_dispatching_item(host=host)
            raise

    async def _dispatch_host_locked(
        self, *, host: str | None
    ) -> CrawlTask | None:
        governance_wait_seconds = (
            await self._dispatch_wait_reader.governance_wait_seconds(host=host)
        )

        async with self._condition:
            if self._host_queue.pending_for_host(host) <= 0:
                self._host_eligibility_queue.remove(host)
                return None

            if not self._host_has_room_locked(host=host):
                self._block_host_locked(host=host)
                condition_notify_all(self._condition)
                return None

            if (
                governance_wait_seconds is not None
                and governance_wait_seconds > 0.0
            ):
                self._host_eligibility_queue.upsert(
                    host=host,
                    next_eligible_at=(monotonic() + governance_wait_seconds),
                    inflight=self._current_inflight(host),
                )
                self._logger.debug(
                    "scheduler_host_wait",
                    host=host,
                    wait_seconds=round(float(governance_wait_seconds), 4),
                )
                condition_notify_all(self._condition)
                return None

            item = self._host_queue.pop_item_for_host(host)
            if item is None:
                self._host_eligibility_queue.remove(host)
                return None

            item_host, priority, sequence, task = item
            self._active_registry.mark_dispatching(
                host=item_host,
                priority=priority,
                sequence=sequence,
                task=task,
            )

            if self._should_skip_task_before_fetch(task=task):
                self._active_registry.remove_dispatching(task=task)
                condition_notify_all(self._condition)
                return None

            self._active_registry.activate_dispatching(task=task)
            condition_notify_all(self._condition)
            return task

    def _pick_host_or_delay_locked(
        self,
    ) -> tuple[str | None, float | None]:
        now = monotonic()
        self._refresh_queues_locked(now=now)

        if self._host_queue.queue_size > 0:
            candidate = self._candidate_host_locked(now=now)
            if candidate is not None:
                return candidate, None

            if self._is_closed():
                raise self._closed_error_factory()

            delay = self._host_eligibility_queue.next_ready_in_seconds(now=now)
            if delay is None:
                return None, None
            return None, self._clamp_poll_delay(delay)

        if self._is_closed():
            raise self._closed_error_factory()

        delay = self._delayed_queue.next_ready_in_seconds(now=now)
        if delay is None:
            return None, None
        return None, self._clamp_poll_delay(delay)

    def _candidate_host_locked(self, *, now: float) -> str | None:
        """Return the host that may be dispatched right now, if any."""
        while True:
            peeked = self._host_eligibility_queue.peek()
            if peeked is None:
                for candidate in self._host_queue.pending_count_by_host():
                    if self._host_queue.pending_for_host(candidate) > 0:
                        return candidate
                return None

            host, ready_at = peeked
            if self._host_queue.pending_for_host(host) <= 0:
                self._host_eligibility_queue.remove(host)
                continue

            if math.isinf(ready_at):
                # Blocked host: not dispatchable until a completion releases
                # it; all other entries are at least as far out.
                return None

            if ready_at <= now:
                return host
            return None

    def _host_has_room_locked(self, *, host: str | None) -> bool:
        if host is None or self._max_inflight_per_host is None:
            return True
        return (
            self._inflight_count_by_host.get(host, 0)
            < self._max_inflight_per_host
        )

    def _block_host_locked(self, *, host: str | None) -> None:
        self._host_eligibility_queue.upsert(
            host=host,
            next_eligible_at=math.inf,
            inflight=self._current_inflight(host),
        )
        self._logger.debug(
            "scheduler_host_inflight_blocked",
            host=host,
            inflight=self._current_inflight(host),
            max_inflight_per_host=self._max_inflight_per_host,
        )

    def _current_inflight(self, host: str | None) -> int:
        if host is None:
            return 0
        return self._inflight_count_by_host.get(host, 0)

    def _should_skip_task_before_fetch(self, *, task: CrawlTask) -> bool:
        if self._skip_reason_before_fetch is None:
            return False
        reason = self._skip_reason_before_fetch(task)
        if not reason:
            return False
        self._logger.info(
            "task_skipped_before_fetch",
            task_id=task.task_id,
            url=task.url,
            kind=task.kind,
            source=task.source_type,
            reason=str(reason),
        )
        return True

    def _refresh_queues_locked(self, *, now: float) -> None:
        dropped = self._prune_suppressed_host_backlog_locked()
        promoted = self._delayed_queue.promote_ready(
            host_queue=self._host_queue,
            now=now,
            limit=self._MAX_DELAYED_PROMOTIONS_PER_TICK,
        )

        if promoted > 0 and self._logger is not None:
            self._logger.debug(
                "scheduler_promoted_delayed_tasks",
                promoted=promoted,
                ready_queue=self._host_queue.queue_size,
                delayed_queue=self._delayed_queue.queue_size,
            )

        if dropped > 0 or promoted > 0:
            condition_notify_all(self._condition)

    def _prune_suppressed_host_backlog_locked(self) -> int:
        """Drop host backlog when suppression is long enough to stall crawl."""

        threshold = self._abandon_suppressed_host_threshold_seconds
        if threshold is None or threshold <= 0:
            return 0

        candidate_hosts = self._candidate_suppressed_hosts()
        if not candidate_hosts:
            return 0

        dropped_total = 0
        for host in candidate_hosts:
            suppression_remaining = (
                self._dispatch_wait_reader.suppression_remaining_seconds(
                    host=host,
                )
            )
            if (
                suppression_remaining is None
                or suppression_remaining < threshold
            ):
                continue

            dropped_ready = self._host_queue.discard_host(host)
            dropped_delayed = self._delayed_queue.discard_host(host)
            self._host_eligibility_queue.remove(host)
            dropped_host_total = dropped_ready + dropped_delayed
            if dropped_host_total <= 0:
                continue

            dropped_total += dropped_host_total
            self._logger.warning(
                "scheduler_suppressed_host_backlog_abandoned",
                host=host,
                dropped_tasks=dropped_host_total,
                dropped_ready=dropped_ready,
                dropped_delayed=dropped_delayed,
                suppression_remaining_seconds=round(
                    float(suppression_remaining),
                    3,
                ),
                threshold_seconds=round(float(threshold), 3),
            )

        return dropped_total

    def _candidate_suppressed_hosts(self) -> set[str]:
        ready_pending = self._host_queue.pending_count_by_host()
        delayed_pending = self._delayed_queue.pending_count_by_host()

        return {
            host
            for host in (*ready_pending, *delayed_pending)
            if host is not None
        }

    async def _requeue_dispatching_item(self, *, host: str | None) -> None:
        """Requeue a dispatching item for a host after a dispatch failure."""
        async with self._condition:
            record = None
            for candidate in self._active_registry.dispatching_values():
                if candidate.host == host:
                    record = candidate
                    break
            if record is None:
                return
            self._active_registry.remove_dispatching(task=record.task)
            self._host_queue.push(
                host=record.host,
                priority=record.priority,
                sequence=record.sequence,
                task=record.task,
            )
            condition_notify_all(self._condition)

    @classmethod
    def _clamp_poll_delay(cls, wait_seconds: float) -> float:
        return max(
            cls._MIN_DELAY_WAIT_SECONDS,
            min(float(wait_seconds), cls._MAX_DELAY_WAIT_SECONDS),
        )


def coerce_max_inflight(value: int | None) -> int | None:
    """Return a positive inflight cap, or None when unlimited."""
    if value is None:
        return None
    resolved = max(0, int(value))
    return resolved if resolved > 0 else None
