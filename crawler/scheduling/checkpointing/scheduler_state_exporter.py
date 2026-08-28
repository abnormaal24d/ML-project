"""Export scheduler runtime structures into a checkpoint payload."""

from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING

from crawler.scheduling.queueing.delayed_task_queue import DelayedTaskQueue

if TYPE_CHECKING:
    from crawler.scheduling.checkpointing.scheduler_task_serializer import (
        SchedulerTaskSerializer,
    )
    from crawler.scheduling.dedupe.seen_url_registry import SeenUrlRegistry
    from crawler.scheduling.queueing.host_task_queue import HostTaskQueue

    from ..progress.active_task_registry import ActiveTaskRegistry


class SchedulerStateExporter:
    """Export scheduler runtime structures into a checkpoint payload."""

    def __init__(self, *, serializer: SchedulerTaskSerializer) -> None:
        self._serializer = serializer

    def export_state(
        self,
        *,
        max_queued_tasks: int,
        include_seen_urls: bool,
        host_queue: HostTaskQueue,
        delayed_queue: DelayedTaskQueue,
        active_registry: ActiveTaskRegistry,
        seen_urls: SeenUrlRegistry,
        next_sequence_value: int,
        total_pending_by_host: dict[str | None, int],
        progress_counters: dict[str, object],
        retry_budget_state: dict[str, object],
    ) -> dict[str, object]:
        """Return the serialized scheduler checkpoint payload."""

        now = monotonic()
        queued_items = host_queue.snapshot_items()
        delayed_items = delayed_queue.snapshot_items()
        if max_queued_tasks >= 0:
            queued_items = queued_items[:max_queued_tasks]
            delayed_items = delayed_items[:max_queued_tasks]

        queued_payload: list[dict[str, object]] = []
        delayed_payload: list[dict[str, object]] = []
        requeued_inflight_payload: list[dict[str, object]] = []
        max_sequence = -1

        for host, priority, sequence, task in queued_items:
            queued_payload.append(
                self._serializer.serialize_queue_item(
                    host=host,
                    priority=priority,
                    sequence=sequence,
                    task=task,
                )
            )
            max_sequence = max(max_sequence, sequence)

        for entry in delayed_items:
            delayed_payload.append(
                DelayedTaskQueue.serialize_entry(
                    entry=entry,
                    now=now,
                    serializer=self._serializer,
                )
            )
            max_sequence = max(max_sequence, entry.sequence)

        for record in active_registry.values():
            requeued_inflight_payload.append(
                self._serializer.serialize_active_record(record=record)
            )
            max_sequence = max(max_sequence, record.sequence)

        for record in active_registry.dead_letter_pending_values():
            requeued_inflight_payload.append(
                self._serializer.serialize_active_record(record=record)
            )
            max_sequence = max(max_sequence, record.sequence)

        dispatching_items = list(active_registry.dispatching_items())
        dispatching_payload: list[dict[str, object]] = []
        for _url, record in dispatching_items:
            dispatching_payload.append(
                self._serializer.serialize_active_record(record=record)
            )
            max_sequence = max(max_sequence, record.sequence)

        next_sequence = max(next_sequence_value, max_sequence + 1)

        payload: dict[str, object] = {
            "schema_version": 2,
            "queue_size": (host_queue.queue_size + delayed_queue.queue_size),
            "ready_queue_size": host_queue.queue_size,
            "delayed_queue_size": delayed_queue.queue_size,
            "pending_hosts": len(total_pending_by_host),
            "ready_pending_hosts": host_queue.host_count,
            "delayed_pending_hosts": delayed_queue.pending_host_count,
            "active_tasks": (
                active_registry.count
                + active_registry.dead_letter_pending_count
            ),
            "dispatching_tasks": dispatching_payload,
            "queued_tasks": queued_payload,
            "delayed_tasks": delayed_payload,
            "requeued_inflight_tasks": requeued_inflight_payload,
            "next_sequence": next_sequence,
            "progress_counters": progress_counters,
            "retry_budget": retry_budget_state,
        }

        if include_seen_urls:
            payload["seen_url_entries"] = [
                {"url": url, "seen_at": seen_at}
                for url, seen_at in seen_urls.export_entries()
            ]

        return payload
