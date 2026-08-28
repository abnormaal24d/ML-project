"""Serialize scheduled task envelopes for checkpoint persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .scheduler_task_envelope import SchedulerTaskEnvelope

if TYPE_CHECKING:
    from crawler.crawl_tasks.crawl_task import CrawlTask

    from ..progress.active_task_registry import ActiveTaskRecord


class SchedulerTaskSerializer:
    """Serialize scheduled task envelopes for checkpoint persistence."""

    @staticmethod
    def serialize(
        *,
        envelope: SchedulerTaskEnvelope,
    ) -> dict[str, object]:
        """Return a checkpoint payload for a scheduled task."""

        task = envelope.task
        return {
            "url": task.url,
            "source_name": task.source_name,
            "task_id": task.task_id,
            "kind": task.kind,
            "depth": task.depth,
            "source_type": task.source_type,
            "priority": envelope.priority,
            "parent_url": task.parent_url,
            "context": task.context.to_dict()
            if task.context is not None
            else None,
            "host": envelope.host,
            "sequence": envelope.sequence,
        }

    def serialize_queue_item(
        self,
        *,
        host: str | None,
        priority: int,
        sequence: int,
        task: CrawlTask,
    ) -> dict[str, object]:
        """Return a serialized payload for a queued scheduler item."""

        return self.serialize(
            envelope=SchedulerTaskEnvelope(
                task=task,
                host=host,
                priority=priority,
                sequence=sequence,
            )
        )

    def serialize_active_record(
        self,
        *,
        record: ActiveTaskRecord,
    ) -> dict[str, object]:
        """Return a serialized payload for an active scheduler record."""

        return self.serialize_queue_item(
            host=record.host,
            priority=record.priority,
            sequence=record.sequence,
            task=record.task,
        )
