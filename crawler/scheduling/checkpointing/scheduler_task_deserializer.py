"""Deserialize scheduled task envelopes from checkpoint persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawler.crawl_tasks.crawl_task import CrawlTask
from crawler.scheduling.scheduling_value_parser import coerce_int, coerce_str
from shared.runtime_primitives import IdGenerator

from .scheduler_task_envelope import SchedulerTaskEnvelope

if TYPE_CHECKING:
    from crawler.extraction.hosts_extractor import HostExtractor
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.scheduling.priority.crawl_task_priority_calculator import (
        CrawlTaskPriorityCalculator,
    )


class SchedulerTaskDeserializer:
    """Deserialize scheduled task envelopes from checkpoint persistence."""

    _REQUIRED_CHECKPOINT_FIELDS = (
        "url",
        "source_name",
        "task_id",
        "kind",
        "depth",
        "source_type",
        "priority",
        "parent_url",
        "context",
        "host",
        "sequence",
    )

    def __init__(
        self,
        *,
        priority_resolver: CrawlTaskPriorityCalculator,
        host_extractor: HostExtractor,
        url_normalizer: UrlNormalizer,
        id_generator: IdGenerator,
    ) -> None:
        if id_generator is None:
            raise ValueError("id_generator is required")

        self._priority_resolver = priority_resolver
        self._id_generator = id_generator
        self._host_extractor = host_extractor
        self._url_normalizer = url_normalizer

    def deserialize_task_payload(
        self,
        *,
        item: object,
        default_kind: str = "page",
        default_source_type: str = "seed",
        default_priority: int = 0,
        resolve_missing_priority: bool = True,
    ) -> CrawlTask | None:
        """Return a CrawlTask reconstructed from arbitrary payload data."""

        if not isinstance(item, dict):
            return None

        return CrawlTask.from_mapping(
            payload=item,
            default_kind=default_kind,
            default_source_type=default_source_type,
            default_priority=default_priority,
            priority_resolver=(
                self._priority_resolver if resolve_missing_priority else None
            ),
            id_generator=self._id_generator,
        )

    def deserialize_dead_letter_task(
        self,
        *,
        item: object,
    ) -> CrawlTask | None:
        """Return a dead-letter task reconstructed from persisted payload."""

        return self.deserialize_task_payload(
            item=item,
            default_source_type="dead_letter",
            resolve_missing_priority=False,
        )

    def deserialize(
        self,
        *,
        item: object,
    ) -> SchedulerTaskEnvelope | None:
        """Return a scheduled task envelope reconstructed from payload."""

        if not isinstance(item, dict):
            return None

        missing_fields = [
            field
            for field in self._REQUIRED_CHECKPOINT_FIELDS
            if field not in item
        ]
        if missing_fields:
            raise ValueError(
                "scheduler checkpoint task missing "
                + ", ".join(missing_fields)
            )

        self._validate_checkpoint_task_fields(item)

        task = self.deserialize_task_payload(
            item=item,
            resolve_missing_priority=False,
        )
        if task is None:
            return None

        normalized_url = self._url_normalizer.normalize(task.url)
        if not normalized_url:
            return None
        priority = coerce_int(item.get("priority"))
        if priority is None:
            restored_task = CrawlTask.with_url_and_resolved_priority(
                task=task,
                url=normalized_url,
                priority_resolver=self._priority_resolver,
            )
        else:
            restored_task = CrawlTask.with_url_and_preserved_priority(
                task=task,
                url=normalized_url,
                priority=priority,
            )

        sequence = coerce_int(item.get("sequence"))
        if sequence is None:
            raise ValueError("scheduler checkpoint task missing sequence")

        host = coerce_str(item.get("host"))

        return SchedulerTaskEnvelope(
            task=restored_task,
            host=host,
            priority=int(restored_task.priority),
            sequence=int(sequence),
        )

    @staticmethod
    def _validate_checkpoint_task_fields(item: dict[str, object]) -> None:
        for field in (
            "url",
            "source_name",
            "task_id",
            "kind",
            "source_type",
        ):
            if (
                not isinstance(item.get(field), str)
                or coerce_str(item.get(field)) is None
            ):
                raise ValueError(f"scheduler checkpoint task missing {field}")

        raw_priority = item.get("priority")
        if isinstance(raw_priority, bool) or not isinstance(raw_priority, int):
            raise ValueError("scheduler checkpoint task missing priority")

        raw_depth = item.get("depth")
        if (
            isinstance(raw_depth, bool)
            or not isinstance(raw_depth, int)
            or raw_depth < 0
        ):
            raise ValueError(
                "scheduler checkpoint task contains invalid depth"
            )

        raw_sequence = item.get("sequence")
        if (
            isinstance(raw_sequence, bool)
            or not isinstance(raw_sequence, int)
            or raw_sequence < 0
        ):
            raise ValueError("scheduler checkpoint task missing sequence")

        raw_host = item.get("host")
        if raw_host is not None and (
            not isinstance(raw_host, str) or not raw_host.strip()
        ):
            raise ValueError("scheduler checkpoint task contains invalid host")

        raw_parent_url = item.get("parent_url")
        if raw_parent_url is not None and not isinstance(raw_parent_url, str):
            raise ValueError(
                "scheduler checkpoint task contains invalid parent_url"
            )

        raw_context = item.get("context")
        if raw_context is not None and not isinstance(raw_context, dict):
            raise ValueError(
                "scheduler checkpoint task contains invalid context"
            )
