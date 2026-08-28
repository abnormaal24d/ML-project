"""Crawler runtime dead-letter writer."""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any

from crawler.runtime.state.crawl_dead_letter_file import (
    dead_letter_path_lock,
)
from crawler.scheduling.completion.dead_letter_writer import DeadLetterRecord
from logger.project_logger import ProjectLogger
from shared.runtime_primitives import Clock

if TYPE_CHECKING:
    from pathlib import Path

    from config.settings.crawler import CrawlStateStoreSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask


class CrawlerDeadLetterWriter:
    """Append terminal crawler tasks to a durable JSONL recovery log."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        settings: CrawlStateStoreSettings,
        dead_letter_path: Path,
        logger: ProjectLogger,
        clock: Clock,
    ) -> None:
        self._settings = settings
        self._dead_letter_path = dead_letter_path
        self._logger = logger
        self._clock = clock
        self._file_lock = dead_letter_path_lock(dead_letter_path)
        if self.enabled:
            self._dead_letter_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        """Return whether dead-letter persistence is enabled."""

        return self._settings.enabled and self._settings.dead_letter_enabled

    async def append(self, record: DeadLetterRecord) -> None:
        """Persist one record without blocking the scheduler event loop."""

        write_task = asyncio.create_task(
            asyncio.to_thread(self._append_record, record),
            name="crawler-dead-letter-file-append",
        )
        while not write_task.done():
            try:
                await asyncio.shield(write_task)
            except asyncio.CancelledError:
                # Appending one JSONL record is an atomic durability boundary.
                # The scheduler wraps this operation in its own shield and will
                # propagate caller cancellation after completion. If global
                # shutdown also cancels this task, finish the disk operation so
                # its outcome is never ambiguous.
                continue
        write_task.result()

    def _append_record(self, record: DeadLetterRecord) -> None:
        """Persist one eligible record on the writer thread."""

        if not self.enabled:
            return
        if record.status not in self._settings.dead_letter_statuses:
            return

        try:
            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "recorded_at": self._clock.now().isoformat(),
                "status": record.status,
                "original_outcome": record.original_outcome,
                "detail": record.detail,
                "task": self._serialize_task(record.task),
                "fields": self._json_safe(record.fields),
            }
            encoded = (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

            path = self._dead_letter_path
            with self._file_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                created = not path.exists()
                needs_separator = self._has_unterminated_tail(path)
                with path.open("ab") as handle:
                    if needs_separator:
                        handle.write(b"\n")
                    handle.write(encoded.encode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())
                if created:
                    _fsync_directory(path.parent)
        except (OSError, TypeError, ValueError) as exc:
            self._logger.error(
                "crawl_dead_letter_write_failed",
                path=str(self._dead_letter_path),
                task_id=record.task.task_id,
                url=record.task.url,
                status=record.status,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    @staticmethod
    def _has_unterminated_tail(path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"

    @classmethod
    def _serialize_task(cls, task: CrawlTask) -> dict[str, object]:
        return {
            "url": task.url,
            "source_name": task.source_name,
            "task_id": task.task_id,
            "kind": task.kind.value,
            "depth": task.depth,
            "source_type": task.source_type,
            "parent_url": task.parent_url,
            "priority": task.priority,
            "context": (
                task.context.to_dict() if task.context is not None else None
            ),
        }

    @classmethod
    def _json_safe(cls, value: Any) -> object:
        if value is None or isinstance(value, str | int | bool):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, Mapping):
            return {
                str(key): cls._json_safe(item)
                for key, item in sorted(
                    value.items(),
                    key=lambda entry: str(entry[0]),
                )
            }
        if isinstance(value, tuple | list | set | frozenset):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, Enum):
            return cls._json_safe(value.value)
        if hasattr(value, "isoformat"):
            return str(value.isoformat())
        return str(value)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = ["CrawlerDeadLetterWriter"]
