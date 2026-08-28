"""Crawler runtime dead-letter reader."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from crawler.runtime.state.crawl_dead_letter_file import (
    dead_letter_path_lock,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from config.settings.crawler import CrawlStateStoreSettings
    from crawler.crawl_tasks.crawl_task import CrawlTask
    from crawler.scheduling.checkpointing.scheduler_task_deserializer import (
        SchedulerTaskDeserializer,
    )


@dataclass(frozen=True, slots=True)
class DeadLetterTaskBatch:
    """A bounded set of requeue tasks and their source JSONL records."""

    tasks: tuple[CrawlTask, ...]
    line_numbers: tuple[int, ...]
    serialized_records: tuple[str, ...]


class CrawlerDeadLetterReader:
    """Read and acknowledge crawler tasks from a dead-letter JSONL file."""

    def __init__(
        self,
        *,
        settings: CrawlStateStoreSettings,
        dead_letter_path: Path,
        logger: ProjectLogger,
        task_deserializer: SchedulerTaskDeserializer,
    ) -> None:
        self._settings = settings
        self._dead_letter_path = dead_letter_path
        self._logger = logger
        self._task_deserializer = task_deserializer
        self._file_lock = dead_letter_path_lock(dead_letter_path)

    @property
    def enabled(self) -> bool:
        """Return whether dead-letter recovery is enabled."""

        return self._settings.enabled and self._settings.dead_letter_enabled

    def load_batch(self) -> DeadLetterTaskBatch:
        """Load a bounded batch without modifying its source file."""

        if not self.enabled or self._settings.max_dead_letters_to_requeue == 0:
            return DeadLetterTaskBatch((), (), ())

        path = self._dead_letter_path
        if not path.exists():
            return DeadLetterTaskBatch((), (), ())

        tasks: list[CrawlTask] = []
        line_numbers: list[int] = []
        serialized_records: list[str] = []

        try:
            with (
                self._file_lock,
                path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as handle,
            ):
                for line_number, line in enumerate(handle):
                    stripped = line.strip()
                    if not stripped:
                        continue

                    try:
                        payload = json.loads(stripped)
                        task = self._deserialize_task(payload=payload)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue

                    if task is None:
                        continue

                    tasks.append(task)
                    line_numbers.append(line_number)
                    serialized_records.append(line)
                    if (
                        len(tasks)
                        >= self._settings.max_dead_letters_to_requeue
                    ):
                        break
        except OSError as exc:
            self._logger.warning(
                "crawl_dead_letter_read_failed",
                path=str(path),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return DeadLetterTaskBatch((), (), ())

        return DeadLetterTaskBatch(
            tuple(tasks),
            tuple(line_numbers),
            tuple(serialized_records),
        )

    def acknowledge(
        self,
        *,
        batch: DeadLetterTaskBatch,
        accepted: tuple[bool, ...],
    ) -> int:
        """Remove only records confirmed durable by the runtime caller."""

        if not self._settings.clear_dead_letters_on_requeue:
            return 0
        if len(accepted) != len(batch.tasks):
            raise ValueError("dead-letter acknowledgement length mismatch")

        acknowledged = {
            line_number: serialized
            for line_number, serialized, was_accepted in zip(
                batch.line_numbers,
                batch.serialized_records,
                accepted,
                strict=True,
            )
            if was_accepted
        }
        if not acknowledged:
            return 0

        path = self._dead_letter_path
        with self._file_lock:
            if not path.exists():
                return 0
            with path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                lines = handle.readlines()
            retained: list[str] = []
            removed = 0
            for line_number, line in enumerate(lines):
                expected = acknowledged.get(line_number)
                if expected is not None and line == expected:
                    removed += 1
                    continue
                retained.append(line)

            if removed == 0:
                return 0
            self._replace_lines(path=path, lines=retained)
            return removed

    def _deserialize_task(self, *, payload: object) -> CrawlTask | None:
        if not isinstance(payload, dict):
            return None
        task_payload = payload.get("task")
        if not isinstance(task_payload, dict):
            return None
        return self._task_deserializer.deserialize_dead_letter_task(
            item=task_payload
        )

    @staticmethod
    def _replace_lines(*, path: Path, lines: list[str]) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                prefix=f"{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.writelines(lines)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            _fsync_directory(path.parent)
        except OSError:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise


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


__all__ = [
    "CrawlerDeadLetterReader",
    "DeadLetterTaskBatch",
]
