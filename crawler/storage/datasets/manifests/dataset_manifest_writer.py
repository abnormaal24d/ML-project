"""Raw dataset manifest append-only writer."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from pathlib import Path

    from config.settings.datasets import RawDatasetWriterSettings
    from crawler.storage.datasets.records.dataset_record import DatasetRecord


class DatasetManifestWriter:
    """Append raw dataset records to the canonical manifest."""

    def __init__(
        self,
        *,
        settings: RawDatasetWriterSettings,
        manifest_path: Path,
    ) -> None:
        self._settings = settings
        self._manifest_path = manifest_path
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self._manifest_path.open(
            "a",
            encoding="utf-8",
        )
        self._write_count = 0

    @property
    def write_count(self) -> int:
        """Return the number of manifest records written."""

        return self._write_count

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def prepare_transaction(self) -> int:
        """Flush prior rows and return the current logical row count."""

        self._flush()
        return self._write_count

    def flush_transaction(self) -> None:
        """Durably flush rows written by the active transaction."""

        self._flush()
        os.fsync(self._handle.fileno())

    def restore_transaction_count(self, write_count: int) -> None:
        self._write_count = write_count

    @property
    def closed(self) -> bool:
        """Return whether the manifest handle has already been closed."""

        return self._handle.closed

    def append(self, record: DatasetRecord) -> bool:
        """Append a raw dataset record and return whether flushing occurred."""

        self._ensure_open()

        self.write_jsonl_row(self._handle, record.model_dump(mode="json"))
        self._write_count += 1

        flush_every = max(1, self._settings.manifest_flush_every_records)
        if self._write_count % flush_every != 0:
            return False

        self._flush()

        if self._should_fsync_current_write():
            os.fsync(self._handle.fileno())

        return True

    def close(self) -> None:
        """Flush, optionally fsync, and close the manifest handle."""

        if self._handle.closed:
            return

        self._flush()

        if self._settings.manifest_fsync_enabled:
            os.fsync(self._handle.fileno())

        self._handle.close()

    def _flush(self) -> None:
        self._handle.flush()

    def _should_fsync_current_write(self) -> bool:
        if not self._settings.manifest_fsync_enabled:
            return False

        fsync_every = max(1, self._settings.manifest_fsync_every_records)
        return self._write_count % fsync_every == 0

    def _ensure_open(self) -> None:
        if self._handle.closed:
            raise RuntimeError("Cannot append to a closed manifest writer.")

    @staticmethod
    def write_jsonl_row(
        handle: TextIO,
        payload: Mapping[str, Any],
    ) -> None:
        """Write one JSON-serializable payload as a JSONL row."""

        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
