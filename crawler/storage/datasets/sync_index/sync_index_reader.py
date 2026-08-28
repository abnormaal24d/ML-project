"""Append-handle management for raw dataset sync indexes."""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from crawler.storage.datasets.run_layout.dataset_path_layout import (
    canonical_modality_filename,
)
from crawler.storage.datasets.sync_index.sync_index_paths import SyncIndexPaths

if TYPE_CHECKING:
    from config.settings.datasets import DatasetPathSettings


class SyncIndexReader:
    """Open and cache append handles for sync-index files."""

    def __init__(
        self,
        *,
        paths: SyncIndexPaths,
        dataset_paths: DatasetPathSettings,
    ) -> None:
        self._paths = paths
        self._dataset_paths = dataset_paths
        self._sync_directory = paths.sync_directory

        with ExitStack() as rollback_stack:
            self._relationships_handle = rollback_stack.enter_context(
                self.open_append_handle(paths.relationships_path),
            )
            self._metadata_handle = rollback_stack.enter_context(
                self.open_append_handle(paths.metadata_path),
            )
            self._updates_handle = rollback_stack.enter_context(
                self.open_append_handle(paths.updates_path),
            )
            self._errors_handle = rollback_stack.enter_context(
                self.open_append_handle(paths.errors_path),
            )
            self._discovered_assets_handle = rollback_stack.enter_context(
                self.open_append_handle(paths.discovered_assets_path),
            )
            self._rejected_assets_handle = rollback_stack.enter_context(
                self.open_append_handle(paths.rejected_assets_path),
            )
            self._superseded_objects_handle = rollback_stack.enter_context(
                self.open_append_handle(paths.superseded_objects_path),
            )
            rollback_stack.pop_all()

        self._modality_manifest_handles: dict[str, TextIO] = {}
        self._canonical_modality_handles: dict[str, TextIO] = {}

    @property
    def relationships_handle(self) -> TextIO:
        return self._relationships_handle

    @property
    def metadata_handle(self) -> TextIO:
        return self._metadata_handle

    @property
    def updates_handle(self) -> TextIO:
        return self._updates_handle

    @property
    def errors_handle(self) -> TextIO:
        return self._errors_handle

    @property
    def discovered_assets_handle(self) -> TextIO:
        return self._discovered_assets_handle

    @property
    def rejected_assets_handle(self) -> TextIO:
        return self._rejected_assets_handle

    @property
    def superseded_objects_handle(self) -> TextIO:
        return self._superseded_objects_handle

    def open_handles(self) -> tuple[TextIO, ...]:
        return (
            self._relationships_handle,
            self._metadata_handle,
            self._updates_handle,
            self._errors_handle,
            self._discovered_assets_handle,
            self._rejected_assets_handle,
            self._superseded_objects_handle,
            *self._modality_manifest_handles.values(),
            *self._canonical_modality_handles.values(),
        )

    def ensure_modality_manifest_handle(self, *, modality: str) -> TextIO:
        existing = self._modality_manifest_handles.get(modality)
        if existing is not None:
            return existing

        path = self._paths.modality_manifest_path(modality=modality)
        handle = self.open_append_handle(path)
        self._modality_manifest_handles[modality] = handle
        return handle

    def ensure_canonical_modality_handle(
        self, *, modality: str
    ) -> TextIO | None:
        existing = self._canonical_modality_handles.get(modality)
        if existing is not None:
            return existing

        filename = canonical_modality_filename(
            dataset_paths=self._dataset_paths,
            modality=modality,
        )
        if filename is None:
            return None

        path = self._sync_directory / filename
        handle = self.open_append_handle(path)
        self._canonical_modality_handles[modality] = handle
        return handle

    def prepare_record_handles(self, *, modality: str) -> tuple[Path, ...]:
        """Open all record index targets and flush earlier committed rows."""

        self.ensure_modality_manifest_handle(modality=modality)
        self.ensure_canonical_modality_handle(modality=modality)
        self.flush_transaction()
        return tuple(Path(handle.name) for handle in self.open_handles())

    def flush_transaction(self) -> None:
        """Durably flush all currently open index handles."""

        for handle in self.open_handles():
            handle.flush()
            os.fsync(handle.fileno())

    def clear_modality_handles(self) -> None:
        self._modality_manifest_handles.clear()
        self._canonical_modality_handles.clear()

    def close(self, *, fsync_enabled: bool) -> None:
        if fsync_enabled:
            self.flush_transaction()

        for handle in self.open_handles():
            handle.close()
        self.clear_modality_handles()

    @staticmethod
    def open_append_handle(path: Path) -> TextIO:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8")
