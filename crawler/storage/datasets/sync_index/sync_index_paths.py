"""Filesystem paths used by the raw sync-index writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from crawler.storage.datasets.run_layout.dataset_path_layout import (
    canonical_modality_filename,
)

if TYPE_CHECKING:
    from config.settings.datasets import DatasetPathSettings


@dataclass(frozen=True)
class SyncIndexPaths:
    """Resolved filesystem paths used by the raw sync-index writer."""

    run_directory: Path
    sync_directory: Path
    sync_by_modality_directory: Path
    relationships_path: Path
    metadata_path: Path
    updates_path: Path
    errors_path: Path
    discovered_assets_path: Path
    rejected_assets_path: Path
    current_objects_path: Path
    superseded_objects_path: Path
    summary_path: Path

    @classmethod
    def from_settings(
        cls,
        *,
        run_directory: Path,
        dataset_paths: DatasetPathSettings,
    ) -> SyncIndexPaths:
        """Build all raw sync-index paths from dataset path settings."""

        sync_directory = run_directory / dataset_paths.raw_sync_directory

        return cls(
            run_directory=run_directory,
            sync_directory=sync_directory,
            sync_by_modality_directory=(
                sync_directory / dataset_paths.raw_sync_by_modality_directory
            ),
            relationships_path=(
                sync_directory / dataset_paths.raw_sync_relationships_filename
            ),
            metadata_path=(
                sync_directory / dataset_paths.raw_sync_metadata_filename
            ),
            updates_path=(
                sync_directory / dataset_paths.raw_sync_updates_filename
            ),
            errors_path=(
                sync_directory / dataset_paths.raw_sync_errors_filename
            ),
            discovered_assets_path=(
                sync_directory
                / dataset_paths.raw_sync_discovered_assets_filename
            ),
            rejected_assets_path=(
                sync_directory
                / dataset_paths.raw_sync_rejected_assets_filename
            ),
            current_objects_path=(
                sync_directory
                / dataset_paths.raw_sync_current_objects_filename
            ),
            superseded_objects_path=(
                sync_directory
                / dataset_paths.raw_sync_superseded_objects_filename
            ),
            summary_path=(
                sync_directory / dataset_paths.raw_sync_summary_filename
            ),
        )

    def ensure_directories(self) -> None:
        """Create sync-index directories if they do not exist yet."""

        self.sync_directory.mkdir(parents=True, exist_ok=True)
        self.sync_by_modality_directory.mkdir(parents=True, exist_ok=True)

    def relative_to_run(self, path: Path) -> str:
        """Return a POSIX-style path relative to the run directory."""

        return path.relative_to(self.run_directory).as_posix()

    def modality_manifest_path(self, *, modality: str) -> Path:
        """Return the per-modality manifest path."""

        return self.sync_by_modality_directory / f"{modality}.jsonl"

    def canonical_modality_path(
        self,
        *,
        dataset_paths: DatasetPathSettings,
        modality: str,
    ) -> Path | None:
        """Return the canonical modality manifest path, if supported."""

        filename = canonical_modality_filename(
            dataset_paths=dataset_paths,
            modality=modality,
        )
        if filename is None:
            return None

        return self.sync_directory / filename
