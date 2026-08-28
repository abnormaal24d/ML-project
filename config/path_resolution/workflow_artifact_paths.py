"""Canonical path registry for workflow artifact and output locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.collection.training_input_gate import DataCheckerSettings
    from config.settings.datasets import DatasetPathSettings


@dataclass(frozen=True, slots=True)
class ArtifactPathRegistry:
    """Resolve canonical artifact paths from already-resolved settings."""

    settings: DataCheckerSettings
    dataset_paths: DatasetPathSettings

    def artifacts_root(self) -> Path:
        return Path(self.dataset_paths.workflow_artifacts_directory)

    def crawl_manifest_path(self) -> Path:
        return self.artifacts_root() / self.settings.crawl_manifest_filename

    def crawl_state_manifest_path(self) -> Path:
        return (
            self.artifacts_root() / self.settings.crawl_state_manifest_filename
        )

    def preprocessing_manifest_path(self) -> Path:
        return (
            self.artifacts_root()
            / self.settings.preprocessing_manifest_filename
        )

    def augmentation_manifest_path(self) -> Path:
        return (
            self.artifacts_root()
            / self.settings.augmentation_manifest_filename
        )

    def training_manifest_path(self) -> Path:
        return self.artifacts_root() / self.settings.training_manifest_filename

    def raw_runs_root(self) -> Path:
        return Path(self.dataset_paths.raw_output_directory)

    def curated_root(self) -> Path:
        return Path(self.dataset_paths.curated_output_directory)

    def training_sets_root(self) -> Path:
        return Path(self.dataset_paths.training_output_directory)

    def augmented_training_sets_root(self) -> Path:
        return Path(self.dataset_paths.augmented_training_output_directory)

    def checkpoint_root(self) -> Path:
        return Path(self.dataset_paths.training_checkpoint_directory)
