"""Training and augmented dataset artifact inventory reader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypedDict

from schemas.multimodal_tasks import canonical_task_name

DeadlineCheckpoint = Callable[[str], None]


def no_deadline_checkpoint(stage: str) -> None:
    pass


if TYPE_CHECKING:
    from config.collection.training_input_gate import DataCheckerSettings
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from datachecker.fingerprints import DatasetFingerprintCalculator

_MANAGED_SNAPSHOT_PREFIXES = ("_tmp", "_repair", "_staging")


@dataclass(slots=True, frozen=True)
class TrainingInventory:
    """Resolved training-like dataset artifacts."""

    directory: Path | None
    manifest_path: Path | None
    stats_path: Path | None
    snapshot_id: str | None
    fingerprint: str | None
    sample_count: int
    modality_counts: dict[str, int]
    task_counts: dict[str, int]
    variants_by_modality: dict[str, int]
    variants_by_operation: dict[str, int]
    rejections_by_modality: dict[str, int]
    media_outputs: dict[str, object]
    quality_checks_passed: bool
    rejected_augmented_count: int
    schema_valid: bool


@dataclass(frozen=True, slots=True)
class TrainingInventoryReaderConfig:
    """File layout settings needed for training inventory reads."""

    splits_directory: str
    train_filename: str
    val_filename: str
    test_filename: str


class _TrainingSnapshotCandidate(TypedDict):
    """Typed paths and metadata for one selectable training snapshot."""

    directory: Path
    manifest_path: Path
    stats_path: Path
    split_paths: tuple[Path, Path, Path]
    manifest_payload: dict[str, object]
    latest_mtime: float


class TrainingInventoryReader:
    """Resolve the latest training and augmented dataset outputs."""

    def __init__(
        self,
        *,
        settings: DataCheckerSettings,
        artifact_path_registry: ArtifactPathRegistry,
        dataset_fingerprint_calculator: DatasetFingerprintCalculator,
        config: TrainingInventoryReaderConfig,
    ) -> None:
        self._settings = settings
        self._artifact_path_registry = artifact_path_registry
        self._config = config
        self._dataset_fingerprint_calculator = dataset_fingerprint_calculator

    def read_training(
        self,
        *,
        checkpoint: DeadlineCheckpoint = no_deadline_checkpoint,
    ) -> TrainingInventory:
        """Build base training dataset inventory."""
        return self._read_training_like_inventory(
            root=self._artifact_path_registry.training_sets_root(),
            checkpoint=checkpoint,
        )

    def read_augmented(
        self,
        *,
        checkpoint: DeadlineCheckpoint = no_deadline_checkpoint,
    ) -> TrainingInventory:
        """Build augmented training dataset inventory."""
        return self._read_training_like_inventory(
            root=self._artifact_path_registry.augmented_training_sets_root(),
            checkpoint=checkpoint,
        )

    def _read_training_like_inventory(
        self,
        *,
        root: Path,
        checkpoint: DeadlineCheckpoint,
    ) -> TrainingInventory:
        if not root.exists():
            return self._empty_inventory()

        candidates: list[_TrainingSnapshotCandidate] = []
        dataset_manifest_filename = self._artifact_path_registry.dataset_paths.dataset_manifest_filename

        for i, manifest_path in enumerate(
            root.rglob(dataset_manifest_filename)
        ):
            if i % 256 == 0:
                checkpoint("training_snapshot_scan")
            if not manifest_path.is_file():
                continue

            directory = manifest_path.parent
            if not self._is_selectable_snapshot_directory(
                directory, root=root
            ):
                continue

            stats_path = (
                directory
                / self._artifact_path_registry.dataset_paths.stats_filename
            )
            if not stats_path.is_file():
                continue

            splits_root = directory / self._config.splits_directory
            split_paths = (
                splits_root / self._config.train_filename,
                splits_root / self._config.val_filename,
                splits_root / self._config.test_filename,
            )
            if any(not p.is_file() for p in split_paths):
                continue

            payload = self._read_json(manifest_path)
            if (
                payload.get("final") is not True
                or str(payload.get("status") or "").strip() != "completed"
                or payload.get("valid") is not True
            ):
                continue

            candidates.append(
                {
                    "directory": directory,
                    "manifest_path": manifest_path,
                    "stats_path": stats_path,
                    "split_paths": split_paths,
                    "manifest_payload": payload,
                    "latest_mtime": max(
                        p.stat().st_mtime
                        for p in (manifest_path, stats_path, *split_paths)
                        if p.exists()
                    ),
                }
            )

        if not candidates:
            return self._empty_inventory()

        best = max(candidates, key=lambda c: c["latest_mtime"])
        augmentation_report = self._read_json(
            best["directory"] / "augmentation_report.json"
        )

        required_paths = (
            best["manifest_path"],
            best["stats_path"],
            *best["split_paths"],
        )
        fingerprint = None
        if required_paths:
            fingerprint = self._dataset_fingerprint_calculator.calculate(
                paths=tuple(p for p in required_paths if p.exists()),
                root=best["directory"],
                checkpoint=checkpoint,
            )

        return TrainingInventory(
            directory=best["directory"],
            manifest_path=best["manifest_path"],
            stats_path=best["stats_path"],
            snapshot_id=self._resolve_snapshot_id(best["manifest_payload"]),
            fingerprint=fingerprint,
            sample_count=self._resolve_training_sample_count(
                best["manifest_payload"]
            ),
            modality_counts=self._resolve_modality_counts(
                best["manifest_payload"]
            ),
            task_counts=self._resolve_task_counts(best["manifest_payload"]),
            variants_by_modality=self._resolve_int_mapping(
                augmentation_report.get("variants_by_modality")
            ),
            variants_by_operation=self._resolve_int_mapping(
                augmentation_report.get("variants_by_operation")
            ),
            rejections_by_modality=self._resolve_int_mapping(
                augmentation_report.get("rejections_by_modality")
            ),
            media_outputs=self._resolve_object_mapping(
                augmentation_report.get("media_outputs")
            ),
            quality_checks_passed=(
                augmentation_report.get("quality_checks_passed") is True
            ),
            rejected_augmented_count=self._resolve_count(
                augmentation_report.get("rejected_augmented_count")
            ),
            schema_valid=True,
        )

    def _empty_inventory(self) -> TrainingInventory:
        return TrainingInventory(
            directory=None,
            manifest_path=None,
            stats_path=None,
            snapshot_id=None,
            fingerprint=None,
            sample_count=0,
            modality_counts={},
            task_counts={},
            variants_by_modality={},
            variants_by_operation={},
            rejections_by_modality={},
            media_outputs={},
            quality_checks_passed=False,
            rejected_augmented_count=0,
            schema_valid=False,
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        if not path.is_file():
            return {}
        try:
            val = json.loads(path.read_text("utf-8"))
            return val if isinstance(val, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _is_selectable_snapshot_directory(
        path: Path | None,
        *,
        root: Path | None = None,
    ) -> bool:
        if path is None or not path.is_dir():
            return False
        name = path.name.strip()
        if (
            not name
            or name in {".", ".."}
            or any(
                name.startswith(prefix)
                for prefix in _MANAGED_SNAPSHOT_PREFIXES
            )
        ):
            return False
        if root is not None:
            try:
                relative_parts = path.relative_to(root).parts
            except ValueError:
                relative_parts = path.parts
            if any(
                part.startswith(_MANAGED_SNAPSHOT_PREFIXES)
                for part in relative_parts
            ):
                return False
        return True

    @staticmethod
    def _resolve_snapshot_id(payload: dict[str, object]) -> str | None:
        raw = payload.get("snapshot_id")
        if not isinstance(raw, str):
            return None
        value = raw.strip()
        if (
            not value
            or value != raw
            or not all(
                character.isalnum() or character in {"-", "_"}
                for character in value
            )
        ):
            return None
        return value

    @staticmethod
    def _resolve_training_sample_count(payload: dict[str, object]) -> int:
        splits = payload.get("splits")
        if not isinstance(splits, dict):
            return 0
        return sum(
            TrainingInventoryReader._resolve_count(value)
            for value in splits.values()
        )

    @staticmethod
    def _resolve_modality_counts(payload: dict[str, object]) -> dict[str, int]:
        raw_counts = payload.get("modalities")
        if not isinstance(raw_counts, dict):
            return {}
        return {
            str(key): TrainingInventoryReader._resolve_count(value)
            for key, value in raw_counts.items()
        }

    @staticmethod
    def _resolve_task_counts(payload: dict[str, object]) -> dict[str, int]:
        raw_counts = payload.get("tasks")
        if not isinstance(raw_counts, dict):
            return {}
        counts: dict[str, int] = {}
        for task_type, value in raw_counts.items():
            normalized = canonical_task_name(task_type)
            counts[normalized] = counts.get(
                normalized, 0
            ) + TrainingInventoryReader._resolve_count(value)
        return counts

    @staticmethod
    def _resolve_int_mapping(value: object) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        return {
            str(key): TrainingInventoryReader._resolve_count(raw_count)
            for key, raw_count in value.items()
        }

    @staticmethod
    def _resolve_object_mapping(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        return {str(k): v for k, v in value.items()}

    @staticmethod
    def _resolve_count(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return 0
        try:
            return max(0, int(value))
        except (OverflowError, ValueError):
            return 0
