"""Select the single dataset stage that training may consume."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from config.collection.training_input_gate import TrainingInputMode
from datachecker.fingerprints import FileFingerprintCalculator
from datachecker.inventory.training_snapshot_inventory import (
    TrainingInventory,
)


@dataclass(frozen=True, slots=True)
class SelectedTrainingInput:
    """The outcome of selecting a training dataset."""

    snapshot_id: str | None
    dataset_root: Path | None
    dataset_manifest_hash: str | None

    sample_count: int
    modality_counts: dict[str, int]
    task_counts: dict[str, int]


def _no_checkpoint(_stage: str) -> None:
    """Provide a no-op checkpoint outside a deadline-bound checker run."""


def select_training_input(
    *,
    training_inventory: TrainingInventory,
    augmented_inventory: TrainingInventory,
    training_input_mode: TrainingInputMode,
    augmentation_enabled: bool,
    augmentation_is_valid: bool,
    file_fingerprint_calculator: FileFingerprintCalculator,
    checkpoint: Callable[[str], None] = _no_checkpoint,
) -> SelectedTrainingInput:
    """Return the one valid dataset selected by the configured mode.

    The selected snapshot id comes only from the persisted dataset manifest
    exposed by :class:`TrainingInventory`; directory names are never used as
    an identity fallback.
    """

    if training_input_mode is TrainingInputMode.PREPROCESSED_ONLY:
        return _from_inventory(
            inventory=training_inventory,
            file_fingerprint_calculator=file_fingerprint_calculator,
            checkpoint=checkpoint,
        )

    can_use_augmented = augmentation_enabled and augmentation_is_valid
    if can_use_augmented:
        selected_augmented = _from_inventory(
            inventory=augmented_inventory,
            file_fingerprint_calculator=file_fingerprint_calculator,
            checkpoint=checkpoint,
        )
        if selected_augmented.dataset_root is not None:
            return selected_augmented

    if training_input_mode is TrainingInputMode.AUGMENTED_REQUIRED:
        return _empty_selection()

    return _from_inventory(
        inventory=training_inventory,
        file_fingerprint_calculator=file_fingerprint_calculator,
        checkpoint=checkpoint,
    )


def _from_inventory(
    *,
    inventory: TrainingInventory,
    file_fingerprint_calculator: FileFingerprintCalculator,
    checkpoint: Callable[[str], None],
) -> SelectedTrainingInput:
    """Build a selection only for a finalized, manifest-owned snapshot."""

    if (
        not inventory.schema_valid
        or inventory.directory is None
        or inventory.manifest_path is None
        or inventory.snapshot_id is None
        or not inventory.manifest_path.is_file()
    ):
        return _empty_selection()

    checkpoint("selected_training_manifest_hash")
    return SelectedTrainingInput(
        snapshot_id=inventory.snapshot_id,
        dataset_root=inventory.directory,
        dataset_manifest_hash=file_fingerprint_calculator.calculate(
            path=inventory.manifest_path,
            checkpoint=checkpoint,
        ),
        sample_count=inventory.sample_count,
        modality_counts=dict(inventory.modality_counts),
        task_counts=dict(inventory.task_counts),
    )


def _empty_selection() -> SelectedTrainingInput:
    """Return the explicit absence of a trainable input."""

    return SelectedTrainingInput(
        snapshot_id=None,
        dataset_root=None,
        dataset_manifest_hash=None,
        sample_count=0,
        modality_counts={},
        task_counts={},
    )
