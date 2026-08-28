"""Training-input selection uses only manifest-owned snapshot identifiers."""

from __future__ import annotations

from pathlib import Path

from config.collection.training_input_gate import TrainingInputMode
from datachecker.fingerprints import FileFingerprintCalculator
from datachecker.inventory.training_snapshot_inventory import TrainingInventory
from datachecker.training_input_selection import (
    SelectedTrainingInput,
    select_training_input,
)


def _inventory(
    tmp_path: Path,
    *,
    directory_name: str,
    snapshot_id: str | None,
    sample_count: int = 10,
) -> TrainingInventory:
    directory = tmp_path / directory_name
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "dataset_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return TrainingInventory(
        directory=directory,
        manifest_path=manifest_path,
        stats_path=None,
        snapshot_id=snapshot_id,
        fingerprint="inventory-fingerprint",
        sample_count=sample_count,
        modality_counts={"text": sample_count},
        task_counts={"text_pretrain": sample_count},
        variants_by_modality={},
        variants_by_operation={},
        rejections_by_modality={},
        media_outputs={},
        quality_checks_passed=True,
        rejected_augmented_count=0,
        schema_valid=True,
    )


def _empty_inventory() -> TrainingInventory:
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


def _select(
    *,
    training_inventory: TrainingInventory,
    augmented_inventory: TrainingInventory,
    mode: TrainingInputMode,
    augmentation_enabled: bool,
    augmentation_is_valid: bool,
) -> SelectedTrainingInput:
    return select_training_input(
        training_inventory=training_inventory,
        augmented_inventory=augmented_inventory,
        training_input_mode=mode,
        augmentation_enabled=augmentation_enabled,
        augmentation_is_valid=augmentation_is_valid,
        file_fingerprint_calculator=FileFingerprintCalculator(),
    )


def test_preprocessed_selection_uses_manifest_owned_snapshot_id(
    tmp_path: Path,
) -> None:
    inventory = _inventory(
        tmp_path,
        directory_name="directory-name-is-not-an-id",
        snapshot_id="canonical-training-id",
    )

    selected = _select(
        training_inventory=inventory,
        augmented_inventory=_empty_inventory(),
        mode=TrainingInputMode.PREPROCESSED_ONLY,
        augmentation_enabled=False,
        augmentation_is_valid=False,
    )

    assert selected.snapshot_id == "canonical-training-id"
    assert selected.dataset_root == inventory.directory
    assert selected.dataset_manifest_hash
    assert selected.snapshot_id != inventory.directory.name


def test_selection_never_uses_directory_name_as_snapshot_id(
    tmp_path: Path,
) -> None:
    inventory = _inventory(
        tmp_path,
        directory_name="looks-like-a-snapshot-id",
        snapshot_id=None,
    )

    selected = _select(
        training_inventory=inventory,
        augmented_inventory=_empty_inventory(),
        mode=TrainingInputMode.PREPROCESSED_ONLY,
        augmentation_enabled=False,
        augmentation_is_valid=False,
    )

    assert selected.snapshot_id is None
    assert selected.dataset_root is None
    assert selected.dataset_manifest_hash is None


def test_augmented_selection_uses_augmented_manifest_snapshot_id(
    tmp_path: Path,
) -> None:
    preprocessed = _inventory(
        tmp_path,
        directory_name="preprocessed",
        snapshot_id="preprocessed-id",
    )
    augmented = _inventory(
        tmp_path,
        directory_name="augmented",
        snapshot_id="augmented-id",
        sample_count=15,
    )

    selected = _select(
        training_inventory=preprocessed,
        augmented_inventory=augmented,
        mode=TrainingInputMode.AUGMENTED_WHEN_AVAILABLE,
        augmentation_enabled=True,
        augmentation_is_valid=True,
    )

    assert selected.snapshot_id == "augmented-id"
    assert selected.dataset_root == augmented.directory
    assert selected.sample_count == 15


def test_augmented_required_mode_refuses_unavailable_augmented_output(
    tmp_path: Path,
) -> None:
    preprocessed = _inventory(
        tmp_path,
        directory_name="preprocessed",
        snapshot_id="preprocessed-id",
    )

    selected = _select(
        training_inventory=preprocessed,
        augmented_inventory=_empty_inventory(),
        mode=TrainingInputMode.AUGMENTED_REQUIRED,
        augmentation_enabled=True,
        augmentation_is_valid=False,
    )

    assert selected.snapshot_id is None
    assert selected.dataset_root is None
