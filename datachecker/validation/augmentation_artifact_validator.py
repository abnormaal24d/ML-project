"""Validation for augmentation workflow artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from datachecker.validation.shared_validation import ArtifactPathPresence
from datachecker.workflow_decision import (
    ValidationResult,
    WorkflowDecisionReason,
)

if TYPE_CHECKING:
    from datachecker.inventory.training_snapshot_inventory import (
        TrainingInventory,
    )
    from datachecker.manifests.augmentation_manifest import (
        AugmentationManifest,
    )


class AugmentationArtifactValidator:
    """
    Validate augmentation output against preprocessing and settings state.
    """

    def validate(
        self,
        *,
        manifest: AugmentationManifest | None,
        augmented_inventory: TrainingInventory,
        current_preprocessing_manifest_hash: str | None,
        current_augmentation_settings_hash: str,
        current_augmentation_strategy_hash: str,
        augmentation_enabled: bool,
    ) -> ValidationResult:
        """Validate augmentation outputs or short-circuit when disabled."""

        if not augmentation_enabled:
            return ValidationResult.valid(
                reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
                details=("augmentation is disabled",),
            )
        if manifest is None:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_OUTPUT_MISSING,
                details=("top-level augmentation manifest is missing",),
            )
        if current_preprocessing_manifest_hash is None:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_INPUT_CHANGED,
                details=(
                    "current preprocessing manifest hash is unavailable",
                ),
            )
        artifact_validation = self._validate_manifest_artifacts(
            manifest=manifest,
            augmented_inventory=augmented_inventory,
        )
        if artifact_validation is not None:
            return artifact_validation
        if (
            manifest.preprocessing_manifest_hash
            != current_preprocessing_manifest_hash
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_INPUT_CHANGED,
                details=(
                    "preprocessing manifest hash changed since augmentation "
                    "ran",
                    f"stored={manifest.preprocessing_manifest_hash}",
                    f"current={current_preprocessing_manifest_hash}",
                ),
            )
        if (
            manifest.augmentation_settings_hash
            != current_augmentation_settings_hash
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_SETTINGS_CHANGED,
                details=(
                    "augmentation settings fingerprint changed",
                    f"stored={manifest.augmentation_settings_hash}",
                    f"current={current_augmentation_settings_hash}",
                ),
            )
        if (
            manifest.augmentation_strategy_hash
            != current_augmentation_strategy_hash
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_SETTINGS_CHANGED,
                details=(
                    "augmentation strategy fingerprint changed",
                    f"stored={manifest.augmentation_strategy_hash}",
                    f"current={current_augmentation_strategy_hash}",
                ),
            )
        if augmented_inventory.fingerprint != manifest.output_fingerprint:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_OUTPUT_INVALID,
                details=(
                    "augmentation output fingerprint no longer matches",
                    f"stored={manifest.output_fingerprint}",
                    f"current={augmented_inventory.fingerprint}",
                ),
            )
        media_validation_errors = self._validate_generated_media_outputs(
            manifest=manifest,
        )
        if media_validation_errors:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_OUTPUT_INVALID,
                details=media_validation_errors,
            )
        return ValidationResult.valid(
            reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
        )

    @staticmethod
    def _validate_manifest_artifacts(
        *,
        manifest: AugmentationManifest,
        augmented_inventory: TrainingInventory,
    ) -> ValidationResult | None:
        if (
            manifest.lifecycle_stage != "augmented"
            or manifest.status != "completed"
            or not manifest.final
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_OUTPUT_INVALID,
                details=(
                    "augmentation manifest is not finalized",
                    f"lifecycle_stage={manifest.lifecycle_stage}",
                    f"status={manifest.status}",
                    f"final={manifest.final}",
                ),
            )
        if (
            augmented_inventory.directory is None
            or augmented_inventory.manifest_path is None
            or not augmented_inventory.schema_valid
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_OUTPUT_MISSING,
                details=("augmented dataset output is missing",),
            )
        absent_paths = ArtifactPathPresence.missing(
            manifest.training_snapshot_directory,
            manifest.augmented_training_directory,
            manifest.augmented_dataset_manifest_path,
        )
        if absent_paths:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_OUTPUT_MISSING,
                details=(
                    "augmentation manifest references missing "
                    "physical artifacts",
                    *absent_paths,
                ),
            )
        if (
            manifest.augmented_training_directory
            != augmented_inventory.directory
            or manifest.augmented_dataset_manifest_path
            != augmented_inventory.manifest_path
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.AUGMENTATION_OUTPUT_INVALID,
                details=(
                    "augmentation manifest does not match the selected "
                    "augmented snapshot",
                ),
            )
        return None

    @staticmethod
    def _validate_generated_media_outputs(
        *,
        manifest: AugmentationManifest,
    ) -> tuple[str, ...]:
        if not manifest.quality_checks_passed:
            return ("augmentation quality checks did not pass",)
        missing_count = AugmentationArtifactValidator._nested_int(
            payload=manifest.media_outputs,
            key="missing_generated_files",
        )
        if missing_count > 0:
            return (f"missing generated media files: {missing_count}",)

        training_directory = manifest.augmented_training_directory
        if training_directory is None:
            return ("augmentation training directory is missing",)
        lineage_path = training_directory / ("augmentation_lineage.jsonl")
        if not lineage_path.is_file():
            return ()
        errors: list[str] = []
        for row in AugmentationArtifactValidator._iter_jsonl(
            path=lineage_path
        ):
            if not row.get("media_transform_applied"):
                continue
            raw_path = row.get("output_path")
            if not raw_path:
                errors.append("augmentation lineage row missing output_path")
                continue
            output_path = Path(str(raw_path))
            if not output_path.is_absolute():
                output_path = training_directory / output_path
            if not output_path.is_file() or output_path.stat().st_size <= 0:
                errors.append(f"generated media output missing: {output_path}")
        return tuple(errors)

    @staticmethod
    def _nested_int(*, payload: Mapping[str, object], key: str) -> int:
        value = payload.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _iter_jsonl(*, path: Path) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    if isinstance(payload, dict):
                        rows.append(payload)
        except (OSError, json.JSONDecodeError):
            return ()
        return tuple(rows)
