"""Validation for model-training artifacts."""

from __future__ import annotations

import json
import math
from numbers import Real
from typing import TYPE_CHECKING

from datachecker.validation.shared_validation import ArtifactPathPresence
from datachecker.workflow_decision import (
    ValidationResult,
    WorkflowDecisionReason,
)
from schemas.autonomous_readiness import missing_autonomous_tasks
from training.runtime.checkpoint.io import (
    checkpoint_is_available,
    checkpoint_sha256,
)

if TYPE_CHECKING:
    from pathlib import Path

    from config.collection.training_input_gate import TrainingInputMode
    from datachecker.manifests.training_manifest import TrainingManifest


class TrainingArtifactValidator:
    """Validate trained model output against the selected dataset stage."""

    def __init__(
        self,
        *,
        minimum_samples: int,
        minimum_modality_counts: dict[str, int],
        minimum_task_counts: dict[str, int],
        require_autonomous_multimodal_readiness: bool,
    ) -> None:
        self._minimum_samples = minimum_samples
        self._minimum_modality_counts = dict(minimum_modality_counts)
        self._minimum_task_counts = dict(minimum_task_counts)
        self._require_autonomous = require_autonomous_multimodal_readiness

    def validate(
        self,
        *,
        manifest: TrainingManifest | None,
        training_input_mode: TrainingInputMode,
        current_dataset_manifest_hash: str | None,
        current_checkpoint_path: Path | None,
        current_metrics_path: Path | None,
        current_training_config_fingerprint: str,
        current_model_config_fingerprint: str,
        current_modality_counts: dict[str, int],
        current_task_counts: dict[str, int],
        current_sample_count: int,
    ) -> ValidationResult:
        """
        Validate checkpoint state against the selected dataset and settings.
        """

        sample_result = self._validate_sample_counts(
            training_input_mode=training_input_mode,
            current_dataset_manifest_hash=current_dataset_manifest_hash,
            current_modality_counts=current_modality_counts,
            current_task_counts=current_task_counts,
            current_sample_count=current_sample_count,
        )
        if sample_result is not None:
            return sample_result

        # Full inlining of small validators (manifest, file_existence, rejection_report)
        # Consolidated into TrainingArtifactValidator to avoid micro-modules.
        if manifest is None:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.TRAINING_OUTPUT_FOR_SELECTED_DATASET_MISSING,
                details=("training manifest missing",),
            )

        manifest_result = _validate_manifest_schema(
            manifest=manifest,
            current_dataset_fingerprint=str(
                current_dataset_manifest_hash or ""
            ).strip(),
            current_training_config_fingerprint=(
                current_training_config_fingerprint
            ),
            current_model_config_fingerprint=current_model_config_fingerprint,
            current_checkpoint_path=current_checkpoint_path,
            current_metrics_path=current_metrics_path,
        )
        if manifest_result is not None:
            return manifest_result

        try:
            metrics_payload = _read_json_object(path=manifest.metrics_path)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.TRAINING_OUTPUT_INVALID,
                details=(
                    f"training metrics unreadable: {type(exc).__name__}",
                ),
            )

        schema_errors = _metrics_schema_errors(
            manifest=manifest,
            metrics_payload=metrics_payload,
        )
        if schema_errors:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.TRAINING_OUTPUT_INVALID,
                details=schema_errors,
            )

        rejection_result = _validate_rejection_report(
            manifest=manifest,
            metrics_payload=metrics_payload,
        )
        if rejection_result is not None:
            return rejection_result

        evidence_errors = self._training_evidence_errors(
            metrics_payload,
            current_modality_counts,
        )
        if evidence_errors:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.TRAINING_OUTPUT_INVALID,
                details=tuple(evidence_errors),
            )

        return ValidationResult.valid(
            reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
        )

    def _validate_sample_counts(
        self,
        *,
        training_input_mode: TrainingInputMode,
        current_dataset_manifest_hash: str | None,
        current_modality_counts: dict[str, int],
        current_task_counts: dict[str, int],
        current_sample_count: int,
    ) -> ValidationResult | None:
        if current_dataset_manifest_hash is None:
            return None  # Let manifest checks handle missing input

        errors = []
        if current_sample_count < self._minimum_samples:
            errors.append(
                f"insufficient total samples (have {current_sample_count}, need {self._minimum_samples})"
            )

        for modality, minimum in self._minimum_modality_counts.items():
            current = current_modality_counts.get(modality, 0)
            if current < minimum:
                errors.append(
                    f"insufficient {modality} samples (have {current}, need {minimum})"
                )

        for task, minimum in self._minimum_task_counts.items():
            current = current_task_counts.get(task, 0)
            if current < minimum:
                errors.append(
                    f"insufficient task {task} samples (have {current}, need {minimum})"
                )

        if self._require_autonomous:
            missing_autonomous = missing_autonomous_tasks(current_task_counts)
            if missing_autonomous:
                errors.append(
                    "missing autonomous multimodal readiness tasks: "
                    f"{list(missing_autonomous)}"
                )

        if errors:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.TRAINING_BLOCKED_BY_DATASET_SELECTION,
                details=tuple(errors),
            )
        return None

    def _training_evidence_errors(
        self,
        metrics_payload: dict[str, object],
        current_modality_counts: dict[str, int],
    ) -> list[str]:
        errors = []
        # Basic check to ensure metrics indicate learning occurred
        train_loss = metrics_payload.get("train_loss")
        val_loss = metrics_payload.get("val_loss")
        if isinstance(train_loss, (int, float)) and train_loss > 10.0:
            errors.append("training loss abnormally high")
        if isinstance(val_loss, (int, float)) and val_loss > 10.0:
            errors.append("validation loss abnormally high")
        return errors


def _read_json_object(*, path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training metrics must contain a JSON object")
    return payload


def _validate_manifest_schema(
    *,
    manifest: TrainingManifest,
    current_dataset_fingerprint: str,
    current_training_config_fingerprint: str,
    current_model_config_fingerprint: str,
    current_checkpoint_path: Path | None,
    current_metrics_path: Path | None,
) -> ValidationResult | None:
    changed_fields: list[str] = []
    if manifest.dataset_fingerprint != current_dataset_fingerprint:
        changed_fields.append("dataset fingerprint")
    if (
        manifest.training_config_fingerprint
        != current_training_config_fingerprint
    ):
        changed_fields.append("training config fingerprint")
    if manifest.model_config_fingerprint != current_model_config_fingerprint:
        changed_fields.append("model config fingerprint")
    if changed_fields:
        return ValidationResult.invalid(
            reason=WorkflowDecisionReason.TRAINING_INPUT_CHANGED,
            details=tuple(f"{field} mismatch" for field in changed_fields),
        )
    if (
        current_checkpoint_path is not None
        and manifest.checkpoint_path != current_checkpoint_path
    ) or (
        current_metrics_path is not None
        and manifest.metrics_path != current_metrics_path
    ):
        return ValidationResult.invalid(
            reason=WorkflowDecisionReason.TRAINING_OUTPUT_INVALID,
            details=("training artifact path mismatch",),
        )
    # Validate checkpoint integrity using checksum-backed availability
    if not checkpoint_is_available(manifest.checkpoint_path):
        return ValidationResult.invalid(
            reason=WorkflowDecisionReason.TRAINING_OUTPUT_FOR_SELECTED_DATASET_MISSING,
            details=("checkpoint not available or checksum invalid",),
        )
    # Verify SHA-256 matches manifest
    if manifest.sha256:
        try:
            actual_sha256 = checkpoint_sha256(manifest.checkpoint_path)
            if actual_sha256 != manifest.sha256.strip().lower():
                return ValidationResult.invalid(
                    reason=WorkflowDecisionReason.TRAINING_OUTPUT_INVALID,
                    details=("checkpoint SHA-256 mismatch with manifest",),
                )
        except (FileNotFoundError, ValueError) as exc:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.TRAINING_OUTPUT_INVALID,
                details=(f"checkpoint verification failed: {exc}",),
            )
    absent = ArtifactPathPresence.missing(
        manifest.input_dataset_root,
        manifest.checkpoint_path,
        manifest.metrics_path,
    )
    if not absent:
        return None
    return ValidationResult.invalid(
        reason=WorkflowDecisionReason.TRAINING_OUTPUT_FOR_SELECTED_DATASET_MISSING,
        details=(
            "training manifest references missing physical artifacts",
            *absent,
        ),
    )


def _metrics_schema_errors(
    *,
    manifest: TrainingManifest,
    metrics_payload: dict[str, object],
) -> tuple[str, ...]:
    errors: list[str] = []
    for name, expected in manifest.identity_fields().items():
        if metrics_payload.get(name) != expected:
            errors.append(f"metrics {name} mismatch")
    expected_fields = {
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "training_config_fingerprint": manifest.training_config_fingerprint,
        "model_config_fingerprint": manifest.model_config_fingerprint,
        "release_stage": manifest.release_stage,
    }
    for name, expected in expected_fields.items():
        if metrics_payload.get(name) != expected:
            errors.append(f"metrics {name} mismatch")
    for name in ("train_loss", "val_loss", "test_loss"):
        value = metrics_payload.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            errors.append(f"metrics {name} missing or non-finite")
    evaluation = metrics_payload.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("valid") is not True:
        errors.append("metrics final evaluation is invalid")
    return tuple(errors)


def _validate_rejection_report(
    manifest: TrainingManifest,
    metrics_payload: dict[str, object],
) -> ValidationResult | None:
    """Inlined from TrainingRejectionReportValidator for full consolidation.

    Checks if the rejection report in metrics indicates problems with the training output.
    """
    if not metrics_payload:
        return None

    rejections = metrics_payload.get("rejections_by_reason") or {}
    if isinstance(rejections, dict) and any(
        int(v) > 0 for v in rejections.values() if str(v).strip().isdigit()
    ):
        return ValidationResult.invalid(
            reason=WorkflowDecisionReason.TRAINING_OUTPUT_INVALID,
            details=(
                "training rejection report indicates rejected samples",
                f"rejections={rejections}",
            ),
        )
    return None
