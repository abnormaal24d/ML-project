"""Persist completed training, evaluation, and acceptance artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from datachecker.manifests.artifact_manifest import format_manifest_path
from datachecker.manifests.manifest_file_writer import ManifestWriterBase, Now
from datachecker.manifests.training_manifest import TrainingManifest
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus
from schemas.release import ReleaseStatus


class AcceptanceDecision(Protocol):
    """Decision surface required by manifest persistence."""

    status: ReleaseStatus

    def to_payload(self) -> dict[str, object]:
        """Return the stable persisted decision payload."""

        ...


class TrainingAcceptanceResult(Protocol):
    """Acceptance surface required by manifest persistence."""

    decision: AcceptanceDecision
    acceptance_report_path: Path
    evidence_bundle_path: Path | None

    def evidence_paths(self, *, checkpoint_path: Path) -> dict[str, Path]:
        """Return persisted evidence paths."""

        ...


if TYPE_CHECKING:
    from config.path_resolution.workflow_artifact_paths import (
        ArtifactPathRegistry,
    )
    from datachecker.fingerprints import SettingsFingerprintCalculator
    from datachecker.manifests.artifact_manifest import RunArtifactIdentity
    from datachecker.manifests.manifest_file_writer import ManifestFileWriter
    from evaluator.results import EvaluationResult
    from logger.project_logger import ProjectLogger
    from training.runtime.results import TrainingRunResult


class TrainingArtifactManifestWriter(ManifestWriterBase):
    """Passively persist already evaluated training artifacts."""

    def __init__(
        self,
        *,
        artifact_path_registry: ArtifactPathRegistry,
        settings_fingerprint_calculator: SettingsFingerprintCalculator,
        training_settings_payload: dict[str, object],
        model_settings_payload: dict[str, object],
        logger: ProjectLogger,
        project_root: Path | None,
        file_writer: ManifestFileWriter,
        artifact_identity: RunArtifactIdentity,
        release_stage: str,
        now: Now,
    ) -> None:
        super().__init__(
            artifact_path_registry=artifact_path_registry,
            logger=logger,
            project_root=project_root,
            file_writer=file_writer,
            artifact_identity=artifact_identity,
            now=now,
        )
        self._settings_fingerprint_calculator = settings_fingerprint_calculator
        self._training_settings_payload = training_settings_payload
        self._model_settings_payload = model_settings_payload
        self._release_stage = release_stage

    def write_training_metrics(
        self,
        *,
        input_dataset_root: Path,
        dataset_manifest_hash: str,
        training_result: TrainingRunResult,
        evaluation_result: EvaluationResult,
    ) -> Path:
        """Persist immutable metrics before the acceptance gate reads them."""

        normalized_dataset_hash = dataset_manifest_hash.strip()
        if not normalized_dataset_hash:
            raise ValueError(
                "Selected dataset manifest hash must not be empty."
            )
        if not input_dataset_root.is_dir():
            raise FileNotFoundError(
                "Training dataset root does not exist or is not a directory: "
                f"{input_dataset_root}"
            )

        metrics = training_result.metrics
        artifacts = training_result.artifacts
        identity = training_result.identity
        checkpoint_path, metrics_path = self._training_metrics_path(
            training_result=training_result,
        )
        completed_at = self._utc_now_iso()
        training_config_fingerprint = (
            self._settings_fingerprint_calculator.calculate(
                payload=self._training_settings_payload,
            )
        )
        model_config_fingerprint = (
            self._settings_fingerprint_calculator.calculate(
                payload=self._model_settings_payload,
            )
        )
        export_directory = (
            self._resolve_manifest_path(
                artifacts.export_directory,
                field_name="export_directory",
            )
            if artifacts.export_directory is not None
            else None
        )
        run_mode = str(
            self._training_settings_payload.get("run_mode") or ""
        ).strip()
        evaluation_payload = evaluation_result.to_payload()

        metrics_payload: dict[str, object] = {
            **self._identity_fields(),
            **metrics.to_payload(),
            "checkpoint_path": checkpoint_path,
            "last_checkpoint_path": artifacts.last_checkpoint_path,
            "dataset_fingerprint": normalized_dataset_hash,
            "training_config_fingerprint": training_config_fingerprint,
            "model_config_fingerprint": model_config_fingerprint,
            **identity.to_payload(),
            "export_paths": dict(artifacts.export_paths),
            "export_directory": (
                format_manifest_path(export_directory)
                if export_directory is not None
                else None
            ),
            "run_mode": run_mode,
            "release_stage": self._release_stage,
            "dataset_root": format_manifest_path(input_dataset_root),
            "evaluation": evaluation_payload,
            "completed_at": completed_at,
        }
        self._write_manifest(path=metrics_path, payload=metrics_payload)

        self._logger.info(
            "workflow_training_metrics_written",
            metrics_path=format_manifest_path(metrics_path),
            checkpoint_path=format_manifest_path(checkpoint_path),
            dataset_root=format_manifest_path(input_dataset_root),
            train_loss=metrics.train_loss,
            val_loss=evaluation_result.validation_loss,
            test_loss=evaluation_result.test_loss,
            model_seed=identity.model_seed,
        )
        return metrics_path

    def write_training_manifests(
        self,
        *,
        input_dataset_root: Path,
        dataset_manifest_hash: str,
        training_result: TrainingRunResult,
        evaluation_result: EvaluationResult,
        acceptance_result: TrainingAcceptanceResult,
    ) -> TrainingManifest:
        """Persist post-acceptance manifests without mutating release evidence."""

        normalized_dataset_hash = dataset_manifest_hash.strip()
        if not normalized_dataset_hash:
            raise ValueError(
                "Selected dataset manifest hash must not be empty."
            )
        if not input_dataset_root.is_dir():
            raise FileNotFoundError(
                "Training dataset root does not exist or is not a directory: "
                f"{input_dataset_root}"
            )

        decision = acceptance_result.decision
        if decision.status is ReleaseStatus.FAILED:
            raise RuntimeError("Cannot persist a rejected training release.")
        if not acceptance_result.acceptance_report_path.is_file():
            raise FileNotFoundError(
                "Acceptance report was not persisted before manifest writing."
            )

        metrics = training_result.metrics
        identity = training_result.identity
        checkpoint_path, metrics_path = self._training_metrics_path(
            training_result=training_result,
        )
        if not metrics_path.is_file():
            raise FileNotFoundError(
                "Training metrics must be persisted before release acceptance."
            )

        completed_at = self._utc_now_iso()
        training_config_fingerprint = (
            self._settings_fingerprint_calculator.calculate(
                payload=self._training_settings_payload,
            )
        )
        model_config_fingerprint = (
            self._settings_fingerprint_calculator.calculate(
                payload=self._model_settings_payload,
            )
        )

        manifest = TrainingManifest(
            **self._identity_fields(),
            dataset_fingerprint=normalized_dataset_hash,
            training_config_fingerprint=training_config_fingerprint,
            model_config_fingerprint=model_config_fingerprint,
            input_dataset_root=input_dataset_root,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            epoch_count=metrics.epochs,
            sample_count=metrics.samples,
            training_completed_at=completed_at,
            release_stage=self._release_stage,
            acceptance_status=decision.status.value,
            lifecycle_stage="training-ready",
            status=WorkflowLifecycleStatus.COMPLETED,
            final=True,
            sha256=self._compute_checkpoint_sha256(checkpoint_path),
        )
        manifest_path = self._artifact_path_registry.training_manifest_path()
        self._write_manifest(path=manifest_path, payload=manifest.to_payload())

        self._logger.info(
            "workflow_training_manifest_written",
            manifest_path=format_manifest_path(manifest_path),
            checkpoint_path=format_manifest_path(checkpoint_path),
            metrics_path=format_manifest_path(metrics_path),
            dataset_root=format_manifest_path(input_dataset_root),
            dataset_fingerprint=normalized_dataset_hash,
            train_loss=metrics.train_loss,
            val_loss=evaluation_result.validation_loss,
            test_loss=evaluation_result.test_loss,
            model_seed=identity.model_seed,
            split_seed=identity.split_seed,
            split_assignment=identity.split_assignment,
            evaluation_valid=evaluation_result.valid,
            acceptance_status=decision.status.value,
        )
        return manifest

    def _training_metrics_path(
        self,
        *,
        training_result: TrainingRunResult,
    ) -> tuple[Path, Path]:
        checkpoint_path = self._resolve_manifest_path(
            training_result.artifacts.checkpoint_path,
            field_name="checkpoint_path",
        )
        metrics_path = (
            checkpoint_path.parent
            / self._artifact_path_registry.dataset_paths.training_metrics_filename
        )
        return checkpoint_path, metrics_path

    def _compute_checkpoint_sha256(self, checkpoint_path: Path) -> str:
        from training.runtime.checkpoint.io import checkpoint_sha256

        return checkpoint_sha256(checkpoint_path)

    def _resolve_manifest_path(
        self,
        value: str | Path,
        *,
        field_name: str,
    ) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        if self._project_root is None:
            raise RuntimeError(
                f"Cannot resolve relative {field_name} without project_root: "
                f"{path}"
            )
        return self._project_root / path
