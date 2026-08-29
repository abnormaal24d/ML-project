"""Training attempt execution (TRAINING -> RECEIPT -> EVALUATION)."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from logger.project_logger import ProjectLogger
from orchestration.workflow.training.protocols import (
    CheckpointLoader,
    TrainingEvaluator,
    TrainingReceiptWriter,
    TrainingRuntimeFactory,
)
from orchestration.workflow.training.stage_executor import (
    StageResult,
    TrainingStageExecutor,
)
from training.runtime.job_status.models import (
    TrainingJobIdentity,
    TrainingOperationStage,
)
from training.runtime.job_status.store import TrainingJobStatusStore
from training.runtime.results import TrainingRunResult

if TYPE_CHECKING:
    from config.multimodal.training_settings import TrainingSettings
    from evaluator.reproducibility import TrainingRunReceipt
    from evaluator.results import EvaluationResult
    from training.runtime.trainer import MultimodalTrainer


class TrainingArtifactPersistenceError(RuntimeError):
    """Training succeeded, but required artifacts were not persisted."""


class TrainingEvaluationError(RuntimeError):
    """Training succeeded, but checkpoint evaluation failed."""


class TrainingStatusPersistenceError(RuntimeError):
    """The training lifecycle status could not be persisted."""


@dataclass(frozen=True, slots=True)
class _TrainingStageOutput:
    """Internal output of the TRAINING stage."""

    trainer: "MultimodalTrainer"
    effective_training: "TrainingSettings"
    execution: TrainingRunResult


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    """Consumable result of one immutable seed run inside a campaign."""

    identity: TrainingJobIdentity
    execution: TrainingRunResult
    receipt: "TrainingRunReceipt | None" = None
    evaluation: "EvaluationResult | None" = None


class TrainingAttemptRunner:
    """Execute a single training attempt: TRAINING -> RECEIPT -> EVALUATION.

    All dependencies are injected as exact values and bound capabilities;
    this object never sees the root settings tree.
    """

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        run_blocking: Callable[..., Awaitable[Any]],
        status_store: TrainingJobStatusStore,
        stage_executor: TrainingStageExecutor,
        runtime_factory: TrainingRuntimeFactory,
        checkpoint_filename: str,
        checkpoint_directory: str,
        model_settings_payload: Mapping[str, object],
        container_digest: str,
        checkpoint_loader: CheckpointLoader,
        receipt_writer: TrainingReceiptWriter,
        evaluator: TrainingEvaluator,
    ) -> None:
        self._logger = logger
        self._run_blocking = run_blocking
        self._status_store = status_store
        self._stage_executor = stage_executor
        self._runtime_factory = runtime_factory
        self._checkpoint_filename = checkpoint_filename
        self._checkpoint_directory = checkpoint_directory
        self._model_settings_payload = model_settings_payload
        self._container_digest = container_digest
        self._checkpoint_loader = checkpoint_loader
        self._receipt_writer = receipt_writer
        self._evaluator = evaluator

    async def run(
        self,
        *,
        dataset_manifest_hash: str,
        training_root: Path,
        candidate_root: Path | None,
        seed: int | None,
        is_primary: bool,
        cancel_event: threading.Event,
        identity: TrainingJobIdentity,
        deterministic: bool | None = None,
        snapshot_id: str,
    ) -> AttemptOutcome:
        """Execute one training attempt."""
        candidate_directory = (
            candidate_root / f"seed-{seed if seed is not None else 'primary'}"
            if candidate_root is not None
            else None
        )

        training_stage = await self._run_training_stage(
            dataset_manifest_hash=dataset_manifest_hash,
            training_root=training_root,
            candidate_directory=candidate_directory,
            seed=seed,
            is_primary=is_primary,
            cancel_event=cancel_event,
            identity=identity,
            deterministic=deterministic,
            snapshot_id=snapshot_id,
        )
        if training_stage.error is not None:
            raise training_stage.error
        training = training_stage.result
        if not isinstance(training, _TrainingStageOutput):
            raise TypeError(
                f"Expected _TrainingStageOutput, got {type(training).__name__}"
            )

        receipt_stage = await self._run_receipt_stage(
            execution=training.execution,
            identity=identity,
            dataset_manifest_hash=dataset_manifest_hash,
            training_root=training_root,
            effective_training=training.effective_training,
        )
        if receipt_stage.error is not None:
            raise receipt_stage.error

        evaluation = None
        if is_primary:
            evaluation_stage = await self._run_evaluation_stage(
                trainer=training.trainer,
                execution=training.execution,
                training_root=training_root,
                identity=identity,
                dataset_manifest_hash=dataset_manifest_hash,
            )
            if evaluation_stage.error is not None:
                raise evaluation_stage.error
            evaluation = evaluation_stage.result

        required_stages = (
            (
                TrainingOperationStage.TRAINING,
                TrainingOperationStage.RECEIPT,
                TrainingOperationStage.EVALUATION,
            )
            if is_primary
            else (
                TrainingOperationStage.TRAINING,
                TrainingOperationStage.RECEIPT,
            )
        )
        self._status_store.write_attempt_completed(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
            result=training.execution,
            required_stages=required_stages,
        )

        return AttemptOutcome(
            identity=identity,
            execution=training.execution,
            receipt=receipt_stage.result,
            evaluation=evaluation,
        )

    async def _run_training_stage(
        self,
        *,
        dataset_manifest_hash: str,
        training_root: Path,
        candidate_directory: Path | None,
        seed: int | None,
        is_primary: bool,
        cancel_event: threading.Event,
        identity: TrainingJobIdentity,
        deterministic: bool | None,
        snapshot_id: str,
    ) -> StageResult[_TrainingStageOutput]:
        async def execute_training() -> _TrainingStageOutput:
            trainer, effective_training = self._runtime_factory.create(
                training_root=training_root,
                seed=seed,
                deterministic=deterministic,
            )

            checkpoint_path = (
                candidate_directory / self._checkpoint_filename
                if candidate_directory is not None
                else self._build_checkpoint_path(
                    checkpoint_directory=self._checkpoint_directory,
                    checkpoint_filename=self._checkpoint_filename,
                    snapshot_id=snapshot_id,
                    is_primary=is_primary,
                    seed=seed,
                )
            )

            export_directory = (
                candidate_directory
                if candidate_directory is not None
                else (
                    training_root / "export"
                    if is_primary
                    else training_root / f"seed-{seed}" / "export"
                )
            )

            execution = await self._run_blocking(
                trainer.train,
                dataset_root=training_root,
                checkpoint_path=checkpoint_path,
                export_directory=export_directory,
                dataset_manifest_sha256=dataset_manifest_hash,
                cancel_event=cancel_event,
                run_id=identity.attempt_id,
                cancel=lambda: cancel_event.set(),
            )

            if not isinstance(execution, TrainingRunResult):
                raise TypeError(
                    f"trainer returned unsupported result type: {type(execution).__name__}"
                )

            return _TrainingStageOutput(
                trainer=trainer,
                effective_training=effective_training,
                execution=execution,
            )

        self._status_store.write_started(
            identity=identity,
            training_root=training_root,
            dataset_manifest_hash=dataset_manifest_hash,
        )

        return await self._stage_executor.execute(
            identity=identity,
            training_root=str(training_root),
            dataset_manifest_hash=dataset_manifest_hash,
            stage=TrainingOperationStage.TRAINING,
            operation=execute_training,
            status_result=lambda output: output.execution,
        )

    async def _run_receipt_stage(
        self,
        *,
        execution: TrainingRunResult,
        identity: TrainingJobIdentity,
        dataset_manifest_hash: str,
        training_root: Path,
        effective_training: "TrainingSettings",
    ) -> StageResult["TrainingRunReceipt"]:
        async def persist_receipt() -> "TrainingRunReceipt":
            checkpoint_payload = await self._checkpoint_loader(
                execution.artifacts.checkpoint_path,
            )
            if not isinstance(checkpoint_payload, Mapping):
                raise TypeError("checkpoint payload must be a mapping")

            return await self._receipt_writer(
                output_path=(
                    execution.artifacts.evaluation_directory
                    / "run_receipt.json"
                ),
                run_id=identity.attempt_id,
                seed=execution.identity.model_seed,
                dataset_manifest_sha256=dataset_manifest_hash,
                checkpoint_payload=checkpoint_payload,
                training_settings=effective_training.model_dump(mode="json"),
                model_settings=self._model_settings_payload,
                container_digest=self._container_digest,
                evaluated_metrics={
                    "train_loss": execution.metrics.train_loss,
                    "validation_loss": execution.metrics.validation_loss,
                    "test_loss": execution.metrics.test_loss,
                },
            )

        receipt_result = await self._stage_executor.execute(
            identity=identity,
            training_root=str(training_root),
            dataset_manifest_hash=dataset_manifest_hash,
            stage=TrainingOperationStage.RECEIPT,
            operation=persist_receipt,
        )
        if receipt_result.error:
            raise TrainingArtifactPersistenceError(
                "Training succeeded, but run receipt persistence failed"
            ) from receipt_result.error
        return receipt_result

    async def _run_evaluation_stage(
        self,
        *,
        trainer: "MultimodalTrainer",
        execution: TrainingRunResult,
        training_root: Path,
        identity: TrainingJobIdentity,
        dataset_manifest_hash: str,
    ) -> StageResult["EvaluationResult"]:
        evaluation_result = await self._stage_executor.execute(
            identity=identity,
            training_root=str(training_root),
            dataset_manifest_hash=dataset_manifest_hash,
            stage=TrainingOperationStage.EVALUATION,
            operation=lambda: self._evaluator(
                trainer=trainer,
                training_result=execution,
                dataset_root=training_root,
                leakage_report_path=(
                    training_root / "evaluation" / "leakage_report.json"
                ),
                reproducibility_report_path=None,
            ),
        )
        if evaluation_result.error:
            raise TrainingEvaluationError(
                "Training succeeded, but checkpoint evaluation failed"
            ) from evaluation_result.error
        return evaluation_result

    def _build_checkpoint_path(
        self,
        *,
        checkpoint_directory: str,
        checkpoint_filename: str,
        snapshot_id: str,
        is_primary: bool,
        seed: int | None,
    ) -> Path:
        base = Path(checkpoint_directory) / snapshot_id
        if not is_primary and seed is not None:
            base = base / f"seed-{seed}"
        return base / checkpoint_filename
