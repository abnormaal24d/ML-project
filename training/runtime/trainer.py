"""Multimodal training orchestration."""

from __future__ import annotations

import threading
from collections import Counter
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from evaluator.loss import evaluate_final_losses
from evaluator.metric_registry import SUPPORTED_EVALUATION_METHODS
from evaluator.results import EvaluationResult
from evaluator.task_metrics import (
    evaluate_task_metrics,
    evaluate_task_metrics_with_runtime,
    summarize_task_metrics,
)
from mmcrawler_datasets.schema import DatasetSplit
from mmcrawler_datasets.validation.training_preflight import (
    validate_effective_training_split,
)
from multimodal.tasks.registry import get_task
from schemas.multimodal_tasks import canonical_task_name
from training.runtime.checkpoint.contract import CheckpointContract

from .artifact_persistence import (
    best_checkpoint_path,
    persist_selected_artifacts,
    save_epoch_checkpoint,
)
from .checkpoint.io import checkpoint_is_available
from .checkpoint.metadata import resolve_dataset_fingerprint
from .checkpoint.service import load_model_weights
from .checkpoint.state import reproducibility_runtime
from .device import (
    destroy_distributed,
    maybe_init_distributed,
    resolve_device,
)
from .loop.runner import run_training_loop
from .loop.state import TrainingLoopState
from .offline import offline_training_guard
from .precision import autocast_context
from .preparation import (
    PreparedTrainingRuntime,
    open_training_split,
    prepare_training_runtime,
)
from .result_assembly import assemble_training_run_result
from .results import TrainingRunResult
from .signal import validate_effective_training_signal

if TYPE_CHECKING:
    from collections.abc import Callable

    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_settings import TrainingSettings
    from logger.project_logger import ProjectLogger
    from mmcrawler_datasets.dataset import MultimodalJsonlDataset
    from mmcrawler_datasets.validation.training_preflight import (
        EffectiveTrainingSplitReport,
    )
    from multimodal.model.model import MultimodalModel
    from multimodal.tokenization.text import VocabularyTokenizer
    from training.losses.objective import SupervisedOrSelfSupervisedLoss


class MultimodalTrainer:
    """Train the multimodal model from one canonical dataset root."""

    def __init__(
        self,
        *,
        model_settings: ModelSettings,
        training_settings: TrainingSettings,
        tokenizer: VocabularyTokenizer,
        model_exporter: Callable[..., dict[str, str]],
        logger: ProjectLogger,
        training_backend: str,
        model_factory: Callable[[ModelSettings], MultimodalModel],
        loss_factory: Callable[
            [TrainingSettings],
            SupervisedOrSelfSupervisedLoss,
        ],
        optimizer_factory: Callable[
            [torch.nn.Module, TrainingSettings],
            torch.optim.Optimizer,
        ],
        scheduler_factory: Callable[..., object | None],
        project_root: Path,
        checkpoint_contract: CheckpointContract | None = None,
    ) -> None:
        if (
            training_settings.training_stage == "PREFERENCE_TUNING"
            and training_backend != "dense_transformer"
        ):
            raise ValueError(
                "PREFERENCE_TUNING requires training_backend="
                "'dense_transformer' for chosen/rejected sequence scoring"
            )
        self._model_settings = model_settings
        self._training_settings = training_settings
        self._tokenizer = tokenizer
        self._model_exporter = model_exporter
        self._logger = logger
        self._training_backend = training_backend
        self._model_factory = model_factory
        self._loss_factory = loss_factory
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory
        self._project_root = project_root.resolve()
        self._checkpoint_contract = checkpoint_contract
        self._device = resolve_device(training_settings.device)

    def train(
        self,
        *,
        dataset_root: Path,
        checkpoint_path: Path,
        export_directory: Path,
        dataset_manifest_sha256: str,
        cancel_event: threading.Event | None = None,
        run_id: str | None = None,
    ) -> TrainingRunResult:
        """Train under the same fail-closed network rules as production."""

        with offline_training_guard(settings=self._training_settings):
            return self._train_distributed(
                dataset_root=dataset_root,
                checkpoint_path=checkpoint_path,
                export_directory=export_directory,
                dataset_manifest_sha256=dataset_manifest_sha256,
                cancel_event=cancel_event,
                run_id=run_id,
            )

    def _train_distributed(
        self,
        *,
        dataset_root: Path,
        checkpoint_path: Path,
        export_directory: Path,
        dataset_manifest_sha256: str,
        cancel_event: threading.Event | None,
        run_id: str | None,
    ) -> TrainingRunResult:
        """Initialize and reliably destroy the distributed runtime."""

        distributed_context = maybe_init_distributed(
            settings=self._training_settings,
            device=self._device,
        )
        try:
            return self._run_training(
                dataset_root=dataset_root,
                checkpoint_path=checkpoint_path,
                export_directory=export_directory,
                distributed_context=distributed_context,
                dataset_manifest_sha256=dataset_manifest_sha256,
                cancel_event=cancel_event,
                run_id=run_id,
            )
        finally:
            destroy_distributed(distributed_context)

    def _run_training(
        self,
        *,
        dataset_root: Path,
        checkpoint_path: Path,
        export_directory: Path,
        distributed_context: dict[str, object],
        dataset_manifest_sha256: str,
        cancel_event: threading.Event | None = None,
        run_id: str | None = None,
    ) -> TrainingRunResult:
        """Coordinate preparation, loop execution, persistence, and results."""

        resolved_dataset_manifest_sha256 = resolve_dataset_fingerprint(
            dataset_manifest_sha256=dataset_manifest_sha256,
        )

        with reproducibility_runtime(settings=self._training_settings):
            train_dataset, train_loader = self._open_split(
                dataset_root=dataset_root,
                split=DatasetSplit.TRAIN,
            )
            readiness = validate_effective_training_split(
                dataset=train_dataset,
                training_settings=self._training_settings,
            )
            num_training_batches = _required_batch_count(train_loader)
            self._logger.info(
                "multimodal_training_loop_started",
                dataset_root=dataset_root.as_posix(),
                checkpoint_path=checkpoint_path.as_posix(),
                training_backend=self._training_backend,
                epochs=self._training_settings.epochs,
                batch_size=self._training_settings.batch_size,
                sample_count=readiness.sample_count,
                task_counts=readiness.task_counts,
                modality_counts=readiness.modality_counts,
                device=str(self._device),
            )

            prepared = prepare_training_runtime(
                model_settings=self._model_settings,
                training_settings=self._training_settings,
                model_factory=self._model_factory,
                loss_factory=self._loss_factory,
                optimizer_factory=self._optimizer_factory,
                scheduler_factory=self._scheduler_factory,
                logger=self._logger,
                device=self._device,
                dataset_root=dataset_root,
                dataset_manifest_sha256=resolved_dataset_manifest_sha256,
                sample_count=readiness.sample_count,
                modalities=tuple(readiness.modality_counts),
                distributed_context=distributed_context,
                num_training_batches=num_training_batches,
            )

            validation_dataset, validation_loader = self._open_split(
                dataset_root=dataset_root,
                split=DatasetSplit.VAL,
            )
            test_dataset, test_loader = self._open_split(
                dataset_root=dataset_root,
                split=DatasetSplit.TEST,
            )

            loop_state, epoch_history = self._execute_training_loop(
                dataset_root=dataset_root,
                checkpoint_path=checkpoint_path,
                readiness=readiness,
                prepared=prepared,
                train_loader=train_loader,
                validation_loader=validation_loader,
                test_loader=test_loader,
                cancel_event=cancel_event,
                run_id=run_id,
            )

            training_signal_by_modality = prepared.signal_tracker.to_payload()
            validate_effective_training_signal(
                modality_counts=readiness.modality_counts,
                training_signal_by_modality=training_signal_by_modality,
            )

            last_checkpoint_path = save_epoch_checkpoint(
                path=checkpoint_path,
                state=loop_state,
                test_loss=loop_state.test_loss,
                prepared=prepared,
                model_settings=self._model_settings,
                training_settings=self._training_settings,
                dataset_root=dataset_root,
                device=self._device,
                sample_count=readiness.sample_count,
                readiness=readiness,
                logger=self._logger,
                run_id=run_id,
                checkpoint_contract=self._checkpoint_contract,
            )
        selected_checkpoint_path = best_checkpoint_path(checkpoint_path)
        load_model_weights(
            model=prepared.model,
            checkpoint_path=selected_checkpoint_path,
            model_settings=self._model_settings,
        )

        selected_train_loss, selected_validation_loss, selected_test_loss = (
            evaluate_final_losses(
                model=prepared.model,
                loss_fn=prepared.loss_fn,
                train_loader=train_loader,
                val_loader=validation_loader,
                test_loader=test_loader,
                device=self._device,
                autocast_factory=lambda: autocast_context(
                    prepared.precision_runtime
                ),
            )
        )

        saved_checkpoint_path, export_paths = persist_selected_artifacts(
            model=prepared.model,
            best_checkpoint_path=selected_checkpoint_path,
            export_directory=export_directory,
            dataset_root=dataset_root,
            prepared=prepared,
            model_exporter=self._model_exporter,
            model_settings=self._model_settings,
            training_settings=self._training_settings,
        )
        self._logger.info(
            "multimodal_training_loop_completed",
            checkpoint_path=saved_checkpoint_path.as_posix(),
            export_paths=export_paths,
            last_epoch_loss=(
                loop_state.epoch_losses[-1]
                if loop_state.epoch_losses
                else None
            ),
            train_loss=selected_train_loss,
            val_loss=selected_validation_loss,
            test_loss=selected_test_loss,
            batch_count=loop_state.total_batches,
            sample_count=readiness.sample_count,
            model_seed=self._training_settings.seed,
        )

        return assemble_training_run_result(
            train_loss=selected_train_loss,
            validation_loss=selected_validation_loss,
            test_loss=selected_test_loss,
            loop_state=loop_state,
            epoch_history=epoch_history,
            sample_count=readiness.sample_count,
            readiness=readiness,
            training_signal_by_modality=training_signal_by_modality,
            saved_checkpoint_path=saved_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            export_directory=export_directory,
            export_paths=export_paths,
            model_seed=self._training_settings.seed,
        )

    def _execute_training_loop(
        self,
        *,
        dataset_root: Path,
        checkpoint_path: Path,
        readiness: EffectiveTrainingSplitReport,
        prepared: PreparedTrainingRuntime,
        train_loader: torch.utils.data.DataLoader[Any],
        validation_loader: torch.utils.data.DataLoader[Any],
        test_loader: torch.utils.data.DataLoader[Any],
        cancel_event: threading.Event | None = None,
        run_id: str | None = None,
    ) -> tuple[TrainingLoopState, tuple[dict[str, object], ...]]:
        """Run epochs with explicit last and best checkpoint callbacks."""

        prepared.model.train()
        loop_state = TrainingLoopState.from_resume_state(prepared.resume_state)
        selected_checkpoint_path = best_checkpoint_path(checkpoint_path)

        def save_last_epoch(state: TrainingLoopState) -> None:
            save_epoch_checkpoint(
                path=checkpoint_path,
                state=state,
                test_loss=None,
                prepared=prepared,
                model_settings=self._model_settings,
                training_settings=self._training_settings,
                dataset_root=dataset_root,
                device=self._device,
                sample_count=readiness.sample_count,
                readiness=readiness,
                logger=self._logger,
                run_id=run_id,
                checkpoint_contract=self._checkpoint_contract,
            )

        def save_best_epoch(state: TrainingLoopState) -> None:
            save_epoch_checkpoint(
                path=selected_checkpoint_path,
                state=state,
                test_loss=None,
                prepared=prepared,
                model_settings=self._model_settings,
                training_settings=self._training_settings,
                dataset_root=dataset_root,
                device=self._device,
                sample_count=readiness.sample_count,
                readiness=readiness,
                logger=self._logger,
                run_id=run_id,
                checkpoint_contract=self._checkpoint_contract,
            )

        loop_state, epoch_history = run_training_loop(
            settings=self._training_settings,
            device=self._device,
            logger=self._logger,
            model=prepared.model,
            loss_fn=prepared.loss_fn,
            optimizer=prepared.optimizer,
            scheduler=prepared.scheduler,
            train_loader=train_loader,
            val_loader=validation_loader,
            test_loader=test_loader,
            loop_state=loop_state,
            signal_tracker=prepared.signal_tracker,
            precision_runtime=prepared.precision_runtime,
            grad_scaler=prepared.grad_scaler,
            distributed_context=prepared.distributed_context,
            last_epoch_checkpoint=save_last_epoch,
            best_epoch_checkpoint=save_best_epoch,
            cancel_event=cancel_event,
        )
        if loop_state.last_val_loss is None or loop_state.test_loss is None:
            raise RuntimeError(
                "training cannot complete without validation and test loss"
            )
        if not checkpoint_is_available(selected_checkpoint_path):
            raise RuntimeError(
                "best checkpoint was not produced; refusing to export the "
                "last model as a substitute"
            )
        return loop_state, epoch_history

    def _open_split(
        self,
        *,
        dataset_root: Path,
        split: DatasetSplit,
        distributed: bool = True,
    ) -> tuple[MultimodalJsonlDataset, torch.utils.data.DataLoader[Any]]:
        return open_training_split(
            dataset_root=dataset_root,
            split=split,
            model_settings=self._model_settings,
            training_settings=self._training_settings,
            tokenizer=self._tokenizer,
            logger=self._logger,
            distributed=distributed,
        )

    def autocast_factory(self) -> Callable[[], AbstractContextManager[object]]:
        """Return an autocast context manager factory for evaluation."""
        from training.runtime.precision import (
            autocast_context,
            resolve_precision_runtime,
        )

        precision_runtime = resolve_precision_runtime(
            settings=self._training_settings,
            device=self._device,
        )
        return lambda: autocast_context(precision_runtime)


def evaluate_selected_checkpoint(
    *,
    trainer: MultimodalTrainer,
    training_result: TrainingRunResult,
    dataset_root: Path,
    leakage_report_path: Path | None = None,
    reproducibility_report_path: Path | None = None,
) -> EvaluationResult:
    """Evaluate task metrics for the selected trained checkpoint."""

    train_dataset, _ = trainer._open_split(
        dataset_root=dataset_root,
        split=DatasetSplit.TRAIN,
        distributed=False,
    )

    validation_dataset, validation_loader = trainer._open_split(
        dataset_root=dataset_root,
        split=DatasetSplit.VAL,
        distributed=False,
    )

    test_dataset, test_loader = trainer._open_split(
        dataset_root=dataset_root,
        split=DatasetSplit.TEST,
        distributed=False,
    )

    evaluation_model = trainer._model_factory(trainer._model_settings).to(
        trainer._device
    )

    load_model_weights(
        model=evaluation_model,
        model_settings=trainer._model_settings,
        checkpoint_path=training_result.artifacts.checkpoint_path,
    )

    autocast_factory = trainer.autocast_factory()

    validation_task_metrics = evaluate_task_metrics(
        model=evaluation_model,
        loader=validation_loader,
        device=trainer._device,
        autocast_factory=autocast_factory,
        tokenizer=trainer._tokenizer,
    )

    test_task_metrics, max_batch_latency_ms, peak_memory_mb = (
        evaluate_task_metrics_with_runtime(
            model=evaluation_model,
            loader=test_loader,
            device=trainer._device,
            autocast_factory=autocast_factory,
            tokenizer=trainer._tokenizer,
        )
    )

    validation_task_counts = Counter(
        canonical_task_name(ref.task_type) for ref in validation_dataset.refs
    )
    test_task_counts = Counter(
        canonical_task_name(ref.task_type) for ref in test_dataset.refs
    )
    failure_reasons = (
        *_evaluation_metric_failures(
            split="validation",
            task_counts=validation_task_counts,
            task_metrics=validation_task_metrics,
        ),
        *_evaluation_metric_failures(
            split="test",
            task_counts=test_task_counts,
            task_metrics=test_task_metrics,
        ),
    )
    labeled_sample_count = len(validation_dataset) + len(test_dataset)
    if labeled_sample_count == 0:
        failure_reasons = (*failure_reasons, "evaluation_splits_empty")

    return EvaluationResult(
        validation_loss=training_result.metrics.validation_loss,
        test_loss=training_result.metrics.test_loss,
        evaluation_mode="supervised_task_metrics",
        labeled_sample_count=labeled_sample_count,
        dataset_split_counts={
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        task_metrics=validation_task_metrics,
        test_task_metrics=test_task_metrics,
        metrics=summarize_task_metrics(validation_task_metrics),
        valid=not failure_reasons,
        failure_reasons=failure_reasons,
        max_batch_latency_ms=max_batch_latency_ms,
        peak_memory_mb=peak_memory_mb,
        leakage_report_path=leakage_report_path,
        reproducibility_report_path=reproducibility_report_path,
    )


def _evaluation_metric_failures(
    *,
    split: str,
    task_counts: Counter[str],
    task_metrics: dict[str, dict[str, float]],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for task_type, sample_count in sorted(task_counts.items()):
        if sample_count <= 0:
            continue
        definition = get_task(task_type)
        if definition is None:
            reasons.append(f"{split}:unknown_task:{task_type}")
            continue
        if definition.evaluation_method not in SUPPORTED_EVALUATION_METHODS:
            reasons.append(
                f"{split}:unsupported_evaluation_method:{task_type}:"
                f"{definition.evaluation_method}"
            )
            continue
        if not task_metrics.get(task_type):
            reasons.append(f"{split}:task_metrics_missing:{task_type}")
    return tuple(reasons)


def _required_batch_count(
    loader: torch.utils.data.DataLoader[Any],
) -> int:
    """Return a positive batch count required by step-based schedulers."""

    try:
        batch_count = int(len(loader))
    except TypeError as exc:
        raise TypeError(
            "training loader must define its number of batches for a "
            "step-based scheduler"
        ) from exc
    if batch_count <= 0:
        raise ValueError("training loader must contain at least one batch")
    return batch_count


def train_and_collect_results(
    *,
    trainer: MultimodalTrainer,
    training_root: Path,
    checkpoint_path: Path,
    export_directory: Path,
    dataset_manifest_sha256: str,
    cancel_event: threading.Event | None = None,
    run_id: str | None = None,
) -> TrainingRunResult:
    """Train once and return typed training results."""

    return trainer.train(
        dataset_root=training_root,
        checkpoint_path=checkpoint_path,
        export_directory=export_directory,
        dataset_manifest_sha256=dataset_manifest_sha256,
        cancel_event=cancel_event,
        run_id=run_id,
    )


__all__ = [
    "MultimodalTrainer",
    "evaluate_selected_checkpoint",
    "train_and_collect_results",
]
