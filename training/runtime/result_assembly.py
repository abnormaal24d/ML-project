"""Assemble immutable training results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from training.runtime.results import (
    TrainingArtifacts,
    TrainingMetrics,
    TrainingRunIdentity,
    TrainingRunResult,
)

if TYPE_CHECKING:
    from mmcrawler_datasets.validation.training_preflight import (
        EffectiveTrainingSplitReport,
    )
    from training.runtime.loop.state import TrainingLoopState


def assemble_training_run_result(
    *,
    train_loss: float,
    validation_loss: float,
    test_loss: float,
    loop_state: TrainingLoopState,
    epoch_history: tuple[dict[str, object], ...],
    sample_count: int,
    readiness: EffectiveTrainingSplitReport,
    training_signal_by_modality: dict[str, dict[str, object]],
    saved_checkpoint_path: Path,
    last_checkpoint_path: Path,
    export_directory: Path,
    export_paths: dict[str, str],
    model_seed: int,
) -> TrainingRunResult:
    """Build the typed result from training and selected-model metrics."""

    if loop_state.total_batches <= 0:
        raise RuntimeError(
            "training result cannot be assembled without completed batches"
        )

    if not loop_state.epoch_losses:
        raise RuntimeError(
            "training result cannot be assembled without epoch losses"
        )

    average_loss = loop_state.cumulative_loss_sum / loop_state.total_batches

    last_epoch_loss = loop_state.epoch_losses[-1]

    return TrainingRunResult(
        metrics=TrainingMetrics(
            train_loss=train_loss,
            validation_loss=validation_loss,
            test_loss=test_loss,
            average_loss=average_loss,
            last_epoch_loss=last_epoch_loss,
            epochs=loop_state.completed_epochs,
            batches=loop_state.total_batches,
            samples=sample_count,
            effective_train_sample_count=sample_count,
            effective_task_counts=dict(readiness.task_counts),
            effective_modality_counts=dict(readiness.modality_counts),
            training_signal_by_modality={
                modality: dict(signal)
                for modality, signal in training_signal_by_modality.items()
            },
            epoch_history=tuple(dict(epoch) for epoch in epoch_history),
        ),
        artifacts=TrainingArtifacts(
            checkpoint_path=saved_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            export_directory=export_directory,
            export_paths=dict(export_paths),
        ),
        identity=TrainingRunIdentity(
            model_seed=model_seed,
        ),
    )


__all__ = ["assemble_training_run_result"]
