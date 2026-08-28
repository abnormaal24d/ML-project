"""Checkpoint callbacks and selected-model artifact persistence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from training.runtime.checkpoint.io import (
    checkpoint_is_available,
    safe_torch_load,
)
from training.runtime.checkpoint.service import save_training_checkpoint
from training.runtime.device import (
    distributed_rank,
    is_fsdp_model,
    unwrap_model,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_settings import TrainingSettings
    from logger.project_logger import ProjectLogger
    from mmcrawler_datasets.validation.training_preflight import (
        EffectiveTrainingSplitReport,
    )
    from training.runtime.checkpoint.contract import CheckpointContract
    from training.runtime.loop.state import TrainingLoopState
    from training.runtime.preparation import PreparedTrainingRuntime


def best_checkpoint_path(last_checkpoint_path: Path) -> Path:
    """Derive the model-selection checkpoint beside the resumable path."""

    suffix = last_checkpoint_path.suffix
    filename = (
        f"{last_checkpoint_path.stem}.best{suffix}"
        if suffix
        else f"{last_checkpoint_path.name}.best"
    )
    return last_checkpoint_path.with_name(filename)


def save_epoch_checkpoint(
    *,
    path: Path,
    state: TrainingLoopState,
    test_loss: float | None,
    prepared: PreparedTrainingRuntime,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    dataset_root: Path,
    device: torch.device,
    sample_count: int,
    readiness: EffectiveTrainingSplitReport,
    logger: ProjectLogger,
    run_id: str | None = None,
    checkpoint_contract: "CheckpointContract | None" = None,
) -> Path:
    """Persist one resumable epoch checkpoint from prepared runtime state."""

    return save_training_checkpoint(
        model=prepared.model,
        checkpoint_path=path,
        model_settings=model_settings,
        training_settings=training_settings,
        training_plan=prepared.training_plan,
        distributed_context=prepared.distributed_context,
        dataset_root=dataset_root,
        device=device,
        sample_count=sample_count,
        effective_training_split=readiness,
        training_signal_by_modality=prepared.signal_tracker.to_payload(),
        loop_state=state,
        train_loss=state.final_loss,
        val_loss=state.last_val_loss,
        test_loss=test_loss,
        optimizer=prepared.optimizer,
        scheduler=prepared.scheduler,
        scaler=prepared.grad_scaler,
        logger=logger,
        initialization_metadata=prepared.initialization_metadata,
        dataset_manifest_sha256=prepared.dataset_manifest_sha256,
        run_id=run_id,
        resumed_from_run_id=prepared.resumed_from_run_id,
        checkpoint_contract=checkpoint_contract,
    )


def persist_selected_artifacts(
    *,
    model: torch.nn.Module,
    best_checkpoint_path: Path,
    export_directory: Path,
    dataset_root: Path,
    prepared: PreparedTrainingRuntime,
    model_exporter: Callable[..., dict[str, str]],
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
) -> tuple[Path, dict[str, str]]:
    """Export the selected best model, never the final in-memory epoch."""

    if not checkpoint_is_available(best_checkpoint_path):
        raise FileNotFoundError(
            f"best checkpoint does not exist: {best_checkpoint_path}"
        )
    export_paths: dict[str, str] = {}
    rank = distributed_rank(prepared.distributed_context)
    export_failure: list[str | None] = [None]
    if training_settings.export_artifacts and rank == 0:
        try:
            export_model = unwrap_model(model)
            if is_fsdp_model(model):
                checkpoint_payload = safe_torch_load(best_checkpoint_path)
                if not isinstance(checkpoint_payload, dict):
                    raise ValueError(
                        "FSDP checkpoint payload must be a dictionary"
                    )
                full_state = checkpoint_payload.get("model_state")
                if not isinstance(full_state, dict) or not full_state:
                    raise ValueError(
                        "FSDP checkpoint lacks consolidated inference model_state"
                    )
                export_model = prepared.model_factory(model_settings).cpu()
                export_model.load_state_dict(full_state)
            export_paths = model_exporter(
                model=export_model,
                export_directory=export_directory,
                model_settings=model_settings,
                training_settings=training_settings,
                dataset_root=dataset_root,
            )
        except Exception as exc:  # exception-rules: collective error relay
            export_failure[0] = f"{type(exc).__name__}: {exc}"
    if prepared.distributed_context.get("enabled"):
        broadcast_payload: list[object] = [export_paths, export_failure[0]]
        dist.broadcast_object_list(broadcast_payload, src=0)
        if not isinstance(broadcast_payload[0], dict):
            raise RuntimeError("rank-zero export result is not an object")
        export_paths = {
            str(key): str(value) for key, value in broadcast_payload[0].items()
        }
        export_failure[0] = (
            str(broadcast_payload[1])
            if broadcast_payload[1] is not None
            else None
        )
    if export_failure[0] is not None:
        raise RuntimeError(
            f"artifact export failed on rank zero: {export_failure[0]}"
        )
    return best_checkpoint_path, export_paths


__all__ = [
    "best_checkpoint_path",
    "persist_selected_artifacts",
    "save_epoch_checkpoint",
]
