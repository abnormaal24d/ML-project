"""Public checkpoint save, restore, and model-weight services."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from mmcrawler_datasets.validation.training_preflight import (
    EffectiveTrainingSplitReport,
)
from training.runtime.checkpoint.contract import (
    CheckpointContract,
    require_blob_checkpoint,
    require_checkpoint_headers,
    write_checkpoint_headers,
)
from training.runtime.checkpoint.io import (
    atomic_torch_save,
    checkpoint_is_available,
    checkpoint_sha256,
    safe_torch_load,
)
from training.runtime.checkpoint.metadata import (
    build_checkpoint_fingerprint_schema,
    build_checkpoint_metadata,
    resolve_resumed_from_run_id,
)
from training.runtime.checkpoint.state import (
    _restore_random_state,
    build_training_state_payload,
    capture_random_state,
    validate_epoch_resume_state,
)
from training.runtime.device import (
    distributed_rank,
    is_fsdp_model,
    unwrap_model,
)

if TYPE_CHECKING:
    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_settings import TrainingSettings
    from training.runtime.loop.state import TrainingLoopState
    from training.runtime.planner import TrainingScalePlan


CURRENT_CHECKPOINT_PAYLOAD_VERSION = 1
DISTRIBUTED_SHARDED_V2_FORMAT = "distributed_sharded_v2"
SINGLE_FILE_FORMAT = "single_file"

_MISSING = object()


def _resolve_checkpoint_format(payload: dict[str, object]) -> str:
    format_value = payload.get("checkpoint_format", _MISSING)
    if format_value is _MISSING:
        return SINGLE_FILE_FORMAT
    if not isinstance(format_value, str):
        raise ValueError("checkpoint_format must be a string")
    if format_value == DISTRIBUTED_SHARDED_V2_FORMAT:
        return DISTRIBUTED_SHARDED_V2_FORMAT
    if format_value == SINGLE_FILE_FORMAT:
        return SINGLE_FILE_FORMAT
    raise ValueError(f"unsupported checkpoint format: {format_value!r}")


def _require_checkpoint_identity(
    payload: dict[str, object],
    *,
    expected_model_family: str,
) -> None:
    family = payload.get("model_family", _MISSING)
    if family is _MISSING:
        raise ValueError(
            "checkpoint payload missing required field: model_family"
        )
    if family is None:
        raise ValueError("checkpoint model_family must not be null")
    if family != expected_model_family:
        raise ValueError(
            f"checkpoint model_family mismatch: expected {expected_model_family!r}, "
            f"got {family!r}"
        )


def load_model_weights(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    model_settings: ModelSettings,
    contract: CheckpointContract | None = None,
) -> None:
    """Load only model weights from a checksum-verified checkpoint."""

    if contract is not None:
        if contract.checkpoint_blob_storage is not None:
            require_blob_checkpoint(
                checkpoint_path=checkpoint_path,
                blob_storage=contract.checkpoint_blob_storage,
            )
        if contract.checkpoint_headers:
            require_checkpoint_headers(
                checkpoint_path=checkpoint_path,
                expected_model_family=model_settings.model_family,
                expected_artifact_version=model_settings.artifact_version,
            )
    payload = safe_torch_load(checkpoint_path)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")

    _require_checkpoint_identity(
        payload, expected_model_family=model_settings.model_family
    )

    payload_version = payload.get("checkpoint_payload_version", _MISSING)
    if payload_version is _MISSING:
        raise ValueError(
            "checkpoint payload missing required field: checkpoint_payload_version"
        )
    if payload_version != CURRENT_CHECKPOINT_PAYLOAD_VERSION:
        raise ValueError(
            f"unsupported checkpoint payload version: {payload_version!r}; "
            f"expected {CURRENT_CHECKPOINT_PAYLOAD_VERSION}"
        )

    format_type = _resolve_checkpoint_format(payload)
    if format_type == DISTRIBUTED_SHARDED_V2_FORMAT:
        if is_fsdp_model(model):
            _load_distributed_model_state(
                model=model, payload=payload, checkpoint_path=checkpoint_path
            )
            return
        model_state = payload.get("model_state")
        if not isinstance(model_state, dict) or not model_state:
            raise ValueError(
                "distributed checkpoint lacks consolidated inference model_state"
            )
        unwrap_model(model).load_state_dict(model_state)
        return
    if format_type == SINGLE_FILE_FORMAT:
        model_state = payload.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("checkpoint lacks complete model_state")
        unwrap_model(model).load_state_dict(model_state)
        return
    raise ValueError(f"unsupported checkpoint format: {format_type!r}")


def restore_checkpoint_if_requested(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    scaler: object | None = None,
    settings: TrainingSettings,
    model_settings: ModelSettings,
    dataset_root: Path,
    initialization_metadata: dict[str, object] | None = None,
    dataset_manifest_sha256: str,
    contract: CheckpointContract | None = None,
) -> dict[str, object] | None:
    """Restore model/optimizer/scheduler state when resume is configured."""

    checkpoint_value = settings.resume_from_checkpoint
    if checkpoint_value is None or not str(checkpoint_value).strip():
        return None

    checkpoint_path = Path(checkpoint_value)
    if contract is not None:
        if contract.checkpoint_blob_storage is not None:
            require_blob_checkpoint(
                checkpoint_path=checkpoint_path,
                blob_storage=contract.checkpoint_blob_storage,
            )
        if contract.checkpoint_headers:
            require_checkpoint_headers(
                checkpoint_path=checkpoint_path,
                expected_model_family=model_settings.model_family,
                expected_artifact_version=model_settings.artifact_version,
            )
    if not checkpoint_is_available(checkpoint_path):
        raise FileNotFoundError(
            f"resume checkpoint not found: {checkpoint_path}"
        )

    payload = safe_torch_load(checkpoint_path)
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint payload must be a dictionary")

    _require_checkpoint_identity(
        payload, expected_model_family=model_settings.model_family
    )

    payload_version = payload.get("checkpoint_payload_version", _MISSING)
    if payload_version is _MISSING:
        raise ValueError(
            "resume checkpoint missing required field: checkpoint_payload_version"
        )
    if payload_version != CURRENT_CHECKPOINT_PAYLOAD_VERSION:
        raise ValueError(
            f"unsupported checkpoint payload version: {payload_version!r}; "
            f"expected {CURRENT_CHECKPOINT_PAYLOAD_VERSION}"
        )

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("resume checkpoint lacks metadata")
    stored_schema = metadata.get("checkpoint_schema")
    if not isinstance(stored_schema, dict):
        raise ValueError("resume checkpoint lacks fingerprint schema")
    expected_schema = build_checkpoint_fingerprint_schema(
        model_settings=model_settings,
        training_settings=settings,
        dataset_root=dataset_root,
        initialization_metadata=initialization_metadata,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    mismatches = sorted(
        name
        for name, expected in expected_schema.items()
        if stored_schema.get(name) != expected
    )
    if mismatches:
        raise ValueError(
            "resume checkpoint fingerprint mismatch: " + ", ".join(mismatches)
        )

    format_type = _resolve_checkpoint_format(payload)
    if format_type == DISTRIBUTED_SHARDED_V2_FORMAT:
        _restore_distributed_training_state(
            model=model,
            optimizer=optimizer,
            payload=payload,
            checkpoint_path=checkpoint_path,
        )
    elif format_type == SINGLE_FILE_FORMAT:
        concrete_model = unwrap_model(model)
        model_state = payload.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("resume checkpoint lacks complete model_state")
        concrete_model.load_state_dict(model_state)

        optimizer_state = payload.get("optimizer_state")
        if not isinstance(optimizer_state, dict):
            raise ValueError(
                "resume checkpoint lacks complete optimizer_state"
            )
        optimizer.load_state_dict(optimizer_state)
    else:
        raise ValueError(f"unsupported checkpoint format: {format_type!r}")

    scheduler_state = payload.get("scheduler_state")
    if scheduler is not None and not isinstance(scheduler_state, dict):
        raise ValueError("resume checkpoint lacks complete scheduler_state")
    if (
        scheduler is not None
        and isinstance(scheduler_state, dict)
        and hasattr(scheduler, "load_state_dict")
    ):
        scheduler.load_state_dict(scheduler_state)

    scaler_state = payload.get("scaler_state")
    if scaler is not None and not isinstance(scaler_state, dict):
        raise ValueError("resume checkpoint lacks complete scaler_state")
    if (
        scaler is not None
        and isinstance(scaler_state, dict)
        and hasattr(scaler, "load_state_dict")
    ):
        scaler.load_state_dict(scaler_state)

    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("resume checkpoint lacks complete training_state")
    validate_epoch_resume_state(training_state)

    random_states_by_rank = training_state.get("random_state_by_rank")
    if isinstance(random_states_by_rank, list):
        if (
            dist.is_initialized()
            and len(random_states_by_rank) != dist.get_world_size()
        ):
            raise ValueError(
                "resume checkpoint RNG world size does not match the active "
                "process group"
            )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank >= len(random_states_by_rank):
            raise ValueError(
                "resume checkpoint lacks RNG state for distributed rank "
                f"{rank}"
            )
        rank_random_state = random_states_by_rank[rank]
        if not isinstance(rank_random_state, dict):
            raise ValueError(
                f"resume checkpoint RNG state for rank {rank} is invalid"
            )
        _restore_random_state(state=rank_random_state)
    else:
        random_state = training_state.get("random_state")
        if isinstance(random_state, dict):
            _restore_random_state(state=random_state)

    return training_state


def resolve_resume_lineage(
    *,
    settings: TrainingSettings,
) -> str | None:
    """Return the direct source run identity configured for this resume.

    Restoring state remains backwards-compatible for legacy checkpoints.  The
    training persistence path calls this separately so any newly written
    resumed checkpoint must carry enough provenance for a receipt.
    """

    checkpoint_value = settings.resume_from_checkpoint
    if checkpoint_value is None or not str(checkpoint_value).strip():
        return None
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_is_available(checkpoint_path):
        raise FileNotFoundError(
            f"resume checkpoint not found: {checkpoint_path}"
        )
    payload = safe_torch_load(checkpoint_path)
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint payload must be a dictionary")

    _require_checkpoint_identity(
        payload, expected_model_family="multimodal_model"
    )

    payload_version = payload.get("checkpoint_payload_version", _MISSING)
    if payload_version is _MISSING:
        raise ValueError(
            "resume checkpoint missing required field: checkpoint_payload_version"
        )
    if payload_version != CURRENT_CHECKPOINT_PAYLOAD_VERSION:
        raise ValueError(
            f"unsupported checkpoint payload version: {payload_version!r}; "
            f"expected {CURRENT_CHECKPOINT_PAYLOAD_VERSION}"
        )

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("resume checkpoint lacks metadata")
    return resolve_resumed_from_run_id(metadata)


def save_checkpoint(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    metadata: dict[str, object],
    logger: object | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    training_state: dict[str, object] | None = None,
    checkpoint_contract: "CheckpointContract | None" = None,
) -> Path:
    """Persist a multimodal model checkpoint at one explicit path."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    training_settings_payload = training_settings.model_dump(mode="json")
    training_settings_payload["effective_min_task_samples"] = (
        training_settings.effective_min_task_samples()
    )
    payload: dict[str, object] = {
        "checkpoint_payload_version": CURRENT_CHECKPOINT_PAYLOAD_VERSION,
        "checkpoint_format": SINGLE_FILE_FORMAT,
        "artifact_version": model_settings.artifact_version,
        "model_family": model_settings.model_family,
        "model_state": unwrap_model(model).state_dict(),
        "model_settings": model_settings.model_dump(mode="json"),
        "training_settings": training_settings_payload,
        "metadata": metadata,
        "training_state": training_state or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler_state"] = scaler.state_dict()

    blob_storage = (
        checkpoint_contract.checkpoint_blob_storage
        if checkpoint_contract is not None
        else None
    )
    atomic_torch_save(
        payload=payload,
        checkpoint_path=checkpoint_path,
        blob_storage=blob_storage,
    )
    if (
        checkpoint_contract is not None
        and checkpoint_contract.checkpoint_headers
    ):
        write_checkpoint_headers(
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            sha256=checkpoint_sha256(checkpoint_path),
            artifact_version=model_settings.artifact_version,
            model_family=model_settings.model_family,
        )
    if logger is not None:
        info = getattr(logger, "info", None)
        if callable(info):
            info(
                "multimodal_checkpoint_saved",
                checkpoint_path=str(checkpoint_path),
            )
    return checkpoint_path


def save_training_checkpoint(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    training_plan: TrainingScalePlan,
    distributed_context: dict[str, object],
    sample_count: int,
    effective_training_split: EffectiveTrainingSplitReport,
    training_signal_by_modality: dict[str, dict[str, object]],
    loop_state: TrainingLoopState,
    train_loss: float | None,
    val_loss: float | None,
    test_loss: float | None,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    scaler: object | None = None,
    logger: object | None = None,
    dataset_root: Path,
    device: object | None = None,
    initialization_metadata: dict[str, object] | None = None,
    dataset_manifest_sha256: str,
    run_id: str | None = None,
    resumed_from_run_id: str | None = None,
    checkpoint_contract: "CheckpointContract | None" = None,
) -> Path:
    """Save the checkpoint (export is a separate action performed by the caller)."""

    if str(distributed_context.get("strategy")) == "fsdp":
        return _save_distributed_training_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            model_settings=model_settings,
            training_settings=training_settings,
            training_plan=training_plan,
            distributed_context=distributed_context,
            sample_count=sample_count,
            effective_training_split=effective_training_split,
            training_signal_by_modality=training_signal_by_modality,
            loop_state=loop_state,
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            logger=logger,
            dataset_root=dataset_root,
            device=device,
            initialization_metadata=initialization_metadata,
            dataset_manifest_sha256=dataset_manifest_sha256,
            run_id=run_id,
            resumed_from_run_id=resumed_from_run_id,
        )

    rank = distributed_rank(distributed_context)
    local_random_state = capture_random_state()
    random_states_by_rank: list[object] | None = None
    if distributed_context.get("enabled"):
        if not dist.is_initialized():
            raise RuntimeError(
                "distributed checkpoint requires a process group"
            )
        random_states_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(random_states_by_rank, local_random_state)
    failure: list[str | None] = [None]
    if rank == 0:
        try:
            training_state = build_training_state_payload(
                state=loop_state,
                val_loss=val_loss,
                random_state=local_random_state,
                training_plan=training_plan,
                stage_state=_active_stage_state(training_settings),
                sampler_position=0,
                gradient_accumulation_position=0,
                rank=rank,
            )
            if random_states_by_rank is not None:
                training_state["random_state_by_rank"] = random_states_by_rank
            save_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                model_settings=model_settings,
                training_settings=training_settings,
                metadata=build_checkpoint_metadata(
                    model_settings=model_settings,
                    settings=training_settings,
                    training_plan=training_plan,
                    distributed_context=distributed_context,
                    dataset_root=dataset_root,
                    device=device,
                    completed_epochs=loop_state.completed_epochs,
                    sample_count=sample_count,
                    effective_training_split=effective_training_split,
                    training_signal_by_modality=training_signal_by_modality,
                    total_batches=loop_state.total_batches,
                    final_loss=loop_state.final_loss,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    test_loss=test_loss,
                    initialization_metadata=initialization_metadata,
                    dataset_manifest_sha256=dataset_manifest_sha256,
                    run_id=run_id,
                    resumed_from_run_id=resumed_from_run_id,
                ),
                logger=logger,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                training_state=training_state,
                checkpoint_contract=checkpoint_contract,
            )
        except Exception as exc:  # exception-rules: collective error relay
            failure[0] = f"{type(exc).__name__}: {exc}"
    if distributed_context.get("enabled"):
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(
            f"checkpoint save failed on rank zero: {failure[0]}"
        )
    if not checkpoint_is_available(checkpoint_path):
        raise FileNotFoundError(
            f"completed checkpoint is not visible: {checkpoint_path}"
        )
    return checkpoint_path


def _active_stage_state(settings: TrainingSettings) -> object:
    from training.runtime.job_status.models import TrainingStageState
    from training.runtime.planner import TrainingStage

    return TrainingStageState(
        current_stage=TrainingStage(settings.training_stage)
    )


def _save_distributed_training_checkpoint(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    training_plan: TrainingScalePlan,
    distributed_context: dict[str, object],
    sample_count: int,
    effective_training_split: EffectiveTrainingSplitReport,
    training_signal_by_modality: dict[str, dict[str, object]],
    loop_state: TrainingLoopState,
    train_loss: float | None,
    val_loss: float | None,
    test_loss: float | None,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    scaler: object | None,
    logger: object | None,
    dataset_root: Path,
    device: object | None,
    initialization_metadata: dict[str, object] | None,
    dataset_manifest_sha256: str,
    run_id: str | None,
    resumed_from_run_id: str | None,
) -> Path:
    """Collectively persist FSDP model/optimizer shards and rank state."""

    if not is_fsdp_model(model):
        raise TypeError("fsdp checkpointing requires an FSDP-wrapped model")
    if not dist.is_initialized():
        raise RuntimeError(
            "FSDP checkpointing requires an initialized process group"
        )
    rank = distributed_rank(distributed_context)
    raw_world_size = distributed_context.get("world_size")
    if not isinstance(raw_world_size, int) or isinstance(raw_world_size, bool):
        raise TypeError("distributed world_size must be an integer")
    world_size = raw_world_size
    root = checkpoint_path.parent / f"{checkpoint_path.name}.distributed.d"
    token: list[str | None] = [uuid.uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(token, src=0)
    if token[0] is None:
        raise RuntimeError("rank zero did not provide a checkpoint version")
    version_name = f"v-{token[0]}"
    version_dir = root / version_name
    if rank == 0:
        version_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_state_dict,
    )
    from torch.distributed.checkpoint.state_dict_saver import (
        save as distributed_save,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=options
    )
    distributed_save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=version_dir / "state",
    )
    full_model_state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    runtime_payload: dict[str, object] = {
        "random_state": capture_random_state(),
        "sampler_position": 0,
        "gradient_accumulation_position": 0,
        "rank": rank,
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        runtime_payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        runtime_payload["scaler_state"] = scaler.state_dict()
    _save_rank_runtime(
        version_dir=version_dir, rank=rank, payload=runtime_payload
    )
    dist.barrier()

    failure: list[str | None] = [None]
    if rank == 0:
        try:
            manifest = _build_distributed_manifest(
                version_dir=version_dir, world_size=world_size
            )
            metadata = build_checkpoint_metadata(
                model_settings=model_settings,
                settings=training_settings,
                training_plan=training_plan,
                distributed_context=distributed_context,
                dataset_root=dataset_root,
                device=device,
                completed_epochs=loop_state.completed_epochs,
                sample_count=sample_count,
                effective_training_split=effective_training_split,
                training_signal_by_modality=training_signal_by_modality,
                total_batches=loop_state.total_batches,
                final_loss=loop_state.final_loss,
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                initialization_metadata=initialization_metadata,
                dataset_manifest_sha256=dataset_manifest_sha256,
                run_id=run_id,
                resumed_from_run_id=resumed_from_run_id,
            )
            descriptor = {
                "checkpoint_payload_version": CURRENT_CHECKPOINT_PAYLOAD_VERSION,
                "artifact_version": model_settings.artifact_version,
                "model_family": model_settings.model_family,
                "checkpoint_format": DISTRIBUTED_SHARDED_V2_FORMAT,
                "distributed_manifest": manifest,
                "model_state": full_model_state,
                "model_settings": model_settings.model_dump(mode="json"),
                "training_settings": training_settings.model_dump(mode="json"),
                "metadata": metadata,
                "scheduler_state": (
                    scheduler.state_dict()
                    if scheduler is not None
                    and hasattr(scheduler, "state_dict")
                    else None
                ),
                "scaler_state": (
                    scaler.state_dict()
                    if scaler is not None and hasattr(scaler, "state_dict")
                    else None
                ),
                "training_state": build_training_state_payload(
                    state=loop_state,
                    val_loss=val_loss,
                    random_state={},
                    training_plan=training_plan,
                    stage_state=_active_stage_state(training_settings),
                    sampler_position=0,
                    gradient_accumulation_position=0,
                    rank=0,
                ),
            }
            atomic_torch_save(
                payload=descriptor, checkpoint_path=checkpoint_path
            )
        except Exception as exc:  # exception-rules: collective error relay
            failure[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        if rank == 0:
            shutil.rmtree(version_dir, ignore_errors=True)
        raise RuntimeError(
            f"distributed checkpoint commit failed: {failure[0]}"
        )
    dist.barrier()
    if logger is not None and rank == 0:
        info = getattr(logger, "info", None)
        if callable(info):
            info(
                "multimodal_distributed_checkpoint_saved",
                checkpoint_path=str(checkpoint_path),
            )
    return checkpoint_path


def _build_distributed_manifest(
    *, version_dir: Path, world_size: int
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(
        candidate
        for candidate in version_dir.rglob("*")
        if candidate.is_file()
    ):
        files.append(
            {
                "path": path.relative_to(version_dir).as_posix(),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
        )
    if not files:
        raise RuntimeError("distributed checkpoint produced no shard files")
    return {
        "version_dir": version_dir.relative_to(
            version_dir.parent.parent
        ).as_posix(),
        "world_size": world_size,
        "files": files,
    }


def _distributed_version_dir(
    *, payload: dict[str, object], checkpoint_path: Path
) -> Path:
    manifest = payload.get("distributed_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("distributed checkpoint lacks a manifest")
    relative = manifest.get("version_dir")
    if not isinstance(relative, str) or not relative:
        raise ValueError("distributed checkpoint manifest lacks version_dir")
    root = (
        checkpoint_path.parent / f"{checkpoint_path.name}.distributed.d"
    ).resolve()
    candidate = (root.parent / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_dir():
        raise ValueError("distributed checkpoint version path is invalid")
    expected_world_size = int(manifest.get("world_size", 0))
    current_world_size = dist.get_world_size() if dist.is_initialized() else 1
    if expected_world_size != current_world_size:
        raise ValueError(
            "distributed checkpoint world-size mismatch: "
            f"stored={expected_world_size}, current={current_world_size}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("distributed checkpoint manifest has no files")
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("distributed checkpoint file record is invalid")
        rel = record.get("path")
        digest = record.get("sha256")
        if not isinstance(rel, str) or not isinstance(digest, str):
            raise ValueError("distributed checkpoint file metadata is invalid")
        path = (candidate / rel).resolve()
        if not path.is_relative_to(candidate) or not path.is_file():
            raise FileNotFoundError(
                f"distributed checkpoint shard is missing: {rel}"
            )
        if _sha256(path) != digest:
            raise ValueError(
                f"distributed checkpoint shard checksum mismatch: {rel}"
            )
    return candidate


def _restore_distributed_training_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: dict[str, object],
    checkpoint_path: Path,
) -> None:
    if not is_fsdp_model(model):
        raise TypeError(
            "distributed checkpoint restore requires an FSDP model"
        )
    version_dir = _distributed_version_dir(
        payload=payload, checkpoint_path=checkpoint_path
    )
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )
    from torch.distributed.checkpoint.state_dict_loader import (
        load as distributed_load,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=options
    )
    state = {"model": model_state, "optimizer": optimizer_state}
    distributed_load(state, checkpoint_id=version_dir / "state")
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
        options=options,
    )
    rank = dist.get_rank()
    runtime = _load_rank_runtime(version_dir=version_dir, rank=rank)
    random_state = runtime.get("random_state")
    if isinstance(random_state, dict):
        _restore_random_state(state=random_state)


def _load_distributed_model_state(
    *,
    model: torch.nn.Module,
    payload: dict[str, object],
    checkpoint_path: Path,
) -> None:
    if not is_fsdp_model(model):
        raise TypeError("sharded model weights require an FSDP-wrapped model")
    version_dir = _distributed_version_dir(
        payload=payload, checkpoint_path=checkpoint_path
    )
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )
    from torch.distributed.checkpoint.state_dict_loader import (
        load as distributed_load,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state = get_model_state_dict(model, options=options)
    state = {"model": model_state}
    distributed_load(state, checkpoint_id=version_dir / "state")
    set_model_state_dict(model, model_state_dict=model_state, options=options)


def _save_rank_runtime(
    *, version_dir: Path, rank: int, payload: dict[str, object]
) -> None:
    destination = version_dir / f"rank-{rank:05d}.runtime.pt"
    fd, temporary_name = tempfile.mkstemp(
        dir=version_dir, prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_rank_runtime(*, version_dir: Path, rank: int) -> dict[str, object]:
    path = version_dir / f"rank-{rank:05d}.runtime.pt"
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("rank runtime checkpoint must be a dictionary")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "load_model_weights",
    "resolve_resume_lineage",
    "restore_checkpoint_if_requested",
    "save_checkpoint",
    "save_training_checkpoint",
]
