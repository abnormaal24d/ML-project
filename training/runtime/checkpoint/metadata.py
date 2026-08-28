"""Checkpoint metadata and reproducibility fingerprints."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mmcrawler_datasets.validation.training_preflight import (
    EffectiveTrainingSplitReport,
)

if TYPE_CHECKING:
    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_settings import TrainingSettings
    from training.runtime.planner import TrainingScalePlan


def build_checkpoint_metadata(
    *,
    model_settings: ModelSettings,
    settings: TrainingSettings,
    training_plan: TrainingScalePlan,
    distributed_context: dict[str, object],
    dataset_root: Path,
    device: object,
    completed_epochs: int,
    sample_count: int,
    effective_training_split: EffectiveTrainingSplitReport,
    training_signal_by_modality: dict[str, dict[str, object]],
    total_batches: int,
    final_loss: float,
    train_loss: float | None,
    val_loss: float | None,
    test_loss: float | None,
    initialization_metadata: dict[str, object] | None = None,
    dataset_manifest_sha256: str,
    run_id: str | None = None,
    resumed_from_run_id: str | None = None,
) -> dict[str, object]:
    """Return metadata stored inside the checkpoint payload.

    The dataset manifest digest is the immutable dataset identity used for
    reproducibility fingerprints. It is always provided by the orchestration
    layer, which avoids an O(dataset) tree scan during checkpoint save.
    """

    payload: dict[str, object] = {
        "epochs": completed_epochs,
        "configured_epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "sample_count": sample_count,
        "effective_train_sample_count": effective_training_split.sample_count,
        "effective_task_counts": effective_training_split.task_counts,
        "effective_modality_counts": (
            effective_training_split.modality_counts
        ),
        "training_signal_by_modality": training_signal_by_modality,
        "batch_count": total_batches,
        "final_loss": final_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "dataset_root": dataset_root.as_posix(),
        "device": str(device),
        "model_seed": settings.seed,
        "run_mode": settings.run_mode,
        "release_stage": settings.release_stage,
        "training_backend": settings.training_backend,
        "effective_min_task_samples": settings.effective_min_task_samples(),
        "training_scale_plan": training_plan.to_dict(),
        "distributed_strategy": distributed_context["strategy"],
        "world_size": distributed_context["world_size"],
        "checkpoint_schema": build_checkpoint_fingerprint_schema(
            model_settings=model_settings,
            training_settings=settings,
            dataset_root=dataset_root,
            initialization_metadata=initialization_metadata,
            dataset_manifest_sha256=dataset_manifest_sha256,
        ),
    }
    if initialization_metadata is not None:
        payload["initialization"] = dict(initialization_metadata)
    if run_id is not None:
        payload["run_id"] = _require_nonempty_run_id(
            run_id,
            field="run_id",
        )
    if _resume_requested(settings=settings):
        if resumed_from_run_id is None:
            raise ValueError(
                "resumed checkpoints require resumed_from_run_id metadata"
            )
    if resumed_from_run_id is not None:
        payload["resumed_from_run_id"] = _require_nonempty_run_id(
            resumed_from_run_id,
            field="resumed_from_run_id",
        )
    return payload


def resolve_resumed_from_run_id(metadata: dict[str, object]) -> str:
    """Return the direct source attempt identity for a resumed checkpoint.

    ``run_id`` is the canonical identity written by current training runs.
    Older checkpoints may use ``attempt_id``.  A checkpoint that itself was
    resumed but lacks either direct identity may still expose its prior source
    through ``resumed_from_run_id``; that fallback intentionally comes last so
    a resume chain retains its direct parent when one is available.
    """

    for field in ("run_id", "attempt_id", "resumed_from_run_id"):
        value = metadata.get(field)
        if value is None:
            continue
        return _require_nonempty_run_id(value, field=field)
    raise ValueError(
        "resume checkpoint metadata lacks a source run_id, attempt_id, or "
        "resumed_from_run_id"
    )


def _resume_requested(*, settings: TrainingSettings) -> bool:
    checkpoint_value = settings.resume_from_checkpoint
    return checkpoint_value is not None and bool(str(checkpoint_value).strip())


def _require_nonempty_run_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"checkpoint metadata {field} must be a non-empty string"
        )
    return value


def build_checkpoint_fingerprint_schema(
    *,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    dataset_root: Path,
    initialization_metadata: dict[str, object] | None = None,
    dataset_manifest_sha256: str,
) -> dict[str, str]:
    """Build exact fingerprints required for safe checkpoint resume.

    The dataset manifest digest is the immutable dataset identity and is
    always required.
    """

    model_payload = model_settings.model_dump(mode="json")
    training_payload = training_settings.model_dump(mode="json")
    # Resume location is an invocation detail, not part of the training
    # semantics. Including it would make every freshly requested resume reject
    # the checkpoint that it is trying to load.
    training_payload.pop("resume_from_checkpoint", None)
    tokenizer_payload = {
        "backend": training_settings.text_tokenizer_backend,
        "name": training_settings.text_tokenizer_name,
        "max_tokens": training_settings.text_tokenizer_max_tokens,
        "artifact_sha256": training_settings.text_tokenizer_sha256,
        "artifact_version": (
            training_settings.text_tokenizer_artifact_version
        ),
        "vocab_size": training_settings.text_tokenizer_vocab_size,
        "special_tokens": training_settings.text_tokenizer_special_tokens,
    }
    label_payload = {
        "tasks": tuple(training_settings.tasks),
        "min_task_samples": training_settings.effective_min_task_samples(),
    }
    dependency_payload = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    dataset_fingerprint = resolve_dataset_fingerprint(
        dataset_manifest_sha256=dataset_manifest_sha256
    )
    schema = {
        "model_config_fingerprint": _stable_payload_hash(model_payload),
        "training_config_fingerprint": _stable_payload_hash(training_payload),
        "dataset_fingerprint": dataset_fingerprint,
        "tokenizer_fingerprint": _stable_payload_hash(tokenizer_payload),
        "label_mapping_fingerprint": _stable_payload_hash(label_payload),
        "dependency_fingerprint": _stable_payload_hash(dependency_payload),
    }
    if initialization_metadata is not None:
        schema["initialization_fingerprint"] = _stable_payload_hash(
            initialization_metadata
        )
    return schema


def resolve_dataset_fingerprint(
    *,
    dataset_manifest_sha256: str,
) -> str:
    """Return the validated immutable dataset identity.

    The orchestration layer always supplies the already-verified manifest
    digest, making checkpoint save and restore O(1). A missing digest is a
    configuration error, not a fallback.
    """

    normalized = dataset_manifest_sha256.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(
            "dataset_manifest_sha256 must be a 64-character hex digest"
        )
    return normalized


def _stable_payload_hash(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


__all__ = [
    "build_checkpoint_fingerprint_schema",
    "build_checkpoint_metadata",
    "resolve_resumed_from_run_id",
    "resolve_dataset_fingerprint",
]
