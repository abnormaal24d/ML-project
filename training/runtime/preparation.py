"""Construct model, optimizer, scheduler, and precision runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch

from mmcrawler_datasets.collation.tensor_ops import IGNORE_LABEL
from mmcrawler_datasets.dataloader import build_dataloader
from mmcrawler_datasets.schema import DatasetSplit
from multimodal.model.contracts import CollatedBatch, LOGICAL_TO_PHYSICAL_MODALITIES
from multimodal.model.initialization import initialize_model_from_scratch
from multimodal.tasks.registry import get_task, task_requires_causal_decoder
from training.losses.objective import SupervisedOrSelfSupervisedLoss
from training.runtime.checkpoint.service import (
    resolve_resume_lineage,
    restore_checkpoint_if_requested,
)
from training.runtime.checkpoint.state import resume_optimizer_steps
from training.runtime.device import wrap_distributed_model
from training.runtime.planner import build_training_scale_plan
from training.runtime.precision import (
    PrecisionRuntime,
    build_grad_scaler,
    resolve_precision_runtime,
)
from training.runtime.signal import TrainingSignalTracker

if TYPE_CHECKING:
    from collections.abc import Callable

    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_settings import TrainingSettings
    from logger.project_logger import ProjectLogger
    from mmcrawler_datasets.dataset import MultimodalJsonlDataset
    from multimodal.model.model import MultimodalModel
    from multimodal.tokenization.text import VocabularyTokenizer
    from training.runtime.planner import TrainingScalePlan


class SchedulerFactory(Protocol):
    """Factory contract for schedulers that advance on optimizer updates."""

    def __call__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        settings: TrainingSettings,
        num_training_batches: int,
        completed_optimizer_steps: int,
    ) -> object | None: ...


@dataclass(frozen=True, slots=True)
class PreparedTrainingBackend:
    """Resolved training backend and its runtime requirements."""

    name: str
    requires_distributed_runtime: bool = False
    requires_gpu: bool = False
    requires_dense_sequence_targets: bool = False


def prepare_training_backend(
    *,
    training_settings: TrainingSettings,
) -> PreparedTrainingBackend:
    """Resolve runtime requirements for the validated training backend."""

    return {
        "pipeline_smoke": PreparedTrainingBackend(name="pipeline_smoke"),
        "dense_transformer": PreparedTrainingBackend(
            name="dense_transformer",
            requires_dense_sequence_targets=True,
        ),
    }[training_settings.training_backend]


def validate_dense_training_configuration(
    *,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    device: torch.device | None = None,
    modalities: tuple[str, ...] = (),
) -> None:
    """Fail fast when a dense campaign cannot execute causal training."""

    if training_settings.training_backend != "dense_transformer":
        return
    errors: list[str] = []
    decoder = model_settings.text_decoder
    if not decoder.enabled:
        errors.append("text_decoder.enabled must be true")
    if training_settings.text_tokenizer_artifact_version != "byte_bpe_v2":
        errors.append("tokenizer artifact version must be byte_bpe_v2")
    if decoder.vocab_size != training_settings.text_tokenizer_vocab_size:
        errors.append(
            "decoder vocabulary must match the configured tokenizer vocabulary"
        )
    configured_context = sum(
        (
            decoder.max_text_context_tokens,
            decoder.max_document_context_tokens,
            decoder.max_image_context_tokens,
            decoder.max_audio_context_tokens,
            decoder.max_video_context_tokens,
        )
    )
    if configured_context > decoder.max_context_tokens:
        errors.append("modality budgets exceed max_context_tokens")
    if training_settings.text_tokenizer_max_tokens > decoder.max_target_tokens:
        errors.append(
            "tokenizer sequence length exceeds text_decoder.max_target_tokens"
        )
    enabled_modalities = set(model_settings.enabled_modalities)
    for task_name in training_settings.tasks:
        definition = get_task(task_name)
        if definition is None:
            continue
        for required in definition.required_input_modalities:
            physical = LOGICAL_TO_PHYSICAL_MODALITIES.get(required, (required,))
            if physical and not set(physical).intersection(enabled_modalities):
                errors.append(
                    f"task {task_name!r} requires disabled modality {required!r}"
                )
    observed_modalities = set(modalities)
    if observed_modalities:
        required_physical = {
            physical
            for task_name in training_settings.tasks
            for definition in (get_task(task_name),)
            if definition is not None
            for required in definition.required_input_modalities
            for physical in LOGICAL_TO_PHYSICAL_MODALITIES.get(required, (required,))
        }
        missing = sorted(required_physical - observed_modalities)
        if missing:
            errors.append(
                "training split lacks required modalities: " + ", ".join(missing)
            )
    if training_settings.release_stage == "production_model":
        if training_settings.run_mode != "full":
            errors.append("production dense training requires run_mode='full'")
        if device is not None and device.type != "cuda":
            errors.append("production dense training requires a CUDA device")
        if training_settings.device == "cpu":
            errors.append("production dense training cannot target CPU")
    if errors:
        raise ValueError("dense_transformer preflight failed: " + "; ".join(errors))


def dense_batch_requires_causal_targets(*, batch: CollatedBatch) -> bool:
    """Return whether this batch contains a causal generative objective."""

    if batch.task_types:
        return any(task_requires_causal_decoder(name) for name in batch.task_types)
    labels = batch.decoder_labels
    return bool(
        labels is not None
        and labels.numel() > 0
        and labels.ne(IGNORE_LABEL).any().item()
    )


def validate_dense_decoder_batch(*, batch: CollatedBatch) -> None:
    """Validate decoder tensors for rows with causal generative objectives."""

    input_ids = batch.decoder_input_ids
    labels = batch.decoder_labels
    attention_mask = batch.decoder_attention_mask
    if input_ids is None or labels is None or attention_mask is None:
        raise ValueError(
            "dense_transformer causal batch requires decoder_input_ids, "
            "decoder_labels, and decoder_attention_mask"
        )
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError(
            "dense decoder inputs and labels must share [batch, tokens]"
        )
    if attention_mask.shape != input_ids.shape:
        raise ValueError("dense decoder attention mask must match input ids")
    if input_ids.shape[1] < 2:
        raise ValueError("dense decoder batch requires at least two tokens")

    required_rows = (
        [
            index
            for index, task_type in enumerate(batch.task_types)
            if task_requires_causal_decoder(task_type)
        ]
        if batch.task_types
        else list(range(input_ids.shape[0]))
    )
    if not required_rows:
        return
    if max(required_rows) >= input_ids.shape[0]:
        raise ValueError("dense decoder task rows exceed tensor batch size")

    supervised = labels.ne(IGNORE_LABEL) & attention_mask.to(dtype=torch.bool)
    missing_targets = [
        index
        for index in required_rows
        if not bool(supervised[index].any().item())
    ]
    if missing_targets:
        raise ValueError(
            "dense decoder causal rows contain no supervised target token: "
            f"rows={missing_targets}"
        )

    shifted = supervised[:, 1:]
    missing_shifted = [
        index
        for index in required_rows
        if not bool(shifted[index].any().item())
    ]
    if missing_shifted:
        raise ValueError(
            "dense decoder causal rows contain no shifted target: "
            f"rows={missing_shifted}"
        )


@dataclass(slots=True)
class PreparedTrainingRuntime:
    """Runtime objects created once before the first optimizer step."""

    model: torch.nn.Module
    model_factory: Callable[[ModelSettings], MultimodalModel]
    initialization_metadata: dict[str, object]
    distributed_context: dict[str, object]
    signal_tracker: TrainingSignalTracker
    loss_fn: SupervisedOrSelfSupervisedLoss
    training_plan: TrainingScalePlan
    optimizer: torch.optim.Optimizer
    scheduler: object | None
    precision_runtime: PrecisionRuntime
    grad_scaler: object | None
    resume_state: dict[str, object] | None
    resumed_from_run_id: str | None
    dataset_manifest_sha256: str


def open_training_split(
    *,
    dataset_root: Path,
    split: DatasetSplit,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    tokenizer: VocabularyTokenizer,
    logger: ProjectLogger,
    distributed: bool = True,
) -> tuple[MultimodalJsonlDataset, torch.utils.data.DataLoader[Any]]:
    """Open one canonical split with the configured dataloader policy."""

    return build_dataloader(
        dataset_root=dataset_root,
        split=split,
        model_settings=model_settings,
        training_settings=training_settings,
        tokenizer=tokenizer,
        logger=logger,
        distributed=distributed,
    )


def prepare_training_runtime(
    *,
    model_settings: ModelSettings,
    training_settings: TrainingSettings,
    model_factory: Callable[[ModelSettings], MultimodalModel],
    loss_factory: Callable[[TrainingSettings], SupervisedOrSelfSupervisedLoss],
    optimizer_factory: Callable[
        [torch.nn.Module, TrainingSettings],
        torch.optim.Optimizer,
    ],
    scheduler_factory: SchedulerFactory,
    logger: ProjectLogger,
    device: torch.device,
    dataset_root: Path,
    dataset_manifest_sha256: str,
    sample_count: int,
    modalities: tuple[str, ...],
    distributed_context: dict[str, object],
    num_training_batches: int,
) -> PreparedTrainingRuntime:
    """Prepare all runtime dependencies before the training loop starts."""

    validate_dense_training_configuration(
        model_settings=model_settings,
        training_settings=training_settings,
        device=device,
        modalities=modalities,
    )
    precision_runtime = resolve_precision_runtime(
        settings=training_settings,
        device=device,
    )
    base_model = model_factory(model_settings)
    initialization_metadata = initialize_model_from_scratch(
        model=base_model,
        seed=training_settings.seed,
    )
    model = wrap_distributed_model(
        model=base_model.to(device),
        settings=training_settings,
        device=device,
        distributed_context=distributed_context,
    )
    signal_tracker = TrainingSignalTracker(model=model, modalities=modalities)
    training_plan = build_training_scale_plan(
        dataset_size=sample_count,
        settings=training_settings,
        model_settings=model_settings,
        device=device,
    )
    logger.info("multimodal_training_scale_plan", **training_plan.to_dict())
    optimizer = optimizer_factory(model, training_settings)
    grad_scaler = build_grad_scaler(precision_runtime)
    completed_optimizer_steps = resume_optimizer_steps(settings=training_settings)
    scheduler = build_training_scheduler(
        scheduler_factory=scheduler_factory,
        optimizer=optimizer,
        settings=training_settings,
        num_training_batches=num_training_batches,
        completed_optimizer_steps=completed_optimizer_steps,
    )
    resume_state = restore_checkpoint_if_requested(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=grad_scaler,
        settings=training_settings,
        model_settings=model_settings,
        dataset_root=dataset_root,
        initialization_metadata=initialization_metadata,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    resumed_from_run_id = resolve_resume_lineage(settings=training_settings)
    return PreparedTrainingRuntime(
        model=model,
        model_factory=model_factory,
        initialization_metadata=initialization_metadata,
        distributed_context=distributed_context,
        signal_tracker=signal_tracker,
        loss_fn=loss_factory(training_settings),
        training_plan=training_plan,
        optimizer=optimizer,
        scheduler=scheduler,
        precision_runtime=precision_runtime,
        grad_scaler=grad_scaler,
        resume_state=resume_state,
        resumed_from_run_id=resumed_from_run_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


def build_training_scheduler(
    *,
    scheduler_factory: SchedulerFactory,
    optimizer: torch.optim.Optimizer,
    settings: TrainingSettings,
    num_training_batches: int,
    completed_optimizer_steps: int,
) -> object | None:
    """Build a scheduler against the explicit optimizer-update cadence."""

    return scheduler_factory(
        optimizer=optimizer,
        settings=settings,
        num_training_batches=num_training_batches,
        completed_optimizer_steps=completed_optimizer_steps,
    )


__all__ = [
    "PreparedTrainingBackend",
    "PreparedTrainingRuntime",
    "SchedulerFactory",
    "build_training_scheduler",
    "open_training_split",
    "prepare_training_backend",
    "prepare_training_runtime",
    "validate_dense_decoder_batch",
    "validate_dense_training_configuration",
]
