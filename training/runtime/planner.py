"""Training scale planning and VRAM estimation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

from training.runtime.job_status.models import TrainingStage
from training.runtime.precision import (
    bytes_per_element,
    precision_from_settings,
)

if TYPE_CHECKING:
    from config.multimodal.model_settings import ModelSettings
    from config.multimodal.training_settings import TrainingSettings


TRAINING_STAGE_ORDER: tuple[TrainingStage, ...] = tuple(TrainingStage)


def stage_index(stage: TrainingStage | str) -> int:
    resolved = TrainingStage(stage)
    return TRAINING_STAGE_ORDER.index(resolved)


def stages_between(
    *,
    start: TrainingStage | str = TrainingStage.DATASET_FREEZE,
    stop: TrainingStage | str = TrainingStage.PROMOTION,
) -> tuple[TrainingStage, ...]:
    first = stage_index(start)
    last = stage_index(stop)
    if first > last:
        raise ValueError("training stage start must not follow stop")
    return TRAINING_STAGE_ORDER[first : last + 1]


@dataclass(frozen=True, slots=True)
class StageExecutionArtifact:
    """Fingerprint-bearing output of one completed campaign stage."""

    output_fingerprint: str
    parent_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if not self.output_fingerprint.strip():
            raise ValueError("stage output fingerprint must be explicit")


class TrainingStageExecutionError(RuntimeError):
    """A campaign stage failed after its resumable state was persisted."""

    def __init__(self, *, stage: TrainingStage, cause: BaseException) -> None:
        super().__init__(f"training stage {stage.value} failed: {cause}")
        self.stage = stage
        self.__cause__ = cause


StageHandler = Callable[[str | None, str | None], StageExecutionArtifact]


class TrainingStageExecutor:
    """Execute only unfinished stages and persist every transition."""

    def __init__(
        self,
        *,
        handlers: Mapping[TrainingStage, StageHandler],
        load_state: Callable[[], object | None],
        persist_state: Callable[[object], None],
    ) -> None:
        self._handlers = dict(handlers)
        self._load_state = load_state
        self._persist_state = persist_state

    def run(
        self,
        *,
        start: TrainingStage | str = TrainingStage.DATASET_FREEZE,
        stop: TrainingStage | str = TrainingStage.PROMOTION,
    ) -> object:
        from training.runtime.job_status.models import TrainingStageState

        selected = stages_between(start=start, stop=stop)
        state = self._load_state()
        if state is None:
            state = TrainingStageState(current_stage=selected[0])
        if not isinstance(state, TrainingStageState):
            raise TypeError("loaded training stage state has an invalid type")
        if state.failed_stage is not None:
            state = state.retry()
            self._persist_state(state)

        while state.current_stage in selected:
            stage = state.current_stage
            handler = self._handlers.get(stage)
            if handler is None:
                failed = state.fail()
                self._persist_state(failed)
                raise TrainingStageExecutionError(
                    stage=stage,
                    cause=KeyError(f"no handler registered for {stage.value}"),
                )
            try:
                artifact = handler(
                    state.input_fingerprint,
                    state.parent_checkpoint,
                )
                if not isinstance(artifact, StageExecutionArtifact):
                    raise TypeError(
                        "stage handlers must return StageExecutionArtifact"
                    )
            except Exception as exc:
                failed = state.fail()
                self._persist_state(failed)
                raise TrainingStageExecutionError(
                    stage=stage, cause=exc
                ) from exc

            index = stage_index(stage)
            next_stage = (
                TRAINING_STAGE_ORDER[index + 1]
                if index + 1 < len(TRAINING_STAGE_ORDER)
                else None
            )
            state = state.complete(
                output_fingerprint=artifact.output_fingerprint,
                next_stage=next_stage,
                parent_checkpoint=artifact.parent_checkpoint,
            )
            self._persist_state(state)
        return state


@dataclass(frozen=True, slots=True)
class TrainingScalePlan:
    """Estimated compute envelope for one training run."""

    dataset_size: int
    model_parameters: int
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    epochs: int
    optimizer_steps_per_epoch: int
    estimated_steps: int
    checkpoint_interval_steps: int
    estimated_parameter_vram_mb: float
    estimated_optimizer_vram_mb: float
    estimated_activation_vram_mb: float
    estimated_total_vram_mb: float
    device_type: str
    available_gpu_memory_mb: float | None
    estimated_sequence_tokens: int
    estimated_audio_tokens: int
    estimated_image_tokens: int
    gradient_checkpointing: bool
    precision: str
    bytes_per_element: int
    recommended_batch_size: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the scale plan to a plain dictionary."""
        return asdict(self)


def build_training_scale_plan(
    *,
    dataset_size: int,
    settings: TrainingSettings,
    model_settings: ModelSettings,
    device: torch.device,
) -> TrainingScalePlan:
    """Estimate training steps, checkpoint cadence, and memory pressure."""

    parameter_count = _estimate_parameter_count(model_settings)
    batch_size = _effective_batch_size(
        configured_batch_size=int(settings.batch_size),
        dataset_size=dataset_size,
    )
    accumulation = int(settings.gradient_accumulation_steps)
    effective_batch_size = batch_size * accumulation
    micro_batches_per_epoch = max(1, math.ceil(dataset_size / batch_size))
    optimizer_steps_per_epoch = max(
        1,
        math.ceil(micro_batches_per_epoch / accumulation),
    )
    estimated_steps = optimizer_steps_per_epoch * int(settings.epochs)
    checkpoint_interval_steps = optimizer_steps_per_epoch

    precision = precision_from_settings(settings)
    element_bytes = bytes_per_element(precision)
    parameter_vram_mb = parameter_count * element_bytes / 1_000_000
    optimizer_vram_mb = parameter_count * 8 / 1_000_000
    activation_vram_mb = _estimate_activation_vram_mb(
        batch_size=batch_size,
        model_settings=model_settings,
        bytes_per_value=element_bytes,
    )
    available_gpu_memory_mb = _available_gpu_memory_mb(device=device)
    recommended_batch_size = _recommended_batch_size(
        batch_size=batch_size,
        estimated_total_vram_mb=(
            parameter_vram_mb + optimizer_vram_mb + activation_vram_mb
        ),
        available_gpu_memory_mb=available_gpu_memory_mb,
    )
    total_vram_mb = parameter_vram_mb + optimizer_vram_mb + activation_vram_mb

    return TrainingScalePlan(
        dataset_size=dataset_size,
        model_parameters=parameter_count,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        effective_batch_size=effective_batch_size,
        epochs=int(settings.epochs),
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        estimated_steps=estimated_steps,
        checkpoint_interval_steps=checkpoint_interval_steps,
        estimated_parameter_vram_mb=round(parameter_vram_mb, 2),
        estimated_optimizer_vram_mb=round(optimizer_vram_mb, 2),
        estimated_activation_vram_mb=round(activation_vram_mb, 2),
        estimated_total_vram_mb=round(total_vram_mb, 2),
        device_type=device.type,
        available_gpu_memory_mb=(
            round(available_gpu_memory_mb, 2)
            if available_gpu_memory_mb is not None
            else None
        ),
        estimated_sequence_tokens=(
            batch_size * int(model_settings.raw_text_max_tokens)
        ),
        estimated_audio_tokens=_estimate_audio_tokens(
            batch_size=batch_size,
            model_settings=model_settings,
        ),
        estimated_image_tokens=_estimate_image_tokens(
            batch_size=batch_size,
            model_settings=model_settings,
        ),
        gradient_checkpointing=bool(model_settings.gradient_checkpointing),
        precision=precision,
        bytes_per_element=element_bytes,
        recommended_batch_size=recommended_batch_size,
    )


def _estimate_parameter_count(model_settings: ModelSettings) -> int:
    hidden = int(model_settings.fusion_dim)
    text_vocab = int(model_settings.raw_text_vocab_size)
    text_params = (
        text_vocab * hidden + hidden * hidden * 8
        if "text" in model_settings.enabled_modalities
        else 0
    )
    vision_params = (
        hidden * hidden * 6
        if "image" in model_settings.enabled_modalities
        else 0
    )
    document_params = (
        text_vocab * hidden + hidden * hidden * 10
        if "document" in model_settings.enabled_modalities
        else 0
    )
    audio_params = (
        hidden * hidden * 4
        if "audio" in model_settings.enabled_modalities
        else 0
    )
    video_params = (
        hidden * hidden * 8
        if "video" in model_settings.enabled_modalities
        else 0
    )
    fusion_params = hidden * hidden * 4
    classifier_params = hidden * int(model_settings.num_classes)
    projection_params = hidden * int(model_settings.projection_dim)
    decoder_params = 0
    if model_settings.text_decoder.enabled:
        decoder_hidden = int(model_settings.text_decoder.hidden_dim)
        decoder_vocab = int(model_settings.text_decoder.vocab_size)
        decoder_params = (
            decoder_vocab * decoder_hidden
            + decoder_hidden * decoder_hidden * 12
        )
    generator_params = 0
    if model_settings.image_generator.enabled:
        generator_params += (
            int(model_settings.image_codec.latent_channels) * hidden * 4
        )
    if model_settings.vocoder.enabled:
        generator_params += hidden * hidden * 4
    return int(
        text_params
        + document_params
        + vision_params
        + audio_params
        + video_params
        + fusion_params
        + classifier_params
        + projection_params
        + decoder_params
        + generator_params
    )


def _effective_batch_size(
    *,
    configured_batch_size: int,
    dataset_size: int,
) -> int:
    if dataset_size <= 0:
        return max(1, configured_batch_size)
    return max(1, min(configured_batch_size, dataset_size))


def _estimate_activation_vram_mb(
    *,
    batch_size: int,
    model_settings: ModelSettings,
    bytes_per_value: int,
) -> float:
    raw_values = (
        (
            batch_size * int(model_settings.raw_text_max_tokens)
            if "text" in model_settings.enabled_modalities
            else 0
        )
        + (
            batch_size * int(model_settings.raw_text_max_tokens)
            if "document" in model_settings.enabled_modalities
            else 0
        )
        + (
            batch_size * 3 * int(model_settings.raw_image_size) ** 2
            if "image" in model_settings.enabled_modalities
            else 0
        )
        + (
            batch_size * int(model_settings.raw_audio_num_samples)
            if "audio" in model_settings.enabled_modalities
            else 0
        )
        + (
            batch_size
            * int(model_settings.raw_video_frames)
            * 3
            * int(model_settings.raw_image_size) ** 2
            if "video" in model_settings.enabled_modalities
            else 0
        )
    )
    hidden_values = batch_size * int(model_settings.fusion_dim) * 32
    checkpoint_factor = 0.65 if model_settings.gradient_checkpointing else 1.0
    return (
        (raw_values + hidden_values)
        * bytes_per_value
        * checkpoint_factor
        / 1_000_000
    )


def _available_gpu_memory_mb(*, device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        target_device = device.index if device.index is not None else None
        free_bytes, _total_bytes = torch.cuda.mem_get_info(target_device)
    except (RuntimeError, OSError):
        return None
    return free_bytes / 1_000_000


def _recommended_batch_size(
    *,
    batch_size: int,
    estimated_total_vram_mb: float,
    available_gpu_memory_mb: float | None,
) -> int:
    if available_gpu_memory_mb is None or estimated_total_vram_mb <= 0:
        return batch_size
    if estimated_total_vram_mb < available_gpu_memory_mb * 0.85:
        return batch_size
    ratio = max(
        0.1, (available_gpu_memory_mb * 0.85) / estimated_total_vram_mb
    )
    return max(1, int(batch_size * ratio))


def _estimate_audio_tokens(
    *,
    batch_size: int,
    model_settings: ModelSettings,
) -> int:
    frame_samples = (
        model_settings.audio_tokenizer.sample_rate
        * model_settings.audio_tokenizer.hop_ms
        // 1000
    )
    return batch_size * max(
        1,
        int(model_settings.raw_audio_num_samples) // max(1, frame_samples),
    )


def _estimate_image_tokens(
    *,
    batch_size: int,
    model_settings: ModelSettings,
) -> int:
    patch_size = 16
    image_size = int(model_settings.raw_image_size)
    per_image = max(1, image_size // patch_size) ** 2
    video = per_image * int(model_settings.raw_video_frames)
    total = 0
    if "image" in model_settings.enabled_modalities:
        total += batch_size * per_image
    if "video" in model_settings.enabled_modalities:
        total += batch_size * video
    return total


__all__ = [
    "StageExecutionArtifact",
    "TRAINING_STAGE_ORDER",
    "TrainingScalePlan",
    "TrainingStage",
    "TrainingStageExecutionError",
    "TrainingStageExecutor",
    "build_training_scale_plan",
    "stage_index",
    "stages_between",
]
