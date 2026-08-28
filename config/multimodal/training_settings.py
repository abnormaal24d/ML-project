"""Training and dataloader settings for multimodal training."""

from __future__ import annotations

import re
from typing import Literal, TypeVar

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel
from schemas.multimodal_tasks import canonical_task_name

_TaskMapValue = TypeVar("_TaskMapValue", int, float)

SUPPORTED_TRAINING_BACKENDS: frozenset[str] = frozenset(
    {"pipeline_smoke", "dense_transformer"}
)

_DEFAULT_TEXT_SPECIAL_TOKEN_IDS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<mask>": 4,
    "<image>": 5,
    "<audio>": 6,
    "<video>": 7,
    "<doc>": 8,
    "<user>": 9,
    "<assistant>": 10,
    "<system>": 11,
    "<tool>": 12,
}


class TrainingSettings(SettingsModel):
    """Validated multimodal training and dataloader settings."""

    run_mode: Literal["smoke", "full"] = "full"
    release_stage: Literal[
        "pipeline_smoke",
        "learning_candidate",
        "candidate",
        "production_model",
    ] = "pipeline_smoke"
    training_backend: Literal["pipeline_smoke", "dense_transformer"] = (
        "pipeline_smoke"
    )
    training_stage: Literal[
        "DATASET_FREEZE",
        "TOKENIZER_BUILD",
        "MODALITY_PRETRAIN",
        "CROSS_MODAL_ALIGNMENT",
        "MULTIMODAL_PRETRAIN",
        "INSTRUCTION_TUNING",
        "PREFERENCE_TUNING",
        "SAFETY_TUNING",
        "BENCHMARK",
        "ACCEPTANCE",
        "PROMOTION",
    ] = "MULTIMODAL_PRETRAIN"
    preference_loss: Literal["pairwise", "dpo"] = "dpo"
    preference_beta: float = Field(default=0.1, gt=0.0)
    reference_free_preference: bool = True
    safety_loss_weight: float = Field(default=1.0, ge=0.0)
    text_tokenizer_backend: Literal["subword"] = "subword"
    text_tokenizer_name: str = "repo_subword"
    text_tokenizer_max_tokens: int = Field(default=128, gt=0)
    text_tokenizer_path: str = "artifacts/tokenizer/tokenizer.json"
    text_tokenizer_sha256: str | None = None
    text_tokenizer_artifact_version: str = "byte_bpe_v2"
    text_tokenizer_vocab_size: int = Field(default=4096, gt=128)
    text_tokenizer_special_tokens: dict[str, int] = Field(
        default_factory=lambda: dict(_DEFAULT_TEXT_SPECIAL_TOKEN_IDS)
    )

    job_status_replace_retry_attempts: int = Field(default=3, ge=1)
    job_status_replace_retry_delay_seconds: float = Field(default=0.05, ge=0.0)

    tasks: tuple[str, ...] = ()
    approved_beta_tasks: tuple[str, ...] = ()
    sensitive_task_approvals: tuple[str, ...] = ()
    task_sampling_weights: dict[str, float] = Field(default_factory=dict)
    task_family_sampling_weights: dict[str, float] = Field(default_factory=dict)
    min_task_samples: dict[str, int] = Field(default_factory=dict)
    disable_undercovered_tasks: bool = True
    drop_samples_with_invalid_targets: bool = True
    dynamic_sampling: bool = True
    task_aware_batching: bool = True
    curriculum_schedule: tuple[str, ...] = ()
    modality_dropout: dict[str, float] = Field(default_factory=dict)

    batch_size: int = Field(default=64, gt=0)
    epochs: int = Field(default=5, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    lr_scheduler: Literal["none", "cosine"] = "cosine"
    scheduler_interval: Literal["step", "epoch"] = "step"
    min_learning_rate: float = Field(default=0.0, ge=0.0)
    gradient_clip_max_norm: float | None = Field(default=1.0, gt=0.0)
    monitor_metric: Literal["validation_loss"] = "validation_loss"
    monitor_mode: Literal["min", "max"] = "min"
    early_stopping_patience: int | None = Field(default=2, ge=1)
    early_stopping_min_delta: float = Field(default=0.0001, ge=0.0)

    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = Field(default=None, gt=0)
    drop_last: bool = False
    shuffle_training_split: bool = True

    device: Literal["cpu", "cuda", "auto"] = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    distributed_strategy: Literal["none", "ddp", "fsdp", "auto"] = "auto"

    export_artifacts: bool = False
    export_directory: str = "models"
    training_manifest_filename: str = "training_manifest.json"

    progress_log_interval_batches: int = Field(default=10, ge=0)

    feature_cache_directory: str = "features"
    materialized_tensors_enabled: bool = True
    materialized_tensor_directory: str = "training_tensors"
    materialized_tensor_version: str = "raw_tensors_v1"
    materialized_tensor_validate_shapes: bool = True
    resume_from_checkpoint: str | None = None
    max_samples: int | None = Field(default=None, gt=0)
    min_split_items: int = Field(default=1, ge=0)
    min_split_items_by_split: dict[str, int] = Field(default_factory=dict)

    seed: int = Field(default=42, ge=0)
    deterministic: bool = False
    offline: bool = True

    min_alignment_score: float = Field(default=0.3, ge=0.0, le=1.0)
    alignment_loss_power: float = Field(default=1.0, ge=0.0)
    mlm_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    mlm_loss_weight: float = Field(default=0.25, ge=0.0)
    language_modeling_loss_weight: float = Field(default=0.25, ge=0.0)
    ocr_sequence_loss_weight: float = Field(default=0.0, ge=0.0)
    audio_token_loss_weight: float = Field(default=0.0, ge=0.0)
    image_generation_loss_weight: float = Field(default=0.0, ge=0.0)
    video_generation_loss_weight: float = Field(default=0.0, ge=0.0)
    image_mask_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    image_patch_loss_weight: float = Field(default=0.0, ge=0.0)
    audio_mask_probability: float = Field(default=0.15, ge=0.0, le=1.0)
    audio_masked_loss_weight: float = Field(default=0.0, ge=0.0)
    video_temporal_loss_weight: float = Field(default=0.0, ge=0.0)
    hard_negative_loss_weight: float = Field(default=0.0, ge=0.0)
    hard_negative_margin: float = Field(default=0.2, ge=0.0)
    use_hard_negative_sampler: bool = False

    @field_validator(
        "tasks",
        "curriculum_schedule",
        "approved_beta_tasks",
        "sensitive_task_approvals",
    )
    @classmethod
    def _normalize_task_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for item in value:
            name = canonical_task_name(item)
            if name in normalized:
                raise ValueError(f"duplicate canonical task name: {name!r}")
            normalized.append(name)
        return tuple(normalized)

    @field_validator("text_tokenizer_sha256")
    @classmethod
    def _normalize_text_tokenizer_sha256(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise ValueError(
                "text_tokenizer_sha256 must be a 64-character SHA-256 hex "
                "digest"
            )
        return normalized

    @field_validator("text_tokenizer_special_tokens")
    @classmethod
    def _validate_text_tokenizer_special_tokens(
        cls,
        value: dict[str, int],
    ) -> dict[str, int]:
        required = {"<pad>", "<unk>", "<bos>", "<eos>", "<mask>"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(
                "text_tokenizer_special_tokens is missing required tokens: "
                f"{missing}"
            )
        ids = [int(token_id) for token_id in value.values()]
        if any(token_id < 0 for token_id in ids):
            raise ValueError(
                "text_tokenizer_special_tokens ids must be non-negative"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(
                "text_tokenizer_special_tokens ids must be unique"
            )
        return {str(token): int(token_id) for token, token_id in value.items()}

    @field_validator("task_sampling_weights", "min_task_samples")
    @classmethod
    def _normalize_task_keyed_maps(
        cls,
        value: dict[str, _TaskMapValue],
    ) -> dict[str, _TaskMapValue]:
        normalized: dict[str, _TaskMapValue] = {}
        for task_name, amount in value.items():
            canonical = canonical_task_name(task_name)
            if canonical in normalized:
                raise ValueError(
                    "training task map contains duplicate canonical task "
                    f"key: {canonical!r}"
                )
            normalized[canonical] = amount
        return normalized

    @model_validator(mode="after")
    def _validate_worker_settings(self) -> TrainingSettings:
        if self.release_stage in {"candidate", "production_model"} and not self.offline:
            raise ValueError(f"{self.release_stage} requires offline=true")
        if (
            self.release_stage == "production_model"
            and self.training_backend == "pipeline_smoke"
        ):
            raise ValueError(
                "production_model cannot use training_backend='pipeline_smoke'"
            )
        self._validate_worker_count()
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        enabled = {canonical_task_name(task) for task in self.tasks}
        inactive_sampling = sorted(
            task_name
            for task_name, weight in self.task_sampling_weights.items()
            if float(weight) > 0.0 and task_name not in enabled
        )
        if inactive_sampling:
            raise ValueError(
                "task_sampling_weights contains positive weights for disabled "
                f"tasks: {inactive_sampling}"
            )

        inactive_minimums = sorted(
            task_name
            for task_name, minimum in self.min_task_samples.items()
            if int(minimum) > 0 and task_name not in enabled
        )
        if inactive_minimums:
            raise ValueError(
                "min_task_samples contains positive minima for disabled tasks: "
                f"{inactive_minimums}"
            )

        curriculum_extras = sorted(
            canonical_task_name(task)
            for task in self.curriculum_schedule
            if canonical_task_name(task) not in enabled
        )
        if curriculum_extras:
            raise ValueError(
                "curriculum_schedule contains tasks not in tasks: "
                f"{curriculum_extras}"
            )
        return self

    def _validate_worker_count(self) -> None:
        """Dataloader / CPU worker count and prefetch/persistent rules."""

        if self.num_workers == 0:
            if self.prefetch_factor is not None:
                raise ValueError(
                    "prefetch_factor must be null when num_workers=0"
                )
            if self.persistent_workers:
                raise ValueError(
                    "persistent_workers must be false when num_workers=0"
                )
        elif self.prefetch_factor is None:
            raise ValueError("prefetch_factor must be set when num_workers > 0")

    def effective_min_task_samples(self) -> dict[str, int]:
        """Return task minima that can apply to the active task set."""

        active_tasks = {canonical_task_name(task) for task in self.tasks}
        if not active_tasks:
            return {
                canonical_task_name(task_name): int(minimum)
                for task_name, minimum in self.min_task_samples.items()
            }
        return {
            canonical_task_name(task_name): int(minimum)
            for task_name, minimum in self.min_task_samples.items()
            if canonical_task_name(task_name) in active_tasks
        }


def resolve_objective_loss_weights(
    settings: TrainingSettings,
) -> dict[str, float]:
    """Resolve the complete concrete loss-weight contract for one run."""

    return {
        "audio_generation": settings.audio_token_loss_weight,
        "audio_reconstruction": settings.audio_masked_loss_weight,
        "contrastive": 1.0,
        "hard_negative": settings.hard_negative_loss_weight,
        "image_generation": settings.image_generation_loss_weight,
        "image_reconstruction": settings.image_patch_loss_weight,
        "label": 1.0,
        "language_modeling": settings.language_modeling_loss_weight,
        "ocr_sequence": settings.ocr_sequence_loss_weight,
        "preference": 1.0,
        "safety": settings.safety_loss_weight,
        "sequence": settings.mlm_loss_weight,
        "text_mlm": settings.mlm_loss_weight,
        "video_generation": settings.video_generation_loss_weight,
        "video_temporal": settings.video_temporal_loss_weight,
    }
