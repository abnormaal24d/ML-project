"""Release-stage and production training rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from schemas.multimodal_tasks import (
    canonical_task_mapping,
    canonical_task_names,
)

if TYPE_CHECKING:
    from config.settings.root import Settings

_PRODUCTION_RELEASE_STAGES = frozenset({"candidate", "production_model"})


def _validate_release_stage(
    settings: Settings,
) -> None:
    """Enforce production artifact requirements before promotion."""

    training = settings.training

    if training.release_stage not in _PRODUCTION_RELEASE_STAGES:
        return

    validator = settings.datasets.training.dataset_validator

    required_flags = (
        "require_dataset_card",
        "require_model_card",
    )

    missing_flags = [
        field_name
        for field_name in required_flags
        if not bool(getattr(validator, field_name))
    ]

    if missing_flags:
        raise ValueError(
            "production release stages require these dataset-validator flags "
            f"to be true: {missing_flags}"
        )


def _validate_production_configuration_guarantees(settings: Settings) -> None:
    """Reject production scope that lacks data or model guarantees."""

    training = settings.training
    if training.release_stage not in _PRODUCTION_RELEASE_STAGES:
        return

    enabled_tasks = canonical_task_names(
        training.tasks,
        field_name="multimodal.training.tasks",
    )
    training_minimums = canonical_task_mapping(
        training.effective_min_task_samples(),
        field_name="multimodal.training.min_task_samples",
    )
    validator = settings.datasets.training.dataset_validator
    validator_minimums = canonical_task_mapping(
        validator.effective_min_task_samples(),
        field_name="datasets.training.dataset_validator.min_task_samples",
    )
    uncovered = sorted(
        task_name
        for task_name in enabled_tasks
        if int(training_minimums.get(task_name, 0)) <= 0
        or int(validator_minimums.get(task_name, 0)) <= 0
    )
    if uncovered:
        raise ValueError(
            "production tasks require positive training and acceptance sample "
            f"minimums: {uncovered}"
        )
    if training.disable_undercovered_tasks:
        raise ValueError(
            "production release stages must fail on undercovered tasks; "
            "disable_undercovered_tasks must be false"
        )
    if not validator.require_generation_targets:
        raise ValueError(
            "production release stages require generation targets for enabled output "
            "modalities"
        )

    model = settings.multimodal
    capacity_minimums = {
        "fusion_dim": (model.fusion_dim, 512),
        "projection_dim": (model.projection_dim, 512),
        "raw_text_vocab_size": (model.raw_text_vocab_size, 32768),
        "raw_text_max_tokens": (model.raw_text_max_tokens, 512),
        "raw_image_size": (model.raw_image_size, 224),
        "raw_audio_num_samples": (model.raw_audio_num_samples, 160000),
        "raw_video_frames": (model.raw_video_frames, 8),
    }
    undersized = {
        field_name: {"configured": int(configured), "minimum": minimum}
        for field_name, (configured, minimum) in capacity_minimums.items()
        if int(configured) < minimum
    }
    if undersized:
        raise ValueError(
            f"production model capacity is below rules: {undersized}"
        )

    augmentation = settings.augmentation
    if augmentation.enabled:
        enabled_media_transforms = [
            modality
            for modality, enabled in (
                (
                    "document",
                    augmentation.document.enabled
                    and augmentation.document.mode == "document_media",
                ),
                ("image", augmentation.image.enabled),
                ("audio", augmentation.audio.enabled),
                ("video", augmentation.video.enabled),
            )
            if enabled
        ]
        if enabled_media_transforms:
            raise ValueError(
                "production release stages permit text-field augmentation only; "
                "media augmentation is enabled for: "
                f"{enabled_media_transforms}"
            )
