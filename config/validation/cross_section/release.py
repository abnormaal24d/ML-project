"""Release-stage and production training rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from schemas.multimodal_tasks import canonical_task_mapping, canonical_task_names

if TYPE_CHECKING:
    from config.settings.root import Settings

_PRODUCTION_RELEASE_STAGES = frozenset({"candidate", "production_model"})


def _validate_release_stage(settings: Settings) -> None:
    """Enforce production artifact requirements before promotion."""

    if settings.training.release_stage not in _PRODUCTION_RELEASE_STAGES:
        return

    validator = settings.datasets.training.dataset_validator
    missing_flags: list[str] = []
    if not validator.require_dataset_card:
        missing_flags.append("require_dataset_card")
    if not validator.require_model_card:
        missing_flags.append("require_model_card")
    if missing_flags:
        raise ValueError(
            "production release stages require these dataset-validator flags "
            f"to be true: {missing_flags}"
        )


def _validate_production_configuration_guarantees(settings: Settings) -> None:
    """Reject production scope that lacks training and release guarantees."""

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
    uncovered = sorted(
        task_name
        for task_name in enabled_tasks
        if int(training_minimums.get(task_name, 0)) <= 0
    )
    if uncovered:
        raise ValueError(
            "production tasks require positive training sample minimums: "
            f"{uncovered}"
        )
    if training.disable_undercovered_tasks:
        raise ValueError(
            "production release stages must fail on undercovered tasks; "
            "disable_undercovered_tasks must be false"
        )

    validator = settings.datasets.training.dataset_validator
    if not validator.require_generation_targets:
        raise ValueError(
            "production release stages require generation targets for enabled output "
            "modalities"
        )

    model = settings.multimodal
    release = settings.release
    capacity_minimums = {
        "fusion_dim": (model.fusion_dim, release.min_model_fusion_dim),
        "projection_dim": (
            model.projection_dim,
            release.min_model_projection_dim,
        ),
        "raw_text_vocab_size": (
            model.raw_text_vocab_size,
            release.min_model_raw_text_vocab_size,
        ),
        "raw_text_max_tokens": (
            model.raw_text_max_tokens,
            release.min_model_raw_text_max_tokens,
        ),
        "raw_image_size": (
            model.raw_image_size,
            release.min_model_raw_image_size,
        ),
        "raw_audio_num_samples": (
            model.raw_audio_num_samples,
            release.min_model_raw_audio_num_samples,
        ),
        "raw_video_frames": (
            model.raw_video_frames,
            release.min_model_raw_video_frames,
        ),
    }
    undersized = {
        field_name: {"configured": int(configured), "minimum": int(minimum)}
        for field_name, (configured, minimum) in capacity_minimums.items()
        if int(configured) < int(minimum)
    }
    if undersized:
        raise ValueError(
            f"production model capacity is below release policy: {undersized}"
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
