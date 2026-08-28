"""Config-owned multimodal shape and consistency rules."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from config.environment.source_selection import PRODUCTION_ENVIRONMENTS
from schemas.multimodal_tasks import (
    canonical_task_mapping,
    canonical_task_names,
)

if TYPE_CHECKING:
    from config.settings.root import Settings


def _validate_multimodal_training_configuration_shape(
    settings: Settings,
) -> None:
    """Validate configuration structure without consulting the task catalog."""

    training = settings.training
    routing = settings.multimodal.modality_routing

    enabled_tasks = canonical_task_names(
        training.tasks,
        field_name="multimodal.training.tasks",
    )
    if training.run_mode == "full" and not enabled_tasks:
        raise ValueError(
            "multimodal.training.tasks must not be empty when run_mode='full'"
        )

    canonical_task_mapping(
        routing.task_input_overrides,
        field_name="multimodal.model.modality_routing.task_input_overrides",
    )
    canonical_task_mapping(
        routing.task_output_overrides,
        field_name="multimodal.model.modality_routing.task_output_overrides",
    )
    _validate_task_weight_maps(settings)


def _validate_task_weight_maps(settings: Settings) -> None:
    """Reject non-finite or negative task-related configuration weights."""

    training = settings.training
    for field_name, weights in (
        (
            "multimodal.training.task_sampling_weights",
            training.task_sampling_weights,
        ),
        (
            "multimodal.training.task_family_sampling_weights",
            training.task_family_sampling_weights,
        ),
    ):
        invalid = sorted(
            str(name)
            for name, value in weights.items()
            if not math.isfinite(float(value)) or float(value) < 0.0
        )
        if invalid:
            raise ValueError(
                f"{field_name} must contain finite non-negative weights: "
                f"{invalid}"
            )


def _validate_generation_loss_backends(settings: Settings) -> None:
    """Validate generation-loss dependencies declared by configuration."""

    model = settings.multimodal
    training = settings.training

    if training.image_generation_loss_weight > 0.0:
        if not model.image_generator.enabled:
            raise ValueError(
                "multimodal.training.image_generation_loss_weight > 0 "
                "requires multimodal.model.image_generator.enabled=true"
            )
        if not model.image_decoder.enabled:
            raise ValueError(
                "multimodal.training.image_generation_loss_weight > 0 "
                "requires multimodal.model.image_decoder.enabled=true"
            )

    if training.video_generation_loss_weight > 0.0:
        if not model.video_generator.enabled:
            raise ValueError(
                "multimodal.training.video_generation_loss_weight > 0 "
                "requires multimodal.model.video_generator.enabled=true"
            )
        if not model.video_decoder.enabled:
            raise ValueError(
                "multimodal.training.video_generation_loss_weight > 0 "
                "requires multimodal.model.video_decoder.enabled=true"
            )


def _validate_dataset_validator_task_alignment(
    settings: Settings,
) -> None:
    """Align dataset task minima with enabled training configuration tasks."""

    training = settings.training
    validator = settings.datasets.training.dataset_validator

    enabled_tasks = canonical_task_names(
        training.tasks,
        field_name="multimodal.training.tasks",
    )
    validator_minimums = canonical_task_mapping(
        validator.effective_min_task_samples(),
        field_name=(
            "datasets.training.dataset_validator.effective_min_task_samples"
        ),
    )

    inactive_minimums = sorted(
        task_name
        for task_name, minimum in validator_minimums.items()
        if int(minimum) > 0 and task_name not in enabled_tasks
    )
    if inactive_minimums:
        raise ValueError(
            "datasets.training.dataset_validator task minima contain "
            "positive values for disabled tasks: "
            f"{inactive_minimums}"
        )

    if settings.application.environment not in PRODUCTION_ENVIRONMENTS:
        return

    missing_minimums = sorted(enabled_tasks - set(validator_minimums))
    if missing_minimums:
        raise ValueError(
            "datasets.training.dataset_validator must define an explicit "
            "task minimum for every enabled production task: "
            f"{missing_minimums}"
        )
