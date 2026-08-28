"""Config-owned multimodal shape and consistency rules."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from schemas.multimodal_tasks import canonical_task_mapping, canonical_task_names

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
