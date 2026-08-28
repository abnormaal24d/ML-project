"""Validate canonical runtime settings against the task catalog.

This module owns the core multimodal task and model capability validation.
Release-specific cross-component validation lives in:
  release.task_contract_validation
"""

from __future__ import annotations

from config.settings.root import Settings
from multimodal.tasks.registry import (
    TASKS,
    resolved_input_modalities,
    resolved_output_modalities,
)
from schemas.multimodal_tasks import (
    canonical_task_mapping,
    canonical_task_name,
    canonical_task_names,
    normalized_modalities,
)


def validate_multimodal_settings(settings: Settings) -> None:
    """Validate settings against the canonical multimodal task domain.

    This function validates core multimodal task and model capability contracts.
    Release-specific cross-component validation (implementation evidence,
    blocked capabilities, evaluation coverage) is owned by
    ``release.task_contract_validation.validate_release_task_contracts``
    and should be called separately by the release/orchestration layer.

    Args:
        settings: The full runtime settings.

    Raises:
        ValueError: If any core multimodal contract is violated.
    """

    _validate_known_tasks_and_families(settings)
    _validate_task_routing(settings)
    _validate_task_model_capabilities(settings)


def _validate_known_tasks_and_families(
    settings: Settings,
) -> None:
    """Reject task and family names absent from the task catalog."""

    training = settings.training
    enabled_tasks = canonical_task_names(
        training.tasks,
        field_name="multimodal.training.tasks",
    )
    beta_approvals = canonical_task_names(
        training.approved_beta_tasks,
        field_name="multimodal.training.approved_beta_tasks",
    )
    sensitive_approvals = canonical_task_names(
        training.sensitive_task_approvals,
        field_name="multimodal.training.sensitive_task_approvals",
    )
    referenced_tasks = enabled_tasks | beta_approvals | sensitive_approvals
    unknown_tasks = sorted(referenced_tasks - set(TASKS))
    if unknown_tasks:
        raise ValueError(
            f"configuration references unknown tasks: {unknown_tasks}"
        )

    configured_families = {
        str(family).strip().lower()
        for family in training.task_family_sampling_weights
    }
    known_families = {definition.family for definition in TASKS.values()}
    unknown_families = sorted(configured_families - known_families)
    if unknown_families:
        raise ValueError(
            "multimodal.training.task_family_sampling_weights references "
            f"unknown task families: {unknown_families}"
        )


def _validate_task_routing(settings: Settings) -> None:
    """Validate task routing overrides against registered task contracts."""

    routing = settings.multimodal.modality_routing
    enabled_tasks = canonical_task_names(
        settings.training.tasks,
        field_name="multimodal.training.tasks",
    )
    input_overrides = canonical_task_mapping(
        routing.task_input_overrides,
        field_name="multimodal.model.modality_routing.task_input_overrides",
    )
    output_overrides = canonical_task_mapping(
        routing.task_output_overrides,
        field_name="multimodal.model.modality_routing.task_output_overrides",
    )
    unknown_input_overrides = sorted(set(input_overrides) - set(TASKS))
    unknown_output_overrides = sorted(set(output_overrides) - set(TASKS))
    if unknown_input_overrides:
        raise ValueError(
            "modality_routing.task_input_overrides references unknown tasks: "
            f"{unknown_input_overrides}"
        )
    if unknown_output_overrides:
        raise ValueError(
            "modality_routing.task_output_overrides references unknown tasks: "
            f"{unknown_output_overrides}"
        )

    for task_name in sorted(enabled_tasks):
        inputs = resolved_input_modalities(
            task_name,
            overrides=input_overrides,
        )
        outputs = resolved_output_modalities(
            task_name,
            overrides=output_overrides,
        )
        if not inputs:
            raise ValueError(
                f"enabled task {task_name!r} has no input modalities"
            )
        if not outputs:
            raise ValueError(
                f"enabled task {task_name!r} has no output modalities"
            )


def _validate_task_model_capabilities(
    settings: Settings,
) -> None:
    """Ensure configured model capabilities cover enabled task semantics."""

    model = settings.multimodal

    enabled_tasks = canonical_task_names(
        settings.training.tasks,
        field_name="multimodal.training.tasks",
    )
    allowed_outputs = frozenset(
        normalized_modalities(
            model.output_modalities,
            field_name="multimodal.model.output_modalities",
        )
    )

    audio_output_tasks: list[str] = []
    for task_name in sorted(enabled_tasks):
        outputs = _resolved_task_output_modalities(settings, task_name)
        unsupported = sorted(set(outputs) - allowed_outputs)
        if unsupported:
            raise ValueError(
                f"task {task_name!r} output modalities {unsupported} "
                "must be a subset of "
                "multimodal.model.output_modalities="
                f"{sorted(allowed_outputs)}"
            )
        if "audio" in outputs:
            audio_output_tasks.append(task_name)

    if audio_output_tasks and not model.vocoder.enabled:
        raise ValueError(
            "enabled tasks with audio output require "
            "multimodal.model.vocoder.enabled=true: "
            f"{audio_output_tasks}"
        )

    # Every active audio-output task is generation in the current model: its
    # only trainable output path is the categorical audio-token head. Do not
    # special-case a task name here, otherwise another audio-output task can
    # silently bypass the discrete-token contract.
    audio_generation_tasks = tuple(sorted(audio_output_tasks))
    if audio_generation_tasks:
        audio_tokenizer = model.audio_tokenizer
        if not audio_tokenizer.enabled:
            raise ValueError(
                "audio generation tasks require "
                "multimodal.model.audio_tokenizer.enabled=true: "
                f"{list(audio_generation_tasks)}"
            )
        if audio_tokenizer.codec != "discrete":
            raise ValueError(
                "audio generation tasks require "
                "multimodal.model.audio_tokenizer.codec='discrete': "
                f"{list(audio_generation_tasks)}"
            )
        if audio_tokenizer.n_codebooks != 1:
            raise ValueError(
                "audio generation tasks require exactly one audio tokenizer "
                f"codebook: {list(audio_generation_tasks)}"
            )

    if "text_to_image" in enabled_tasks and (
        not model.image_generator.enabled or not model.image_decoder.enabled
    ):
        raise ValueError(
            "text_to_image requires image_generator and image_decoder "
            "to be enabled"
        )

    if "text_to_video" in enabled_tasks and (
        not model.video_generator.enabled or not model.video_decoder.enabled
    ):
        raise ValueError(
            "text_to_video requires video_generator and video_decoder "
            "to be enabled"
        )


def _resolved_task_output_modalities(
    settings: Settings,
    task_name: str,
) -> tuple[str, ...]:
    """Resolve output modalities from routing overrides or the registry."""

    canonical_name = canonical_task_name(task_name)
    overrides = canonical_task_mapping(
        settings.multimodal.modality_routing.task_output_overrides,
        field_name="multimodal.model.modality_routing.task_output_overrides",
    )
    return normalized_modalities(
        resolved_output_modalities(canonical_name, overrides=overrides),
        field_name=f"task_output_modalities.{canonical_name}",
    )
