"""Validate release task contracts against evaluator and dataset capabilities.

This module owns the cross-component validation that requires both evaluator
and dataset capabilities. It depends on:
  - evaluator.metric_registry (supported evaluation methods)
  - mmcrawler_datasets.schema (supported target fields)
  - multimodal.tasks.registry (task catalog and metadata)

This is a RELEASE/ORCHESTRATION concern, not a core multimodal concern.
"""

from __future__ import annotations

from typing import Callable

from config.multimodal.model_settings import ModelSettings
from config.releases.release_requirements import (
    ReleaseRequirements,
    RequiredTaskEvidence,
    TaskImplementationEvidence,
    required_task_evidence,
)
from config.settings.root import Settings
from evaluator.metric_registry import SUPPORTED_EVALUATION_METHODS
from mmcrawler_datasets.schema import SUPPORTED_TARGET_FIELDS
from multimodal.tasks.contracts import MODEL_OUTPUT_MODALITIES
from multimodal.tasks.registry import (
    COLLATION_SUPPORTED_TASKS,
    SAMPLE_BUILDER_TASKS,
    TASKS,
    resolved_output_modalities,
    task_has_trainable_loss,
)
from schemas.multimodal_tasks import (
    canonical_task_mapping,
    canonical_task_name,
    canonical_task_names,
    normalized_modalities,
)

_TASK_GENERATION_OUTPUT_MODALITIES = frozenset({"text", "json", "code"})
_NON_GENERATIVE_OUTPUT_MODALITIES = frozenset({"class", "embedding"})
_UNSUPPORTED_MATURITY = frozenset({"experimental", "disabled"})

# Production-blocked capabilities mapped to their settings activation
# switches. Capabilities without an activation mapping are documented
# out-of-scope declarations that cannot be activated by settings.
_BLOCKED_CAPABILITY_ACTIVATIONS: dict[str, Callable[[ModelSettings], bool]] = {
    "audio_generation": lambda model: (
        model.audio_tokenizer.enabled or model.vocoder.enabled
    ),
    "continuous_audio_tokens": lambda model: (
        model.audio_tokenizer.enabled
        and model.audio_tokenizer.codec == "continuous"
    ),
    "image_generation": lambda model: (
        model.image_generator.enabled or model.image_decoder.enabled
    ),
    "video_generation": lambda model: (
        model.video_generator.enabled or model.video_decoder.enabled
    ),
}


def validate_release_task_contracts(
    settings: Settings,
    release_requirements: ReleaseRequirements,
) -> None:
    """Validate all release task implementation contracts.

    This combines:
    - Implementation evidence coverage (sample builder, collation, model outputs, loss, evaluation, inference)
    - Blocked capability activation checks
    - Evaluation method coverage against evaluator capabilities

    Args:
        settings: The full runtime settings.
        release_requirements: The release requirements to validate against.

    Raises:
        ValueError: If any release contract is violated.
    """
    _validate_release_task_implementation_coverage(
        settings,
        release_requirements,
    )
    _validate_release_generation_contracts(
        settings,
        release_requirements,
    )
    _validate_release_evaluation_coverage(
        settings,
        release_requirements,
    )


def _validate_release_task_implementation_coverage(
    settings: Settings,
    release_requirements: ReleaseRequirements,
) -> None:
    """Fail closed when a required release task lacks implementation evidence."""

    evidence_requirements = {
        required.task_name: required
        for required in required_task_evidence(release_requirements)
    }
    for task_name in sorted(release_requirements.required_tasks):
        evidence = _task_implementation_evidence(settings, task_name)
        required = evidence_requirements.get(task_name)
        if required is None:
            required = RequiredTaskEvidence(task_name=task_name)
        missing = evidence.missing_requirements(required)
        if missing:
            raise ValueError(
                f"release task {task_name!r} lacks required implementation "
                f"evidence: {', '.join(missing)}"
            )


def _validate_release_generation_contracts(
    settings: Settings,
    release_requirements: ReleaseRequirements,
) -> None:
    """Fail closed when a blocked capability is activated or a required
    generative task has no decoding path."""

    model = settings.multimodal

    for capability in release_requirements.blocked_capabilities:
        if capability.production_status != "blocked":
            continue
        activation = _BLOCKED_CAPABILITY_ACTIVATIONS.get(capability.capability)
        if activation is not None and activation(model):
            raise ValueError(
                f"blocked capability {capability.capability!r} is activated "
                "in production settings"
            )

    enabled_tasks = canonical_task_names(
        settings.training.tasks,
        field_name="multimodal.training.tasks",
    )
    for task_name in sorted(release_requirements.required_tasks):
        if task_name not in enabled_tasks:
            continue
        outputs = _resolved_task_output_modalities(settings, task_name)
        if (
            _TASK_GENERATION_OUTPUT_MODALITIES.intersection(outputs)
            and not model.text_decoder.enabled
        ):
            raise ValueError(
                f"release task {task_name!r} generates text but "
                "multimodal.model.text_decoder.enabled=false"
            )


def _validate_release_evaluation_coverage(
    settings: Settings,
    release_requirements: ReleaseRequirements,
) -> None:
    """Fail closed when an enabled production task declares an evaluation
    method the evaluation stack cannot serve."""

    supported = SUPPORTED_EVALUATION_METHODS
    enabled_tasks = canonical_task_names(
        settings.training.tasks,
        field_name="multimodal.training.tasks",
    )
    for task_name in sorted(enabled_tasks):
        definition = TASKS.get(canonical_task_name(task_name))
        if definition is None:
            continue
        method = str(definition.evaluation_method or "").strip().lower()
        if method not in supported:
            raise ValueError(
                f"release task {task_name!r} evaluation method {method!r} "
                "is not supported by the evaluator"
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


def _task_implementation_evidence(
    settings: Settings,
    task_name: str,
) -> TaskImplementationEvidence:
    """Collect observed implementation evidence for one task."""

    canonical_name = canonical_task_name(task_name)
    definition = TASKS.get(canonical_name)
    outputs = _resolved_task_output_modalities(settings, canonical_name)
    output_set = set(outputs)

    if definition is not None:
        required_target_fields = tuple(definition.required_target_fields)
        maturity = str(definition.maturity).strip().lower()
    else:
        required_target_fields = ()
        maturity = ""

    evaluation_method = (
        str(definition.evaluation_method or "").strip().lower()
        if definition is not None
        else ""
    )

    if output_set <= _NON_GENERATIVE_OUTPUT_MODALITIES:
        inference_supported = True
    elif bool(_TASK_GENERATION_OUTPUT_MODALITIES.intersection(output_set)):
        inference_supported = settings.multimodal.text_decoder.enabled
    else:
        inference_supported = False

    return TaskImplementationEvidence(
        task_name=canonical_name,
        has_definition=definition is not None,
        has_sample_builder=(canonical_name in SAMPLE_BUILDER_TASKS),
        has_target_fields=(
            all(
                field in SUPPORTED_TARGET_FIELDS
                for field in required_target_fields
            )
        ),
        has_collation=(canonical_name in COLLATION_SUPPORTED_TASKS),
        has_dataset_coverage=(
            settings.training.min_task_samples.get(canonical_name, 0) > 0
        ),
        has_model_output=(
            bool(output_set) and output_set <= MODEL_OUTPUT_MODALITIES
        ),
        has_loss=task_has_trainable_loss(canonical_name),
        has_metric=(
            bool(evaluation_method)
            and evaluation_method in SUPPORTED_EVALUATION_METHODS
        ),
        has_inference=inference_supported,
        maturity_sufficient=maturity not in _UNSUPPORTED_MATURITY,
    )
