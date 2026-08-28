"""Immutable multimodal task registry owned by the multimodal package.

This registry is intentionally self-contained to avoid import cycles when
other packages import multimodal.tasks. It mirrors the previous
orchestration-side registry but lives in the model package so callers
should import from multimodal.tasks.registry directly.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from multimodal.tasks.contracts import TaskDefinition
from multimodal.tasks.definitions.audio import AUDIO_TASK_DEFINITIONS
from multimodal.tasks.definitions.document import DOCUMENT_TASK_DEFINITIONS
from multimodal.tasks.definitions.image import IMAGE_TASK_DEFINITIONS
from multimodal.tasks.definitions.multimodal import MULTIMODAL_TASK_DEFINITIONS
from multimodal.tasks.definitions.text import TEXT_TASK_DEFINITIONS
from multimodal.tasks.definitions.video import VIDEO_TASK_DEFINITIONS
from schemas.multimodal_tasks import canonical_task_name

CAUSAL_TEXT_OBJECTIVES: frozenset[str] = frozenset(
    {
        "language_modeling",
        "ocr_sequence",
        "chart_reasoning",
        "math_reasoning",
        "speech_translation",
    }
)

GENERATION_OBJECTIVES: frozenset[str] = frozenset(
    {
        *CAUSAL_TEXT_OBJECTIVES,
        "audio_generation",
        "image_generation",
        "image_editing",
        "video_generation",
    }
)

TaskLossRequirements = tuple[frozenset[str], ...]

_PAIR_LOSS_REQUIREMENTS: TaskLossRequirements = (
    frozenset({"contrastive", "hard_negative"}),
)
_LANGUAGE_LOSS_REQUIREMENTS: TaskLossRequirements = (
    frozenset({"language_modeling", "sequence"}),
)
_OCR_LOSS_REQUIREMENTS: TaskLossRequirements = (
    frozenset({"ocr_sequence", "sequence"}),
)

OBJECTIVE_LOSS_REQUIREMENTS: Mapping[str, TaskLossRequirements] = (
    MappingProxyType(
        {
            "classification": (frozenset({"label"}),),
            "contrastive": _PAIR_LOSS_REQUIREMENTS,
            "language_modeling": _LANGUAGE_LOSS_REQUIREMENTS,
            "chart_reasoning": _LANGUAGE_LOSS_REQUIREMENTS,
            "math_reasoning": _LANGUAGE_LOSS_REQUIREMENTS,
            "ocr_sequence": _OCR_LOSS_REQUIREMENTS,
            "text_mlm": (frozenset({"text_mlm"}),),
            "audio_generation": (frozenset({"audio_generation"}),),
            "image_generation": (frozenset({"image_generation"}),),
            "video_generation": (frozenset({"video_generation"}),),
            "speech_translation": (
                frozenset({"language_modeling", "sequence"}),
                frozenset({"audio_generation"}),
            ),
        }
    )
)

# Settings validation consumes these implementation contracts without
# importing tensor or training runtime packages.
SAMPLE_BUILDER_TASKS: frozenset[str] = frozenset(
    {
        "audio_qa",
        "audio_text_pair",
        "cross_modal_consistency",
        "doc_qa",
        "document_text_pair",
        "image_captioning",
        "image_text_pair",
        "instruction_following",
        "multimodal_retrieval",
        "ocr_parse",
        "pdf_text_pair",
        "speech_to_audio",
        "speech_transcription",
        "text_pretrain",
        "causal_text_pretrain",
        "text_to_video",
        "video_captioning",
        "video_qa",
        "video_text_pair",
        "vqa",
    }
)

COLLATION_SUPPORTED_TASKS: frozenset[str] = frozenset(
    {
        "audio_qa",
        "audio_summarization",
        "audio_text_pair",
        "cross_modal_consistency",
        "doc_qa",
        "document_summarization",
        "document_text_pair",
        "image_captioning",
        "image_text_pair",
        "information_extraction",
        "instruction_following",
        "multimodal_evidence_qa",
        "multimodal_retrieval",
        "ocr_parse",
        "pdf_text_pair",
        "speech_transcription",
        "summarization",
        "text_pretrain",
        "causal_text_pretrain",
        "video_captioning",
        "video_qa",
        "video_summarization",
        "video_text_pair",
        "vqa",
    }
)

_ALL_TASK_DEFINITIONS = (
    *TEXT_TASK_DEFINITIONS,
    *IMAGE_TASK_DEFINITIONS,
    *AUDIO_TASK_DEFINITIONS,
    *VIDEO_TASK_DEFINITIONS,
    *DOCUMENT_TASK_DEFINITIONS,
    *MULTIMODAL_TASK_DEFINITIONS,
)


def _build_task_mapping(
    definitions: tuple[TaskDefinition, ...],
) -> dict[str, TaskDefinition]:
    tasks: dict[str, TaskDefinition] = {}
    for definition in definitions:
        if definition.name in tasks:
            raise ValueError(f"duplicate task definition: {definition.name!r}")
        tasks[definition.name] = definition
    return tasks


TASKS: Mapping[str, TaskDefinition] = MappingProxyType(
    _build_task_mapping(_ALL_TASK_DEFINITIONS)
)


def get_task(task_name: str | None) -> TaskDefinition | None:
    if task_name is None:
        return None
    try:
        return TASKS.get(canonical_task_name(task_name))
    except ValueError:
        return None


def require_task(task_name: str) -> TaskDefinition:
    definition = get_task(task_name)
    if definition is None:
        raise ValueError(f"unknown task: {task_name!r}")
    return definition


def resolved_input_modalities(
    task_name: str, *, overrides: Mapping[str, tuple[str, ...]] | None = None
) -> tuple[str, ...]:
    canonical = canonical_task_name(task_name)
    if overrides and canonical in overrides:
        return tuple(overrides[canonical])
    return require_task(canonical).required_input_modalities


def resolved_output_modalities(
    task_name: str, *, overrides: Mapping[str, tuple[str, ...]] | None = None
) -> tuple[str, ...]:
    canonical = canonical_task_name(task_name)
    if overrides and canonical in overrides:
        return tuple(overrides[canonical])
    return require_task(canonical).output_modalities


def task_family_for(task_type: str | None) -> str | None:
    definition = get_task(task_type)
    if definition is None:
        return None
    return definition.family


def task_requires_causal_decoder(task_name: str | None) -> bool:
    """Return whether a task is trained with causal decoder labels."""

    definition = get_task(task_name)
    return bool(
        definition is not None
        and definition.loss_key in CAUSAL_TEXT_OBJECTIVES
    )


def task_loss_requirements(
    task_name: str,
) -> TaskLossRequirements | None:
    """Return concrete trainable loss groups for one registered task."""

    definition = get_task(task_name)
    if definition is None:
        return None
    return OBJECTIVE_LOSS_REQUIREMENTS.get(definition.loss_key)


def task_has_trainable_loss(task_name: str) -> bool:
    """Return whether the task registry declares an executable objective."""

    return task_loss_requirements(task_name) is not None


def task_producible_loss_terms(task_name: str) -> frozenset[str]:
    """Return every concrete loss term the registered task can produce."""

    requirements = task_loss_requirements(task_name)
    if requirements is None:
        return frozenset()
    return frozenset(term for group in requirements for term in group)
