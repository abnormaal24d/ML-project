"""Multimodal reasoning task definitions."""

from __future__ import annotations

from multimodal.tasks.contracts import TaskDefinition

MULTIMODAL_TASK_DEFINITIONS: tuple[TaskDefinition, ...] = (
    TaskDefinition(
        name="multimodal_evidence_qa",
        family="multimodal_reasoning",
        required_input_modalities=("text",),
        evidence_modalities=(
            "document",
            "image",
            "audio",
            "video",
        ),
        min_evidence_modalities=2,
        output_modalities=("text",),
        required_target_fields=("target_text",),
        evaluation_method="exact_match_f1",
        loss_key="language_modeling",
        sample_source="crawler_derived",
        maturity="experimental",
        optional_annotation_fields=(
            "evidence_modality_ids",
            "evidence_records",
            "modality_attribution",
        ),
        supports_hard_negatives=True,
        description=(
            "Answer a question using evidence from at least two "
            "different non-text modalities."
        ),
    ),
)
