"""Filesystem and task-schema validation for parsed samples."""

from __future__ import annotations

from pathlib import Path

from mmcrawler_datasets.schema import MultimodalSample
from multimodal.tasks.registry import get_task


def assert_object_paths_exist(
    *,
    sample: MultimodalSample,
    ref_path: Path,
    line_number: int,
) -> None:
    for modality_name, modality in (
        ("document", sample.document),
        ("image", sample.image),
        ("audio", sample.audio),
        ("video", sample.video),
    ):
        if modality is None or modality.path is None:
            continue
        if not Path(modality.path).is_file():
            raise FileNotFoundError(
                f"missing {modality_name} object path in {ref_path} "
                f"at line {line_number}: {modality.path}"
            )


def sample_has_modality(*, sample: MultimodalSample, modality: str) -> bool:
    if modality == "text":
        return sample.has_text
    if modality == "image":
        return sample.has_image
    if modality == "audio":
        return sample.has_audio
    if modality == "video":
        return sample.has_video
    if modality == "document":
        return sample.has_document
    if modality == "layout":
        return sample.has_layout
    if modality == "mask":
        return sample.has_mask
    if modality == "code":
        return sample.has_code
    if modality == "json":
        return sample.has_json
    return False


def sample_has_field(*, sample: MultimodalSample, field_name: str) -> bool:
    if field_name == "target_text":
        return bool(sample.target_text or sample.answer)
    value = getattr(sample, field_name, None)
    if value is None:
        value = sample.task_target.get(field_name)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (tuple, list, dict)):
        return bool(value)
    return value is not None


def validate_sample(*, sample: MultimodalSample) -> tuple[str, ...]:
    """Return task-schema validation errors for one sample."""

    definition = get_task(sample.task_type)
    if definition is None:
        return (f"unknown_task_type:{sample.task_type}",)

    errors: list[str] = []
    for modality in definition.required_input_modalities:
        if not sample_has_modality(sample=sample, modality=modality):
            errors.append(f"missing_required_modality:{modality}")
    if definition.evidence_modalities:
        present_evidence = sum(
            sample_has_modality(sample=sample, modality=modality)
            for modality in definition.evidence_modalities
        )
        if present_evidence < definition.min_evidence_modalities:
            errors.append(
                f"insufficient_evidence_modalities:"
                f"{present_evidence}/{definition.min_evidence_modalities}"
            )
    for field_name in definition.required_target_fields:
        if not sample_has_field(sample=sample, field_name=field_name):
            errors.append(f"missing_required_field:{field_name}")
    return tuple(errors)
