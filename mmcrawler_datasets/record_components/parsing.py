"""Training record validation and aggregate parsing."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mmcrawler_datasets.schema import ModalityObject, MultimodalSample
from mmcrawler_datasets.training_samples.targets import (
    conversation_turns_from_mapping,
)
from multimodal.tasks.registry import require_task
from schemas.versions import TRAINING_DATASET_SCHEMA_VERSION

from .annotations import parse_annotations
from .coercion import (
    optional_bool,
    optional_float,
    optional_int,
    optional_path,
    optional_string,
    require_mapping,
    require_string,
    resolve_path,
    str_tuple,
)

_SUPPORTED_MODALITIES = frozenset(
    {
        "text",
        "image",
        "audio",
        "video",
        "document",
    }
)
_TASK_TARGET_FIELDS = frozenset(
    {
        "alignment_score",
        "arousal",
        "answer",
        "answer_evidence_ids",
        "assistant_text",
        "audio_tensor_path",
        "background_noise_label",
        "chart_data",
        "code_language",
        "conversation_turns",
        "dominance",
        "edit_mask_path",
        "edit_mask_tensor_path",
        "emotion_label",
        "form_fields",
        "generator_id",
        "generator_version",
        "geometry_annotations",
        "humor_explanation",
        "image_tensor_path",
        "instruction",
        "layout_boxes",
        "math_expression",
        "math_solution",
        "negative_ids",
        "objective",
        "object_boxes",
        "output_modalities",
        "overlapping_speech",
        "positive_id",
        "prosody",
        "question",
        "sarcasm_label",
        "scene_graph",
        "source_image_path",
        "source_image_tensor_path",
        "sample_source",
        "speaker_label",
        "speaker_segments",
        "system_text",
        "target_audio_path",
        "target_audio_tokens_path",
        "target_code",
        "target_image_path",
        "target_image_tensor_path",
        "target_table_structure",
        "target_text",
        "target_video_path",
        "target_video_tensor_path",
        "target_video_tokens_path",
        "task_family",
        "task_type",
        "text_tokens_path",
        "tool_arguments_json",
        "tool_name",
        "tool_result_json",
        "ui_elements",
        "user_text",
        "valence",
        "verification_status",
        "video_tensor_path",
        "evidence_records",
        "reading_order",
        "ocr_confidences",
        "negative_verification",
        "chosen_response",
        "rejected_response",
        "preference_reason",
        "preference_source",
        "must_refuse",
        "requires_uncertainty",
        "requires_source_citation",
        "prompt_injection_present",
        "untrusted_document_instruction",
        "sensitive_data_present",
        "requires_tool_confirmation",
    }
)


def require_training_record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("training record must be a JSON object")

    schema_version = value.get("schema_version")
    if schema_version != TRAINING_DATASET_SCHEMA_VERSION:
        raise ValueError(
            "unsupported training schema: "
            f"expected {TRAINING_DATASET_SCHEMA_VERSION!r}, "
            f"received {schema_version!r}"
        )

    if not isinstance(value.get("task_target"), dict):
        raise ValueError("task_target must be a JSON object")

    if not isinstance(value.get("objects"), list):
        raise ValueError("objects must be a JSON array")

    misplaced = sorted(_TASK_TARGET_FIELDS.intersection(value))
    if misplaced:
        fields = ", ".join(misplaced)
        raise ValueError(
            f"task target fields must be nested under task_target: {fields}"
        )

    return value


def parse_text(record: Mapping[str, object]) -> str | None:
    """Return the canonical top-level text field only."""

    return optional_string(record.get("text"))


def parse_modality_objects(
    *,
    raw_objects: object,
    dataset_root: Path,
) -> dict[str, ModalityObject]:
    if not isinstance(raw_objects, list):
        raise ValueError("objects must be a JSON array")

    objects: dict[str, ModalityObject] = {}
    for raw_object in raw_objects:
        if not isinstance(raw_object, dict):
            raise ValueError("every objects[] item must be an object")

        role = require_string(raw_object, "role").lower()
        if role not in _SUPPORTED_MODALITIES:
            raise ValueError(f"unsupported object role: {role!r}")
        if role == "text":
            continue
        if role in objects:
            raise ValueError(f"duplicate objects[] role: {role!r}")

        objects[role] = _modality_object_from_payload(
            payload=raw_object,
            dataset_root=dataset_root,
        )
    return objects


def indexed_modality_signature(
    *,
    objects: list[object],
    has_text: bool,
) -> tuple[str, ...]:
    modalities: set[str] = {"text"} if has_text else set()

    for raw_object in objects:
        if not isinstance(raw_object, dict):
            raise ValueError("every objects[] item must be an object")

        role = require_string(raw_object, "role").lower()
        if role not in _SUPPORTED_MODALITIES:
            raise ValueError(f"unsupported object role: {role!r}")

        modalities.add(role)

    return tuple(sorted(modalities))


def parse_record(
    *,
    record: dict[str, object],
    dataset_root: Path,
) -> MultimodalSample:
    record = require_training_record(record)
    dataset_root = Path(dataset_root).resolve()

    sample_id = require_string(record, "sample_id")
    record_id = require_string(record, "record_id")
    task_target = require_mapping(record, "task_target")
    task_type = require_task(require_string(task_target, "task_type")).name

    objects = parse_modality_objects(
        raw_objects=record["objects"],
        dataset_root=dataset_root,
    )
    annotations = parse_annotations(task_target=task_target)
    text = parse_text(record)

    alignment = optional_float(task_target.get("alignment_score"))
    if alignment is None:
        alignment = 1.0
    else:
        alignment = max(0.0, min(1.0, alignment))

    return MultimodalSample(
        sample_id=sample_id,
        record_id=record_id,
        task_type=task_type,
        task_family=optional_string(task_target.get("task_family")),
        text=text,
        title=optional_string(record.get("title")),
        instruction=optional_string(task_target.get("instruction")),
        question=optional_string(task_target.get("question")),
        answer=optional_string(task_target.get("answer")),
        source_url=optional_string(record.get("source_url")),
        label=optional_int(record.get("label")),
        target_text=optional_string(task_target.get("target_text")),
        target_code=annotations.target_code,
        code_language=annotations.code_language,
        target_audio_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("target_audio_path"),
        ),
        target_audio_tokens_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("target_audio_tokens_path"),
        ),
        target_image_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("target_image_path"),
        ),
        target_video_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("target_video_path"),
        ),
        source_image_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("source_image_path"),
        ),
        edit_mask_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("edit_mask_path"),
        ),
        text_tokens_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("text_tokens_path"),
        ),
        image_tensor_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("image_tensor_path"),
        ),
        audio_tensor_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("audio_tensor_path"),
        ),
        video_tensor_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("video_tensor_path"),
        ),
        target_image_tensor_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("target_image_tensor_path"),
        ),
        target_video_tensor_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("target_video_tensor_path"),
        ),
        source_image_tensor_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("source_image_tensor_path"),
        ),
        edit_mask_tensor_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("edit_mask_tensor_path"),
        ),
        target_video_tokens_path=optional_path(
            dataset_root=dataset_root,
            value=task_target.get("target_video_tokens_path"),
        ),
        output_modalities=annotations.output_modalities,
        positive_id=optional_string(task_target.get("positive_id")),
        negative_ids=str_tuple(task_target.get("negative_ids")),
        alignment_score=alignment,
        system_text=optional_string(task_target.get("system_text")),
        user_text=optional_string(task_target.get("user_text")),
        assistant_text=optional_string(task_target.get("assistant_text")),
        conversation_turns=conversation_turns_from_mapping(
            task_target.get("conversation_turns")
        ),
        answer_evidence_ids=str_tuple(task_target.get("answer_evidence_ids")),
        tool_name=optional_string(task_target.get("tool_name")),
        tool_arguments_json=optional_string(
            task_target.get("tool_arguments_json")
        ),
        tool_result_json=optional_string(task_target.get("tool_result_json")),
        sample_source=optional_string(task_target.get("sample_source")),
        generator_id=optional_string(task_target.get("generator_id")),
        generator_version=optional_string(
            task_target.get("generator_version")
        ),
        verification_status=optional_string(
            task_target.get("verification_status")
        ),
        evidence_records=_dict_tuple(task_target.get("evidence_records")),
        reading_order=str_tuple(task_target.get("reading_order")),
        ocr_confidences=_float_tuple(task_target.get("ocr_confidences")),
        negative_verification=_dict_tuple(
            task_target.get("negative_verification")
        ),
        chosen_response=optional_string(task_target.get("chosen_response")),
        rejected_response=optional_string(
            task_target.get("rejected_response")
        ),
        preference_reason=optional_string(
            task_target.get("preference_reason")
        ),
        preference_source=optional_string(
            task_target.get("preference_source")
        ),
        must_refuse=optional_bool(task_target.get("must_refuse")),
        requires_uncertainty=optional_bool(
            task_target.get("requires_uncertainty")
        ),
        requires_source_citation=optional_bool(
            task_target.get("requires_source_citation")
        ),
        prompt_injection_present=optional_bool(
            task_target.get("prompt_injection_present")
        ),
        untrusted_document_instruction=optional_bool(
            task_target.get("untrusted_document_instruction")
        ),
        sensitive_data_present=optional_bool(
            task_target.get("sensitive_data_present")
        ),
        requires_tool_confirmation=optional_bool(
            task_target.get("requires_tool_confirmation")
        ),
        dataset_version=optional_string(record.get("dataset_version")),
        content_hash=optional_string(record.get("content_hash")),
        processing_version=optional_string(record.get("processing_version")),
        language=optional_string(record.get("language")),
        language_confidence=optional_float(record.get("language_confidence")),
        language_script=optional_string(record.get("language_script")),
        safety_status=optional_string(record.get("safety_status")),
        license=optional_string(record.get("license")),
        license_url=optional_string(record.get("license_url")),
        robots_status=optional_string(record.get("robots_status")),
        terms_source=optional_string(record.get("terms_source")),
        usage_rules=optional_string(record.get("usage_rules")),
        task_target=dict(task_target),
        document=objects.get("document"),
        image=objects.get("image"),
        audio=objects.get("audio"),
        video=objects.get("video"),
        layout_boxes=annotations.layout_boxes,
        ui_elements=annotations.ui_elements,
        geometry_annotations=annotations.geometry_annotations,
        object_boxes=annotations.object_boxes,
        chart_data=annotations.chart_data,
        math_expression=annotations.math_expression,
        math_solution=annotations.math_solution,
        humor_explanation=annotations.humor_explanation,
        target_table_structure=annotations.target_table_structure,
        form_fields=annotations.form_fields or {},
        scene_graph=annotations.scene_graph or {},
        speaker_segments=annotations.speaker_segments,
        prosody=annotations.prosody,
        emotion_label=annotations.emotion_label,
        arousal=annotations.arousal,
        valence=annotations.valence,
        dominance=annotations.dominance,
        sarcasm_label=annotations.sarcasm_label,
        speaker_label=annotations.speaker_label,
        background_noise_label=annotations.background_noise_label,
        overlapping_speech=annotations.overlapping_speech,
        metadata={
            **dict(record),
            "record_id": record_id,
        },
    )


def _dict_tuple(value: object) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a JSON array of objects")
    rows: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("expected a JSON array of objects")
        rows.append(dict(item))
    return tuple(rows)


def _float_tuple(value: object) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a JSON array of numbers")
    rows: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("expected a JSON array of numbers")
        try:
            rows.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ValueError("expected a JSON array of numbers") from exc
    return tuple(rows)


def _modality_object_from_payload(
    *,
    payload: Mapping[str, object],
    dataset_root: Path,
) -> ModalityObject:
    raw_path = optional_string(payload.get("object_path"))
    raw_url = optional_string(payload.get("object_url"))
    if raw_path is None and raw_url is None:
        raise ValueError("objects[] item requires object_path or object_url")
    path = (
        resolve_path(dataset_root=dataset_root, raw_path=raw_path)
        if raw_path is not None
        else None
    )
    canonical_fields = {
        "object_id",
        "object_path",
        "object_url",
        "object_mime_type",
        "byte_size",
        "role",
    }
    return ModalityObject(
        path=path,
        url=raw_url,
        mime_type=optional_string(payload.get("object_mime_type")),
        byte_size=optional_int(payload.get("byte_size")),
        metadata={
            str(key): value
            for key, value in payload.items()
            if key not in canonical_fields
        },
    )
