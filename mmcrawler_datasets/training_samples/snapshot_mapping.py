"""Map canonical training JSON rows to and from multimodal samples."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mmcrawler_datasets.record_components.parsing import (
    parse_record,
    require_training_record,
)
from mmcrawler_datasets.schema import ModalityObject, MultimodalSample
from schemas.versions import TRAINING_DATASET_SCHEMA_VERSION

_ALLOWED_MODALITIES = frozenset(
    {
        "text",
        "page",
        "document",
        "image",
        "audio",
        "video",
        "multimodal",
    }
)

_TOP_LEVEL_SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "record_id",
        "modality",
        "objects",
        "text",
        "title",
        "source_url",
        "label",
        "dataset_version",
        "content_hash",
        "processing_version",
        "language",
        "language_confidence",
        "language_script",
        "safety_status",
        "license",
        "license_url",
        "robots_status",
        "terms_source",
        "usage_rules",
        "task_target",
    }
)

_TASK_TARGET_SAMPLE_FIELDS = frozenset(
    {
        "task_type",
        "task_family",
        "instruction",
        "question",
        "answer",
        "target_text",
        "target_code",
        "code_language",
        "target_audio_path",
        "target_audio_tokens_path",
        "target_image_path",
        "target_video_path",
        "source_image_path",
        "edit_mask_path",
        "text_tokens_path",
        "image_tensor_path",
        "audio_tensor_path",
        "video_tensor_path",
        "target_image_tensor_path",
        "target_video_tensor_path",
        "source_image_tensor_path",
        "edit_mask_tensor_path",
        "target_video_tokens_path",
        "output_modalities",
        "positive_id",
        "negative_ids",
        "alignment_score",
        "system_text",
        "user_text",
        "assistant_text",
        "conversation_turns",
        "answer_evidence_ids",
        "tool_name",
        "tool_arguments_json",
        "tool_result_json",
        "sample_source",
        "generator_id",
        "generator_version",
        "verification_status",
        "layout_boxes",
        "ui_elements",
        "geometry_annotations",
        "object_boxes",
        "chart_data",
        "math_expression",
        "math_solution",
        "humor_explanation",
        "target_table_structure",
        "form_fields",
        "scene_graph",
        "speaker_segments",
        "prosody",
        "emotion_label",
        "arousal",
        "valence",
        "dominance",
        "sarcasm_label",
        "speaker_label",
        "background_noise_label",
        "overlapping_speech",
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

_PATH_TARGET_FIELDS = (
    "target_audio_path",
    "target_audio_tokens_path",
    "target_image_path",
    "target_video_path",
    "source_image_path",
    "edit_mask_path",
    "text_tokens_path",
    "image_tensor_path",
    "audio_tensor_path",
    "video_tensor_path",
    "target_image_tensor_path",
    "target_video_tensor_path",
    "source_image_tensor_path",
    "edit_mask_tensor_path",
    "target_video_tokens_path",
)


def build_snapshot_sample(
    *,
    payload: dict[str, object],
    dataset_root: Path,
    source_path: Path,
    line_number: int,
) -> MultimodalSample:
    """Parse an augmentation row through the canonical schema-3 parser."""

    del source_path, line_number
    record = require_training_record(payload)
    _optional_exact_label(record.get("label"))
    return parse_record(record=record, dataset_root=dataset_root)


def _optional_exact_label(value: object) -> int | None:
    """Accept only exact integers or null for classification labels."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("label must be an integer or null")
    return value


def serialize_snapshot_sample(
    *,
    sample: MultimodalSample,
    dataset_root: Path,
) -> dict[str, object]:
    """Serialize the current sample state without stale canonical metadata."""

    payload = {
        key: value
        for key, value in dict(sample.metadata or {}).items()
        if key not in _TOP_LEVEL_SAMPLE_FIELDS
    }
    payload.update(
        {
            "schema_version": TRAINING_DATASET_SCHEMA_VERSION,
            "sample_id": sample.sample_id,
            "record_id": sample.record_id.strip() or sample.sample_id,
            "modality": _serialized_modality(sample=sample),
            "objects": _serialized_objects(
                sample=sample,
                dataset_root=dataset_root,
            ),
        }
    )

    for key in (
        "text",
        "title",
        "source_url",
        "label",
        "dataset_version",
        "content_hash",
        "processing_version",
        "language",
        "language_confidence",
        "language_script",
        "safety_status",
        "license",
        "license_url",
        "robots_status",
        "terms_source",
        "usage_rules",
    ):
        _set_optional(payload, key, getattr(sample, key))

    payload["task_target"] = _serialized_task_target(
        sample=sample,
        dataset_root=dataset_root,
    )
    return payload


def _serialized_task_target(
    *,
    sample: MultimodalSample,
    dataset_root: Path,
) -> dict[str, object]:
    task_target = {
        key: value
        for key, value in dict(sample.task_target or {}).items()
        if key not in _TASK_TARGET_SAMPLE_FIELDS
    }
    task_target["task_type"] = sample.task_type
    _set_optional(task_target, "task_family", sample.task_family)

    for key in (
        "instruction",
        "question",
        "answer",
        "target_text",
        "target_code",
        "code_language",
        "positive_id",
        "system_text",
        "user_text",
        "assistant_text",
        "tool_name",
        "tool_arguments_json",
        "tool_result_json",
        "sample_source",
        "generator_id",
        "generator_version",
        "verification_status",
        "math_expression",
        "math_solution",
        "humor_explanation",
        "target_table_structure",
        "emotion_label",
        "arousal",
        "valence",
        "dominance",
        "sarcasm_label",
        "speaker_label",
        "background_noise_label",
        "overlapping_speech",
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
    ):
        _set_optional(task_target, key, getattr(sample, key))

    for key in _PATH_TARGET_FIELDS:
        _set_optional(
            task_target,
            key,
            _serialized_path(
                value=getattr(sample, key),
                dataset_root=dataset_root,
            ),
        )

    task_target["output_modalities"] = list(sample.output_modalities)
    task_target["negative_ids"] = list(sample.negative_ids)
    task_target["alignment_score"] = float(sample.alignment_score)
    task_target["answer_evidence_ids"] = list(sample.answer_evidence_ids)
    task_target["reading_order"] = list(sample.reading_order)
    task_target["ocr_confidences"] = list(sample.ocr_confidences)

    _set_collection(
        task_target,
        "conversation_turns",
        [turn.to_mapping() for turn in sample.conversation_turns],
    )
    _set_collection(
        task_target,
        "layout_boxes",
        _json_value(sample.layout_boxes),
    )
    _set_collection(
        task_target,
        "ui_elements",
        _json_value(sample.ui_elements),
    )
    _set_collection(
        task_target,
        "geometry_annotations",
        _json_value(sample.geometry_annotations),
    )
    _set_collection(
        task_target,
        "object_boxes",
        _json_value(sample.object_boxes),
    )
    _set_collection(
        task_target,
        "speaker_segments",
        _json_value(sample.speaker_segments),
    )
    _set_collection(
        task_target,
        "evidence_records",
        _json_value(sample.evidence_records),
    )
    _set_collection(
        task_target,
        "negative_verification",
        _json_value(sample.negative_verification),
    )
    _set_mapping(task_target, "form_fields", sample.form_fields)
    _set_mapping(task_target, "scene_graph", sample.scene_graph)
    _set_optional(task_target, "chart_data", _json_value(sample.chart_data))
    _set_optional(task_target, "prosody", _json_value(sample.prosody))
    return task_target


def _serialized_objects(
    *,
    sample: MultimodalSample,
    dataset_root: Path,
) -> list[dict[str, object]]:
    raw_existing = (sample.metadata or {}).get("objects")
    if raw_existing is not None and not isinstance(raw_existing, list):
        raise ValueError("schema 3.0 field 'objects' must be an array")
    existing = [
        dict(item) for item in (raw_existing or []) if isinstance(item, dict)
    ]
    if raw_existing and len(existing) != len(raw_existing):
        raise ValueError("every schema 3.0 objects[] item must be an object")

    objects: list[dict[str, object]] = []
    for modality, source in (
        ("document", sample.document),
        ("image", sample.image),
        ("audio", sample.audio),
        ("video", sample.video),
    ):
        if source is None:
            continue
        matching = _matching_existing_object(
            objects=existing,
            modality=modality,
        )
        objects.append(
            _serialize_object(
                sample=sample,
                source=source,
                modality=modality,
                dataset_root=dataset_root,
                existing=matching,
            )
        )

    objects.extend(
        dict(item)
        for item in existing
        if str(item.get("role") or "").strip().lower() == "text"
    )
    return objects


def _serialize_object(
    *,
    sample: MultimodalSample,
    source: ModalityObject,
    modality: str,
    dataset_root: Path,
    existing: dict[str, object] | None,
) -> dict[str, object]:
    old = dict(existing or {})
    old_path = old.get("object_path")
    old_url = old.get("object_url")
    path = _serialized_path(value=source.path, dataset_root=dataset_root)
    same_reference = old_path == path and old_url == source.url

    canonical_keys = {
        "object_id",
        "object_path",
        "object_url",
        "object_mime_type",
        "byte_size",
        "role",
    }
    payload = {
        key: value for key, value in old.items() if key not in canonical_keys
    }
    source_metadata = _json_value(source.metadata)
    if not isinstance(source_metadata, dict):
        raise TypeError("modality object metadata must serialize to an object")
    payload.update(
        {
            key: value
            for key, value in source_metadata.items()
            if key not in canonical_keys
        }
    )
    payload["object_id"] = (
        old["object_id"]
        if same_reference and old.get("object_id") is not None
        else f"{sample.sample_id}:{modality}"
    )
    payload["role"] = modality
    _set_optional(payload, "object_path", path)
    _set_optional(payload, "object_url", source.url)
    _set_optional(payload, "object_mime_type", source.mime_type)
    _set_optional(payload, "byte_size", source.byte_size)
    return payload


def _matching_existing_object(
    *,
    objects: list[dict[str, object]],
    modality: str,
) -> dict[str, object] | None:
    for item in objects:
        role = str(item.get("role") or "").strip().lower()
        mime = str(item.get("object_mime_type") or "").strip().lower()
        if role == modality or mime.startswith(f"{modality}/"):
            return item
    return None


def _serialized_modality(*, sample: MultimodalSample) -> str:
    modality = str(sample.modality).strip().lower()
    if modality != "unknown":
        return _normalize_modality(modality)
    if sample.has_conversation:
        return "text"
    stored = str((sample.metadata or {}).get("modality") or "").strip()
    return _normalize_modality(stored)


def _normalize_modality(value: object) -> str:
    modality = str(value or "").strip().lower()
    if modality in _ALLOWED_MODALITIES:
        return modality
    raise ValueError(f"unknown modality: {value!r}")


def _serialized_path(*, value: Path | None, dataset_root: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError:
        return path.as_posix()


def _set_optional(
    payload: dict[str, object],
    key: str,
    value: object,
) -> None:
    if value is None:
        payload.pop(key, None)
    else:
        payload[key] = value


def _set_collection(
    payload: dict[str, object],
    key: str,
    value: object,
) -> None:
    if value:
        payload[key] = value
    else:
        payload.pop(key, None)


def _set_mapping(
    payload: dict[str, object],
    key: str,
    value: dict[str, Any],
) -> None:
    if value:
        payload[key] = _json_value(value)
    else:
        payload.pop(key, None)


def _json_value(value: object) -> object:
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
