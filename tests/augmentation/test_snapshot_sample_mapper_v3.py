from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mmcrawler_datasets.schema import ModalityObject
from mmcrawler_datasets.training_samples.snapshot_mapping import (
    build_snapshot_sample,
    serialize_snapshot_sample,
)


def _image_row() -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "sample_id": "image-1",
        "record_id": "record-1",
        "text": "A source caption",
        "task_target": {"task_type": "representation"},
        "objects": [
            {
                "object_id": "image-object-1",
                "object_path": "media/source.png",
                "object_mime_type": "image/png",
                "role": "image",
                "byte_size": 42,
                "origin": "crawler",
            }
        ],
    }


def test_snapshot_mapper_round_trips_canonical_objects(tmp_path: Path) -> None:
    row = _image_row()
    sample = build_snapshot_sample(
        payload=row,
        dataset_root=tmp_path,
        source_path=tmp_path / "train.jsonl",
        line_number=1,
    )

    assert sample.image is not None
    assert sample.image.path == tmp_path / "media/source.png"
    assert sample.image.mime_type == "image/png"

    serialized = serialize_snapshot_sample(
        sample=sample, dataset_root=tmp_path
    )
    assert serialized["schema_version"] == "3.0"
    assert "media_path" not in serialized
    assert "image_path" not in serialized
    assert serialized["objects"] == row["objects"]


def test_augmented_media_replaces_the_primary_object(tmp_path: Path) -> None:
    sample = build_snapshot_sample(
        payload=_image_row(),
        dataset_root=tmp_path,
        source_path=tmp_path / "train.jsonl",
        line_number=1,
    )
    augmented = replace(
        sample,
        sample_id="image-1:augmented",
        image=ModalityObject(
            path=tmp_path / "augmented/image-1.png",
            mime_type="image/png",
            byte_size=84,
            metadata={
                "origin": "augmentation",
                "source_sha256": "a" * 64,
            },
        ),
    )

    serialized = serialize_snapshot_sample(
        sample=augmented,
        dataset_root=tmp_path,
    )
    objects = serialized["objects"]
    assert isinstance(objects, list)
    assert objects == [
        {
            "object_id": "image-1:augmented:image",
            "object_path": "augmented/image-1.png",
            "object_mime_type": "image/png",
            "role": "image",
            "byte_size": 84,
            "origin": "augmentation",
            "source_sha256": "a" * 64,
        }
    ]


def test_snapshot_mapper_rejects_old_schema(tmp_path: Path) -> None:
    old_schema = _image_row()
    old_schema["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="unsupported training schema"):
        build_snapshot_sample(
            payload=old_schema,
            dataset_root=tmp_path,
            source_path=tmp_path / "train.jsonl",
            line_number=1,
        )


def test_snapshot_mapper_rejects_missing_required_ids(tmp_path: Path) -> None:
    row = _image_row()
    del row["sample_id"]
    with pytest.raises(
        ValueError, match="sample_id must be a non-empty string"
    ):
        build_snapshot_sample(
            payload=row,
            dataset_root=tmp_path,
            source_path=tmp_path / "train.jsonl",
            line_number=1,
        )


def test_snapshot_mapper_round_trips_conversation_task_target(
    tmp_path: Path,
) -> None:
    row = {
        "schema_version": "3.0",
        "sample_id": "conversation-1",
        "record_id": "record-conversation-1",
        "modality": "text",
        "text": "",
        "objects": [],
        "task_target": {
            "task_type": "instruction_following",
            "system_text": "Be precise.",
            "user_text": "Use the tool.",
            "assistant_text": "Done.",
            "conversation_turns": [
                {"role": "system", "text": "Be precise.", "turn_index": 0},
                {"role": "user", "text": "Use the tool.", "turn_index": 1},
                {
                    "role": "assistant",
                    "text": "",
                    "turn_index": 2,
                    "tool_name": "lookup",
                    "tool_arguments_json": '{"id":1}',
                },
                {
                    "role": "tool",
                    "text": "",
                    "turn_index": 3,
                    "tool_result_json": '{"value":"x"}',
                },
                {
                    "role": "assistant",
                    "text": "Done.",
                    "turn_index": 4,
                    "is_assistant_answer": True,
                },
            ],
            "answer_evidence_ids": ["evidence-1"],
            "tool_name": "lookup",
            "tool_arguments_json": '{"id":1}',
            "tool_result_json": '{"value":"x"}',
            "sample_source": "verified_external",
            "generator_id": "generator",
            "generator_version": "1",
            "verification_status": "verified",
        },
    }

    first = build_snapshot_sample(
        payload=row,
        dataset_root=tmp_path,
        source_path=tmp_path / "train.jsonl",
        line_number=1,
    )
    serialized = serialize_snapshot_sample(
        sample=first,
        dataset_root=tmp_path,
    )
    second = build_snapshot_sample(
        payload=serialized,
        dataset_root=tmp_path,
        source_path=tmp_path / "train.jsonl",
        line_number=2,
    )

    assert second.system_text == first.system_text
    assert second.user_text == first.user_text
    assert second.assistant_text == first.assistant_text
    assert second.conversation_turns == first.conversation_turns
    assert second.answer_evidence_ids == first.answer_evidence_ids
    assert second.tool_name == first.tool_name
    assert second.tool_arguments_json == first.tool_arguments_json
    assert second.tool_result_json == first.tool_result_json
    assert second.sample_source == first.sample_source
    assert second.generator_id == first.generator_id
    assert second.generator_version == first.generator_version
    assert second.verification_status == first.verification_status


def test_snapshot_mapper_preserves_complete_canonical_contract(
    tmp_path: Path,
) -> None:
    row = {
        "schema_version": "3.0",
        "sample_id": "complete-1",
        "record_id": "record-complete-1",
        "modality": "multimodal",
        "text": "source text",
        "title": "source title",
        "source_url": "https://example.test/source",
        "dataset_version": "2026.08",
        "content_hash": "a" * 64,
        "processing_version": "processor-3",
        "language": "nl",
        "language_confidence": 0.99,
        "language_script": "Latn",
        "safety_status": "approved",
        "license": "CC-BY-4.0",
        "license_url": "https://example.test/license",
        "robots_status": "allowed",
        "terms_source": "publisher",
        "usage_rules": "training_allowed",
        "objects": [
            {
                "object_id": "document-1",
                "object_path": "media/source.pdf",
                "object_mime_type": "application/pdf",
                "role": "document",
            }
        ],
        "task_target": {
            "task_type": "document_text_pair",
            "task_family": "document",
            "instruction": "Match the passage.",
            "question": "Which passage?",
            "answer": "The first passage.",
            "target_text": "first passage",
            "text_tokens_path": "training_tensors/text.pt",
            "target_image_path": "targets/page.png",
            "target_video_tokens_path": "targets/video_tokens.pt",
            "output_modalities": ["embedding"],
            "positive_id": "positive-1",
            "negative_ids": ["negative-1", "negative-2"],
            "alignment_score": 0.75,
            "layout_boxes": [
                {
                    "text": "Header",
                    "box": {
                        "x": 0.1,
                        "y": 0.2,
                        "width": 0.3,
                        "height": 0.1,
                        "page": 1,
                        "coordinate_system": "relative",
                    },
                    "role": "heading",
                    "reading_order": 0,
                    "confidence": 0.98,
                }
            ],
            "object_boxes": [
                {
                    "object_id": "object-1",
                    "label": "chart",
                    "box": {
                        "x": 0.2,
                        "y": 0.3,
                        "width": 0.4,
                        "height": 0.5,
                    },
                    "confidence": 0.9,
                    "attributes": {"color": "blue"},
                }
            ],
            "geometry_annotations": [
                {
                    "subject_id": "object-1",
                    "relation": "below",
                    "object_id": "header-1",
                    "confidence": 0.8,
                }
            ],
            "speaker_segments": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.5,
                    "speaker_id": "speaker-1",
                    "confidence": 0.95,
                    "transcript": "hello",
                    "overlap": False,
                }
            ],
            "prosody": {
                "pitch_hz": 170.0,
                "energy": 0.6,
                "tempo": 125.0,
                "pause_ratio": 0.1,
                "emphasis": ["hello"],
            },
            "evidence_records": [{"id": "evidence-1"}],
            "reading_order": ["header-1", "object-1"],
            "ocr_confidences": [0.98, 0.91],
            "negative_verification": [{"id": "negative-1", "valid": True}],
            "chosen_response": "accepted",
            "rejected_response": "rejected",
            "preference_reason": "grounded",
            "preference_source": "human",
            "must_refuse": False,
            "requires_uncertainty": True,
            "requires_source_citation": True,
            "prompt_injection_present": False,
            "untrusted_document_instruction": True,
            "sensitive_data_present": False,
            "requires_tool_confirmation": True,
            "extension_field": {"retained": True},
        },
        "extension_metadata": {"retained": True},
    }

    first = build_snapshot_sample(
        payload=row,
        dataset_root=tmp_path,
        source_path=tmp_path / "train.jsonl",
        line_number=1,
    )
    serialized = serialize_snapshot_sample(
        sample=first,
        dataset_root=tmp_path,
    )
    second = build_snapshot_sample(
        payload=serialized,
        dataset_root=tmp_path,
        source_path=tmp_path / "train.jsonl",
        line_number=2,
    )

    assert second.record_id == first.record_id
    assert second.task_type == "document_text_pair"
    assert second.task_family == "document"
    assert second.instruction == first.instruction
    assert second.question == first.question
    assert second.answer == first.answer
    assert second.target_text == first.target_text
    assert second.output_modalities == ("embedding",)
    assert second.positive_id == "positive-1"
    assert second.negative_ids == ("negative-1", "negative-2")
    assert second.text_tokens_path == tmp_path / "training_tensors/text.pt"
    assert second.target_image_path == tmp_path / "targets/page.png"
    assert second.target_video_tokens_path == (
        tmp_path / "targets/video_tokens.pt"
    )
    assert second.layout_boxes == first.layout_boxes
    assert second.object_boxes == first.object_boxes
    assert second.geometry_annotations == first.geometry_annotations
    assert second.speaker_segments == first.speaker_segments
    assert second.prosody == first.prosody
    assert second.evidence_records == first.evidence_records
    assert second.reading_order == first.reading_order
    assert second.ocr_confidences == first.ocr_confidences
    assert second.negative_verification == first.negative_verification
    assert second.chosen_response == first.chosen_response
    assert second.rejected_response == first.rejected_response
    assert second.preference_reason == first.preference_reason
    assert second.preference_source == first.preference_source
    assert second.must_refuse is False
    assert second.requires_uncertainty is True
    assert second.requires_source_citation is True
    assert second.prompt_injection_present is False
    assert second.untrusted_document_instruction is True
    assert second.sensitive_data_present is False
    assert second.requires_tool_confirmation is True
    assert serialized["extension_metadata"] == {"retained": True}
    assert serialized["task_target"]["extension_field"] == {"retained": True}


def test_snapshot_serializer_replaces_stale_transformed_annotations(
    tmp_path: Path,
) -> None:
    row = _image_row()
    task_target = dict(row["task_target"])
    task_target["layout_boxes"] = [
        {
            "text": "old",
            "box": {"x": 0.0, "y": 0.0, "width": 0.2, "height": 0.2},
        }
    ]
    row["task_target"] = task_target
    original = build_snapshot_sample(
        payload=row,
        dataset_root=tmp_path,
        source_path=tmp_path / "train.jsonl",
        line_number=1,
    )
    old_box = original.layout_boxes[0]
    assert old_box.box is not None
    transformed = replace(
        original,
        layout_boxes=(
            replace(
                old_box,
                text="new",
                box=replace(old_box.box, x=0.5, width=0.4),
            ),
        ),
    )

    serialized = serialize_snapshot_sample(
        sample=transformed,
        dataset_root=tmp_path,
    )
    reparsed = build_snapshot_sample(
        payload=serialized,
        dataset_root=tmp_path,
        source_path=tmp_path / "train.jsonl",
        line_number=2,
    )

    assert reparsed.layout_boxes == transformed.layout_boxes
    assert reparsed.layout_boxes[0].text == "new"
    assert reparsed.layout_boxes[0].box is not None
    assert reparsed.layout_boxes[0].box.x == pytest.approx(0.5)
    assert serialized["task_target"]["layout_boxes"][0]["text"] == "new"
