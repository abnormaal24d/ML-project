from __future__ import annotations

import pytest

from mmcrawler_datasets.record_components.coercion import require_string
from mmcrawler_datasets.record_components.parsing import (
    indexed_modality_signature,
    require_training_record,
)


def _image_record() -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "sample_id": "image-1",
        "record_id": "record-1",
        "text": "What is shown?",
        "task_target": {
            "task_type": "vqa",
            "question": "What is shown?",
        },
        "objects": [
            {
                "object_id": "object-1",
                "object_path": "media/image.png",
                "object_mime_type": "image/png",
                "role": "image",
            }
        ],
    }


def test_objects_drive_modality_indexing() -> None:
    record = require_training_record(_image_record())
    task_target = record["task_target"]
    assert isinstance(task_target, dict)
    assert require_string(task_target, "task_type") == "vqa"
    assert indexed_modality_signature(
        objects=list(record["objects"]),  # type: ignore[arg-type]
        has_text=True,
    ) == ("image", "text")


def test_training_rows_require_explicit_schema_3() -> None:
    record = _image_record()
    del record["schema_version"]

    with pytest.raises(ValueError, match="unsupported training schema"):
        require_training_record(record)


def test_missing_task_type_is_rejected() -> None:
    record = _image_record()
    task_target = dict(record["task_target"])  # type: ignore[arg-type]
    del task_target["task_type"]
    record["task_target"] = task_target
    # schema still valid; indexing requires explicit task_type later
    parsed = require_training_record(record)
    with pytest.raises(
        ValueError, match="task_type must be a non-empty string"
    ):
        require_string(parsed["task_target"], "task_type")  # type: ignore[arg-type]
