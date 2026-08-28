from __future__ import annotations

from augmentation.augmentation_artifact_writer import build_lineage_row


def test_lineage_uses_canonical_primary_object_location() -> None:
    row: dict[str, object] = {
        "schema_version": "3.0",
        "sample_id": "image-1:augmented",
        "augmentation_source_sample_id": "image-1",
        "augmentation_name": "horizontal_flip",
        "modality": "image",
        "task_target": {"task_type": "representation"},
        "objects": [
            {
                "object_id": "image-1:augmented:image",
                "object_path": "augmented/image-1.png",
                "object_mime_type": "image/png",
                "role": "image",
            }
        ],
    }

    lineage = build_lineage_row(row=row)

    assert lineage["output_path"] == "augmented/image-1.png"
    assert lineage["source_sample_id"] == "image-1"
