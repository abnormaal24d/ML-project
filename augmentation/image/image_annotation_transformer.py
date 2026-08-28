"""Apply image spatial transforms to every supported annotation family."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from augmentation.annotations.spatial_transform import (
    SpatialTransform,
    transform_layout_boxes,
    transform_mapping,
    transform_object_boxes,
    transform_ui_elements,
)
from mmcrawler_datasets.schema import ModalityObject, MultimodalSample


def transform_image_sample(
    *,
    sample: MultimodalSample,
    variant_id: str,
    output_path: Path,
    source_path: str,
    source_sha256: str,
    output_sha256: str,
    operation: str,
    parameters: dict[str, object],
    transform: SpatialTransform,
    metadata: dict[str, object],
) -> MultimodalSample:
    return replace(
        sample,
        sample_id=variant_id,
        image=ModalityObject(
            path=output_path,
            mime_type="image/webp",
            byte_size=output_path.stat().st_size,
            metadata={
                "source_path": source_path,
                "source_sha256": source_sha256,
                "output_sha256": output_sha256,
                "operation": operation,
                "parameters": parameters,
            },
        ),
        layout_boxes=transform_layout_boxes(sample.layout_boxes, transform),
        ui_elements=transform_ui_elements(sample.ui_elements, transform),
        object_boxes=transform_object_boxes(sample.object_boxes, transform),
        form_fields=transform_mapping(sample.form_fields, transform),
        scene_graph=transform_mapping(sample.scene_graph, transform),
        task_target=transform_mapping(sample.task_target, transform),
        image_tensor_path=None,
        target_image_tensor_path=None,
        source_image_tensor_path=None,
        edit_mask_tensor_path=None,
        metadata=metadata,
    )
