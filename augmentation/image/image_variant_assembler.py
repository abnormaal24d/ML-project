"""Assemble image variant metadata and final multimodal samples."""

from __future__ import annotations

from pathlib import Path

from augmentation.annotations.spatial_transform import SpatialTransform
from augmentation.image.image_annotation_transformer import (
    transform_image_sample,
)
from augmentation.variant_lineage import (
    MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
    MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION,
)
from augmentation.variant_metadata import media_variant_metadata
from mmcrawler_datasets.schema import MultimodalSample
from mmcrawler_datasets.training_samples.artifact_path import (
    relative_dataset_path,
)


def stable_dataset_path(root: Path, path: Path) -> str:
    return relative_dataset_path(dataset_root=root, output_path=path)


def assemble_image_variant(
    *,
    sample: MultimodalSample,
    root: Path,
    source_path: Path,
    output_path: Path,
    operation: str,
    cache_key: str,
    source_sha256: str,
    output_sha256: str,
    config_hash: str,
    variant_id: str,
    parameters: dict[str, object],
    transform: SpatialTransform,
) -> MultimodalSample:
    metadata = media_variant_metadata(
        sample=sample,
        variant_id=variant_id,
        augmentation_name=f"image_{operation}",
        augmentation_type="media_transform",
        modifies=("image", "spatial_annotations"),
    )
    metadata.update(
        {
            "augmentation_operation": operation,
            "augmentation_parameters": parameters,
            "augmentation_spatial_transform": transform.receipt(),
            "augmentation_cache_key": cache_key,
            "augmentation_source_path": stable_dataset_path(root, source_path),
            "augmentation_output_path": stable_dataset_path(root, output_path),
            "augmentation_output_mime_type": "image/webp",
            "augmentation_output_byte_size": output_path.stat().st_size,
            "augmentation_source_sha256": source_sha256,
            "augmentation_output_sha256": output_sha256,
            "augmentation_config_hash": config_hash,
            "augmentation_implementation_version": MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION,
            "augmentation_implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
        }
    )
    return transform_image_sample(
        sample=sample,
        variant_id=variant_id,
        output_path=output_path,
        source_path=stable_dataset_path(root, source_path),
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        operation=operation,
        parameters=parameters,
        transform=transform,
        metadata=metadata,
    )
