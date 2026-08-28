"""Build document augmentation lineage metadata and stable artifact paths."""

from __future__ import annotations

import hashlib
from pathlib import Path

from augmentation.variant_lineage import (
    MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
    MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION,
)
from augmentation.variant_metadata import media_variant_metadata
from mmcrawler_datasets.schema import MultimodalSample
from mmcrawler_datasets.training_samples.artifact_path import (
    relative_dataset_path,
)


def text_sha256(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def relative_artifact_path(root: Path, path: Path) -> str:
    return relative_dataset_path(dataset_root=root, output_path=path)


def document_variant_metadata(
    *,
    sample: MultimodalSample,
    root: Path,
    source_path: Path | None,
    output_path: Path | None,
    operation: str,
    source_sha256: str,
    output_sha256: str,
    config_hash: str,
    variant_id: str,
    parameters: dict[str, object],
    spatial_receipt: dict[str, object] | None,
    output_mime_type: str,
    output_byte_size: int,
    modifies: tuple[str, ...],
) -> dict[str, object]:
    metadata = media_variant_metadata(
        sample=sample,
        variant_id=variant_id,
        augmentation_name=f"document_{operation}",
        augmentation_type="media_transform",
        modifies=modifies,
    )
    metadata.update(
        {
            "augmentation_operation": operation,
            "augmentation_parameters": parameters,
            "augmentation_source_path": (
                relative_artifact_path(root, source_path)
                if source_path
                else None
            ),
            "augmentation_output_path": (
                relative_artifact_path(root, output_path)
                if output_path
                else None
            ),
            "augmentation_output_mime_type": output_mime_type,
            "augmentation_output_byte_size": output_byte_size,
            "augmentation_source_sha256": source_sha256,
            "augmentation_output_sha256": output_sha256,
            "augmentation_config_hash": config_hash,
            "augmentation_implementation_version": MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION,
            "augmentation_implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
        }
    )
    if spatial_receipt is not None:
        metadata["augmentation_spatial_transform"] = spatial_receipt
    return metadata
