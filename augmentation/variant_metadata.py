"""Shared media-augmentation metadata builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmcrawler_datasets.schema import MultimodalSample

__all__ = ["media_variant_metadata"]


def media_variant_metadata(
    *,
    sample: MultimodalSample,
    variant_id: str,
    augmentation_name: str,
    augmentation_type: str,
    modifies: tuple[str, ...],
) -> dict[str, object]:
    """Build shared media-augmentation metadata invariants.

    Modality-specific fields (cache key, mime type, output size, layout,
    keyframe timestamps, etc.) remain the caller's responsibility.
    """

    metadata = dict(sample.metadata or {})
    metadata.update(
        {
            "sample_id": variant_id,
            "augmentation_name": augmentation_name,
            "augmentation_source_sample_id": sample.sample_id,
            "augmentation_variant_id": variant_id,
            "augmentation_type": augmentation_type,
            "augmentation_modifies": list(modifies),
            "augmentation_media_transform_applied": True,
        }
    )
    return metadata
