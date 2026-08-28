"""Build lineage-bound metadata for accepted video variants (clip and keyframe)."""

from __future__ import annotations

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

__all__ = [
    "build_video_clip_variant_metadata",
    "build_video_keyframe_variant_metadata",
    "keyframe_timestamp",
]

_CLIP_OPERATION = "video_clip_transform"
_KEYFRAME_OPERATION = "video_keyframe_view"


def keyframe_timestamp(
    *, duration_seconds: float, fps: float, fraction: float
) -> float:
    if duration_seconds <= 0.0 or fps <= 0.0:
        raise ValueError("video_probe_invalid_for_keyframe")
    latest = max(0.0, duration_seconds - (1.0 / fps))
    return min(max(0.0, duration_seconds * fraction), latest)


def _video_variant_metadata_base(
    *,
    sample: MultimodalSample,
    variant_id: str,
    operation: str,
    augmentation_type: str,
    modifies: tuple[str, ...],
    dataset_root: Path,
    source_path: Path,
    output_path: Path,
    mime_type: str,
    cache_key: str,
    source_sha256: str,
    output_sha256: str,
    config_hash: str,
    receipt_field: str,
    receipt: dict[str, object],
    validation: dict[str, object],
    annotation_receipt: dict[str, int],
) -> dict[str, object]:
    metadata = media_variant_metadata(
        sample=sample,
        variant_id=variant_id,
        augmentation_name=operation,
        augmentation_type=augmentation_type,
        modifies=modifies,
    )
    metadata.update(
        {
            "augmentation_source_path": relative_dataset_path(
                dataset_root=dataset_root,
                output_path=source_path,
            ),
            "augmentation_output_path": relative_dataset_path(
                dataset_root=dataset_root,
                output_path=output_path,
            ),
            "augmentation_output_mime_type": mime_type,
            "augmentation_output_byte_size": output_path.stat().st_size,
        }
    )
    metadata.update(
        {
            "augmentation_cache_key": cache_key,
            "augmentation_source_sha256": source_sha256,
            "augmentation_output_sha256": output_sha256,
            "augmentation_config_hash": config_hash,
            "augmentation_implementation_version": (
                MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION
            ),
            "augmentation_implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
            receipt_field: receipt,
            "augmentation_video_validation": validation,
            "augmentation_annotation_transform": annotation_receipt,
        }
    )
    return metadata


def build_video_clip_variant_metadata(
    *,
    sample: MultimodalSample,
    dataset_root: Path,
    source_path: Path,
    output_path: Path,
    cache_key: str,
    source_sha256: str,
    output_sha256: str,
    config_hash: str,
    variant_id: str,
    receipt: dict[str, object],
    validation: dict[str, object],
    annotation_receipt: dict[str, int],
) -> dict[str, object]:
    return _video_variant_metadata_base(
        sample=sample,
        variant_id=variant_id,
        operation=_CLIP_OPERATION,
        augmentation_type="media_transform",
        modifies=("video", "spatial_annotations", "temporal_annotations"),
        dataset_root=dataset_root,
        source_path=source_path,
        output_path=output_path,
        mime_type="video/mp4",
        cache_key=cache_key,
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        config_hash=config_hash,
        receipt_field="augmentation_video_transform",
        receipt=receipt,
        validation=validation,
        annotation_receipt=annotation_receipt,
    )


def build_video_keyframe_variant_metadata(
    *,
    sample: MultimodalSample,
    dataset_root: Path,
    output_path: Path,
    source_path: Path,
    cache_key: str,
    source_sha256: str,
    output_sha256: str,
    config_hash: str,
    variant_id: str,
    receipt: dict[str, object],
    validation: dict[str, object],
    annotation_receipt: dict[str, int],
) -> dict[str, object]:
    metadata = _video_variant_metadata_base(
        sample=sample,
        variant_id=variant_id,
        operation=_KEYFRAME_OPERATION,
        augmentation_type=_KEYFRAME_OPERATION,
        modifies=(
            "video_keyframe_view",
            "spatial_annotations",
            "temporal_annotations",
        ),
        dataset_root=dataset_root,
        source_path=source_path,
        output_path=output_path,
        mime_type="image/jpeg",
        cache_key=cache_key,
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        config_hash=config_hash,
        receipt_field="augmentation_video_keyframe_transform",
        receipt=receipt,
        validation=validation,
        annotation_receipt=annotation_receipt,
    )
    metadata.update(
        {
            "derived_from_modality": "video",
            "output_modality": "image",
            "source_video_path": source_path.as_posix(),
            "keyframe_timestamp_sec": receipt["timestamp_seconds"],
        }
    )
    return metadata
