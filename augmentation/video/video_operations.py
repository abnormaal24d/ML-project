"""Registered video augmentation operations as pure mappings."""

from __future__ import annotations

_VIDEO_OUTPUT_KIND = {
    "frame_sampling": "keyframe_view",
    "thumbnail_extraction": "keyframe_view",
    "clip_extraction": "video_clip",
    "temporal_crop": "video_clip",
    "fps_normalization": "video_clip",
    "resolution_normalization": "video_clip",
}


def resolve_video_operations(
    names: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate configured video operation names."""

    unknown = set(names) - _VIDEO_OUTPUT_KIND.keys()
    if unknown:
        raise ValueError(
            f"unknown video augmentation operations: {sorted(unknown)}"
        )
    return names


def resolve_video_output_kinds(
    operations: tuple[str, ...],
) -> frozenset[str]:
    """Map configured operations to clip/keyframe output kinds."""

    resolve_video_operations(operations)
    return frozenset(_VIDEO_OUTPUT_KIND[operation] for operation in operations)
