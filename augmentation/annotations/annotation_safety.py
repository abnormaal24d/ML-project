"""Fail-closed guards for media annotations that are not transformed yet."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from mmcrawler_datasets.schema import MultimodalSample

MediaKind = Literal["document", "image", "audio", "video"]

_SPATIAL_TASK_KEY_MARKERS = (
    "annotation",
    "bbox",
    "bounding_box",
    "bounding_boxes",
    "box_coordinates",
    "coordinate",
    "coordinates",
    "edit_mask",
    "geometry",
    "keypoint",
    "layout_box",
    "layout_boxes",
    "mask_path",
    "object_box",
    "object_boxes",
    "polygon",
    "region",
    "segmentation",
)
_TEMPORAL_TASK_KEY_MARKERS = (
    "end_ms",
    "end_seconds",
    "frame_index",
    "frame_indices",
    "speaker_segment",
    "speaker_segments",
    "start_ms",
    "start_seconds",
    "time_range",
    "timestamp",
    "timestamps",
    "transcript_segment",
    "transcript_segments",
    "word_timestamps",
)
_DERIVED_MEDIA_KEY_MARKERS = (
    "audio_tensor",
    "image_tensor",
    "video_tensor",
    "target_audio_tokens",
    "target_video_tokens",
)


def non_transformable_annotations(
    *,
    sample: MultimodalSample,
    media_kind: MediaKind,
) -> tuple[str, ...]:
    """Return annotation fields that would become stale after a transform."""

    unsafe: set[str] = set()

    # Canonical axis-aligned layout, UI and object boxes are transformed by
    # the shared spatial mapper. Geometry relations, chart data, form fields
    # and scene graphs remain valid or have nested canonical boxes mapped.

    if media_kind in {"document", "image", "video"}:
        _mark(unsafe, "edit_mask_path", sample.edit_mask_path)
        _mark(unsafe, "edit_mask_tensor_path", sample.edit_mask_tensor_path)
        _mark(unsafe, "image_tensor_path", sample.image_tensor_path)
        _mark(
            unsafe, "target_image_tensor_path", sample.target_image_tensor_path
        )
        _mark(
            unsafe, "source_image_tensor_path", sample.source_image_tensor_path
        )

    if media_kind in {"audio", "video"}:
        _mark(unsafe, "prosody", sample.prosody)
        _mark(unsafe, "audio_tensor_path", sample.audio_tensor_path)
        _mark(
            unsafe, "target_audio_tokens_path", sample.target_audio_tokens_path
        )

    if media_kind == "video":
        _mark(unsafe, "video_tensor_path", sample.video_tensor_path)
        _mark(
            unsafe, "target_video_tensor_path", sample.target_video_tensor_path
        )
        _mark(
            unsafe, "target_video_tokens_path", sample.target_video_tokens_path
        )

    markers = list(_DERIVED_MEDIA_KEY_MARKERS)
    if media_kind in {"document", "image"}:
        # Canonical nested ``box`` values are transformable. Masks, polygons,
        # keypoints and segmentations remain fail-closed.
        markers.extend(
            ("edit_mask", "keypoint", "mask_path", "polygon", "segmentation")
        )
    elif media_kind == "video":
        # Canonical rectangles and timestamps are transformed by the video
        # annotation mapper. Unsupported masks, polygons and keypoints remain
        # fail-closed because no lossless mapping is implemented for them.
        markers.extend(
            ("edit_mask", "keypoint", "mask_path", "polygon", "segmentation")
        )
    if media_kind == "audio":
        markers.extend(_TEMPORAL_TASK_KEY_MARKERS)

    transformable_collections = (
        frozenset(
            {"speaker_segments", "transcript_segments", "word_timestamps"}
        )
        if media_kind in {"audio", "video"}
        else frozenset()
    )
    _scan_mapping(
        value=sample.task_target,
        path="task_target",
        key_markers=tuple(markers),
        unsafe=unsafe,
        transformable_collections=transformable_collections,
    )
    _scan_mapping(
        value=sample.metadata,
        path="metadata",
        key_markers=tuple(markers),
        unsafe=unsafe,
        transformable_collections=transformable_collections,
    )
    return tuple(sorted(unsafe))


def rejection_message(*, fields: tuple[str, ...]) -> str:
    """Build a stable diagnostic for a blocked media transform."""

    return "non-transformable annotations: " + ", ".join(fields)


def _mark(unsafe: set[str], name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, (str, bytes, Sequence, Mapping)) and not value:
        return
    unsafe.add(name)


def _scan_mapping(
    *,
    value: object,
    path: str,
    key_markers: tuple[str, ...],
    unsafe: set[str],
    transformable_collections: frozenset[str] = frozenset(),
) -> None:
    if not isinstance(value, Mapping):
        return
    for raw_key, child in value.items():
        key = str(raw_key).strip().lower()
        child_path = f"{path}.{key}"
        if key in transformable_collections:
            continue
        if _has_value(child) and any(marker in key for marker in key_markers):
            unsafe.add(child_path)
        if isinstance(child, Mapping):
            _scan_mapping(
                value=child,
                path=child_path,
                key_markers=key_markers,
                unsafe=unsafe,
                transformable_collections=transformable_collections,
            )
        elif isinstance(child, Sequence) and not isinstance(
            child, (str, bytes)
        ):
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    _scan_mapping(
                        value=item,
                        path=f"{child_path}[{index}]",
                        key_markers=key_markers,
                        unsafe=unsafe,
                        transformable_collections=transformable_collections,
                    )


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, Sequence, Mapping)):
        return bool(value)
    return True
