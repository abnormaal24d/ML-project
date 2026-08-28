"""Project video annotations onto a selected keyframe."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from augmentation.video.annotations.annotation_receipt import (
    VideoAnnotationReceipt,
)
from augmentation.video.annotations.spatial_annotation_mapping import (
    _BOX_FIELDS,
    _read_box,
    _write_box,
    transform_bounding_box,
)
from augmentation.video.annotations.temporal_annotation_mapping import (
    _TIME_FIELDS,
    _read_interval_seconds,
    _read_point_seconds,
    _validate_interval,
)
from augmentation.video.video_transform_backend import SpatialTransform


def _keyframe_mapping(
    *,
    value: Mapping[str, object],
    spatial: SpatialTransform,
    timestamp_seconds: float,
    source_fps: float,
    include_marker: bool = True,
) -> tuple[dict[str, object], VideoAnnotationReceipt]:
    mapped, receipt, keep = _keyframe_mapping_item(
        value=value,
        spatial=spatial,
        timestamp_seconds=timestamp_seconds,
        source_fps=source_fps,
        include_marker=include_marker,
    )
    return (mapped if keep else {}, receipt)


def _keyframe_mapping_item(
    *,
    value: Mapping[str, object],
    spatial: SpatialTransform,
    timestamp_seconds: float,
    source_fps: float,
    include_marker: bool,
) -> tuple[dict[str, object], VideoAnnotationReceipt, bool]:
    interval = _read_interval_seconds(value)
    if interval is not None:
        _validate_interval(interval[0], interval[1])
        if not (interval[0] <= timestamp_seconds <= interval[1]):
            return {}, VideoAnnotationReceipt(dropped_intervals=1), False
        receipt = VideoAnnotationReceipt(transformed_intervals=1)
    else:
        point = _read_point_seconds(value, source_fps=source_fps)
        if point is not None:
            tolerance = 0.5 / source_fps
            if abs(point - timestamp_seconds) > tolerance:
                return {}, VideoAnnotationReceipt(dropped_points=1), False
            receipt = VideoAnnotationReceipt(transformed_points=1)
        else:
            receipt = VideoAnnotationReceipt()

    result: dict[str, object] = {}
    direct_box = _read_box(value)
    if direct_box is not None:
        transformed = transform_bounding_box(box=direct_box, spatial=spatial)
        if transformed is None:
            return (
                {},
                receipt.merge(VideoAnnotationReceipt(dropped_boxes=1)),
                False,
            )
        result = dict(value)
        _write_box(result=result, source=value, box=transformed)
        for time_key in _TIME_FIELDS:
            result.pop(time_key, None)
        receipt = receipt.merge(VideoAnnotationReceipt(transformed_boxes=1))

    for raw_key, child in value.items():
        key = str(raw_key)
        normalized = key.strip().lower()
        if normalized in _TEMPORAL_COLLECTIONS:
            count = (
                len(child)
                if isinstance(child, Sequence)
                and not isinstance(child, (str, bytes))
                else 1
            )
            receipt = receipt.merge(
                VideoAnnotationReceipt(dropped_intervals=count)
            )
            continue
        if normalized in _TIME_FIELDS or normalized in _BOX_FIELDS:
            continue
        if normalized in {"video_duration_seconds", "media_duration_seconds"}:
            result[key] = 0.0
            continue
        if normalized in {"video_fps", "fps"}:
            result[key] = 0.0
            continue
        if isinstance(child, Mapping):
            mapped, child_receipt, keep = _keyframe_mapping_item(
                value=child,
                spatial=spatial,
                timestamp_seconds=timestamp_seconds,
                source_fps=source_fps,
                include_marker=False,
            )
            receipt = receipt.merge(child_receipt)
            if keep:
                result[key] = mapped
            continue
        if isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
            items: list[object] = []
            for item in child:
                if isinstance(item, Mapping):
                    mapped, child_receipt, keep = _keyframe_mapping_item(
                        value=item,
                        spatial=spatial,
                        timestamp_seconds=timestamp_seconds,
                        source_fps=source_fps,
                        include_marker=False,
                    )
                    receipt = receipt.merge(child_receipt)
                    if keep:
                        items.append(mapped)
                else:
                    items.append(item)
            result[key] = items
            continue
        result[key] = child
    if include_marker:
        result["selected_keyframe_timestamp_seconds"] = timestamp_seconds
    return result, receipt, True


_TEMPORAL_COLLECTIONS = frozenset(
    {
        "speaker_segments",
        "transcript_segments",
        "word_timestamps",
        "timestamps",
        "frame_indices",
    }
)
