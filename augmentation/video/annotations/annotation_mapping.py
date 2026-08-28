"""Recursively transform mixed spatial and temporal video annotation mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from augmentation.video.annotations.annotation_receipt import (
    VideoAnnotationReceipt,
)
from augmentation.video.annotations.annotation_value_parsing import (
    finite_number as _finite_number,
)
from augmentation.video.annotations.spatial_annotation_mapping import (
    _BOX_FIELDS,
    _read_box,
    _write_box,
    transform_bounding_box,
)
from augmentation.video.annotations.temporal_annotation_mapping import (
    _TIME_FIELDS,
    _map_frame_indices,
    _map_interval,
    _map_point,
    _read_interval_seconds,
    _read_point_seconds,
    _write_interval,
    _write_point,
)
from augmentation.video.video_transform import TimelineTransform
from augmentation.video.video_transform_backend import SpatialTransform


def transform_video_annotation_mapping(
    *,
    value: Mapping[str, object],
    spatial: SpatialTransform,
    timeline: TimelineTransform,
) -> tuple[dict[str, object], VideoAnnotationReceipt]:
    """Recursively map canonical box/time fields inside metadata and targets."""

    mapped, receipt, keep = _transform_mapping(
        value=value,
        spatial=spatial,
        timeline=timeline,
    )
    return (mapped if keep else {}, receipt)


def _transform_mapping(
    *,
    value: Mapping[str, object],
    spatial: SpatialTransform,
    timeline: TimelineTransform,
) -> tuple[dict[str, object], VideoAnnotationReceipt, bool]:
    interval = _read_interval_seconds(value)
    if interval is not None:
        transformed = _map_interval(
            start_seconds=interval[0],
            end_seconds=interval[1],
            timeline=timeline,
        )
        if transformed is None:
            return {}, VideoAnnotationReceipt(dropped_intervals=1), False
        result = dict(value)
        _write_interval(result=result, source=value, mapped=transformed)
        receipt = VideoAnnotationReceipt(transformed_intervals=1)
    else:
        point = _read_point_seconds(value, source_fps=timeline.source_fps)
        if point is not None:
            transformed_point = _map_point(seconds=point, timeline=timeline)
            if transformed_point is None:
                return {}, VideoAnnotationReceipt(dropped_points=1), False
            result = dict(value)
            _write_point(
                result=result,
                source=value,
                mapped=transformed_point,
                output_fps=timeline.output_fps,
                output_duration_seconds=timeline.output_duration_seconds,
            )
            receipt = VideoAnnotationReceipt(transformed_points=1)
        else:
            result = dict(value)
            receipt = VideoAnnotationReceipt()

    box = _read_box(value)
    if box is not None:
        transformed_box = transform_bounding_box(box=box, spatial=spatial)
        if transformed_box is None:
            return (
                {},
                receipt.merge(VideoAnnotationReceipt(dropped_boxes=1)),
                False,
            )
        _write_box(result=result, source=value, box=transformed_box)
        receipt = receipt.merge(VideoAnnotationReceipt(transformed_boxes=1))

    for raw_key, child in value.items():
        key = str(raw_key)
        normalized = key.strip().lower()
        if normalized in _TIME_FIELDS or normalized in _BOX_FIELDS:
            continue
        if normalized in {"video_duration_seconds", "media_duration_seconds"}:
            result[key] = timeline.output_duration_seconds
            continue
        if normalized in {"video_fps", "fps"}:
            result[key] = timeline.output_fps
            continue
        if normalized == "frame_indices":
            result[key], child_receipt = _map_frame_indices(
                value=child,
                timeline=timeline,
            )
            receipt = receipt.merge(child_receipt)
            continue
        if normalized == "timestamps":
            result[key], child_receipt = _map_timestamps(
                value=child,
                timeline=timeline,
                spatial=spatial,
            )
            receipt = receipt.merge(child_receipt)
            continue
        if isinstance(child, Mapping):
            child_result, child_receipt, keep = _transform_mapping(
                value=child,
                spatial=spatial,
                timeline=timeline,
            )
            receipt = receipt.merge(child_receipt)
            if keep:
                result[key] = child_result
            else:
                result.pop(key, None)
            continue
        if isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
            items: list[object] = []
            for item in child:
                if isinstance(item, Mapping):
                    child_result, child_receipt, keep = _transform_mapping(
                        value=item,
                        spatial=spatial,
                        timeline=timeline,
                    )
                    receipt = receipt.merge(child_receipt)
                    if keep:
                        items.append(child_result)
                else:
                    items.append(item)
            result[key] = items
    return result, receipt, True


def _map_timestamps(
    *,
    value: object,
    timeline: TimelineTransform,
    spatial: SpatialTransform,
) -> tuple[list[object], VideoAnnotationReceipt]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("video_timestamps_must_be_array")
    output: list[object] = []
    transformed = dropped = 0
    for item in value:
        if isinstance(item, Mapping):
            mapped, receipt, keep = _transform_mapping(
                value=item,
                spatial=spatial,
                timeline=timeline,
            )
            transformed += (
                receipt.transformed_points + receipt.transformed_intervals
            )
            dropped += receipt.dropped_points + receipt.dropped_intervals
            if keep:
                output.append(mapped)
            continue
        seconds = _finite_number(item, field="timestamps[]")
        mapped_point = _map_point(seconds=seconds, timeline=timeline)
        if mapped_point is None:
            dropped += 1
            continue
        output.append(mapped_point)
        transformed += 1
    return output, VideoAnnotationReceipt(
        transformed_points=transformed, dropped_points=dropped
    )
