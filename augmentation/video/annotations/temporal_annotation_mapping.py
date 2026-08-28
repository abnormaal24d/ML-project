"""Transform video temporal annotations and frame coordinates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from augmentation.video.annotations.annotation_receipt import (
    VideoAnnotationReceipt,
)
from augmentation.video.annotations.annotation_value_parsing import (
    finite_number as _finite_number,
)
from augmentation.video.video_transform import TimelineTransform
from mmcrawler_datasets.schema import SpeakerSegment


def _transform_speaker_segments(
    values: tuple[SpeakerSegment, ...],
    timeline: TimelineTransform,
) -> tuple[tuple[SpeakerSegment, ...], VideoAnnotationReceipt]:
    output: list[SpeakerSegment] = []
    transformed = dropped = 0
    for item in values:
        interval = _map_interval(
            start_seconds=item.start_seconds,
            end_seconds=item.end_seconds,
            timeline=timeline,
        )
        if interval is None:
            dropped += 1
            continue
        output.append(
            replace(item, start_seconds=interval[0], end_seconds=interval[1])
        )
        transformed += 1
    return tuple(output), VideoAnnotationReceipt(
        transformed_intervals=transformed,
        dropped_intervals=dropped,
    )


def _map_interval(
    *,
    start_seconds: float,
    end_seconds: float,
    timeline: TimelineTransform,
) -> tuple[float, float] | None:
    _validate_interval(start_seconds, end_seconds)
    start = max(start_seconds, timeline.crop_start_seconds)
    end = min(end_seconds, timeline.crop_end_seconds)
    if end <= start:
        return None
    return (
        min(
            max(0.0, start - timeline.crop_start_seconds),
            timeline.output_duration_seconds,
        ),
        min(
            max(0.0, end - timeline.crop_start_seconds),
            timeline.output_duration_seconds,
        ),
    )


def _map_point(*, seconds: float, timeline: TimelineTransform) -> float | None:
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("invalid_video_annotation_timestamp")
    if (
        seconds < timeline.crop_start_seconds
        or seconds > timeline.crop_end_seconds
    ):
        return None
    return min(
        max(0.0, seconds - timeline.crop_start_seconds),
        timeline.output_duration_seconds,
    )


def _read_interval_seconds(
    value: Mapping[str, object],
) -> tuple[float, float] | None:
    for start_key, end_key, factor in (
        ("start_seconds", "end_seconds", 1.0),
        ("start_ms", "end_ms", 0.001),
        ("start", "end", 1.0),
    ):
        if start_key in value or end_key in value:
            return (
                _finite_number(value.get(start_key), field=start_key) * factor,
                _finite_number(value.get(end_key), field=end_key) * factor,
            )
    time_range = value.get("time_range")
    if isinstance(time_range, Sequence) and not isinstance(
        time_range, (str, bytes)
    ):
        if len(time_range) != 2:
            raise ValueError("video_time_range_requires_two_values")
        return (
            _finite_number(time_range[0], field="time_range[0]"),
            _finite_number(time_range[1], field="time_range[1]"),
        )
    return None


def _read_point_seconds(
    value: Mapping[str, object], *, source_fps: float
) -> float | None:
    for key, factor in (
        ("timestamp_seconds", 1.0),
        ("time_seconds", 1.0),
        ("frame_timestamp_seconds", 1.0),
        ("timestamp_ms", 0.001),
        ("time_ms", 0.001),
    ):
        if key in value:
            return _finite_number(value.get(key), field=key) * factor
    if "frame_index" in value:
        return (
            _finite_number(value.get("frame_index"), field="frame_index")
            / source_fps
        )
    timestamp = value.get("timestamp")
    if timestamp is not None and not isinstance(timestamp, Sequence):
        return _finite_number(timestamp, field="timestamp")
    return None


def _write_interval(
    *,
    result: dict[str, Any],
    source: Mapping[str, object],
    mapped: tuple[float, float],
) -> None:
    if "start_seconds" in source or "end_seconds" in source:
        result["start_seconds"], result["end_seconds"] = mapped
    elif "start_ms" in source or "end_ms" in source:
        result["start_ms"], result["end_ms"] = (
            mapped[0] * 1000.0,
            mapped[1] * 1000.0,
        )
    elif "start" in source or "end" in source:
        result["start"], result["end"] = mapped
    elif "time_range" in source:
        result["time_range"] = [mapped[0], mapped[1]]


def _write_point(
    *,
    result: dict[str, Any],
    source: Mapping[str, object],
    mapped: float,
    output_fps: float,
    output_duration_seconds: float,
) -> None:
    if "timestamp_seconds" in source:
        result["timestamp_seconds"] = mapped
    elif "time_seconds" in source:
        result["time_seconds"] = mapped
    elif "frame_timestamp_seconds" in source:
        result["frame_timestamp_seconds"] = mapped
    elif "timestamp_ms" in source:
        result["timestamp_ms"] = mapped * 1000.0
    elif "time_ms" in source:
        result["time_ms"] = mapped * 1000.0
    elif "frame_index" in source:
        result["frame_index"] = _output_frame_index(
            seconds=mapped,
            output_fps=output_fps,
            output_duration_seconds=output_duration_seconds,
        )
    elif "timestamp" in source:
        result["timestamp"] = mapped


def _map_frame_indices(
    *, value: object, timeline: TimelineTransform
) -> tuple[list[int], VideoAnnotationReceipt]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("video_frame_indices_must_be_array")
    output: list[int] = []
    transformed = dropped = 0
    for item in value:
        source_seconds = (
            _finite_number(item, field="frame_indices[]") / timeline.source_fps
        )
        mapped = _map_point(seconds=source_seconds, timeline=timeline)
        if mapped is None:
            dropped += 1
            continue
        output.append(
            _output_frame_index(
                seconds=mapped,
                output_fps=timeline.output_fps,
                output_duration_seconds=timeline.output_duration_seconds,
            )
        )
        transformed += 1
    return output, VideoAnnotationReceipt(
        transformed_points=transformed, dropped_points=dropped
    )


def _output_frame_index(
    *, seconds: float, output_fps: float, output_duration_seconds: float
) -> int:
    frame_count = max(1, int(round(output_duration_seconds * output_fps)))
    return min(max(0, int(round(seconds * output_fps))), frame_count - 1)


def _validate_interval(start: float, end: float) -> None:
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0.0
        or end < start
    ):
        raise ValueError("invalid_video_annotation_interval")


_TIME_FIELDS = frozenset(
    {
        "start_seconds",
        "end_seconds",
        "start_ms",
        "end_ms",
        "start",
        "end",
        "timestamp_seconds",
        "time_seconds",
        "frame_timestamp_seconds",
        "timestamp_ms",
        "time_ms",
        "timestamp",
        "time_range",
        "frame_index",
    }
)
