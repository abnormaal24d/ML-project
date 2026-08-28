"""Transform audio transcript and speaker timing onto an augmented timeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from mmcrawler_datasets.schema import SpeakerSegment


@dataclass(frozen=True, slots=True)
class AudioTimelineTransform:
    """Trim-and-speed timeline mapping shared by samples and annotations."""

    trim_start_seconds: float
    trim_end_seconds: float
    speed_factor: float
    output_duration_seconds: float

    def interval(
        self,
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> tuple[float, float] | None:
        """Clip one source interval to retained audio and map to output time."""

        if start_seconds < 0.0 or end_seconds < start_seconds:
            raise ValueError("invalid_audio_annotation_interval")
        retained_start = max(start_seconds, self.trim_start_seconds)
        retained_end = min(end_seconds, self.trim_end_seconds)
        if retained_end <= retained_start:
            return None
        start = (retained_start - self.trim_start_seconds) / self.speed_factor
        end = (retained_end - self.trim_start_seconds) / self.speed_factor
        start = min(max(0.0, start), self.output_duration_seconds)
        end = min(max(start, end), self.output_duration_seconds)
        if end <= start:
            return None
        return start, end

    def point(self, *, seconds: float) -> float | None:
        """Map one timestamp when it lies in the retained source interval."""

        if seconds < 0.0:
            raise ValueError("invalid_audio_annotation_timestamp")
        if (
            seconds < self.trim_start_seconds
            or seconds > self.trim_end_seconds
        ):
            return None
        value = (seconds - self.trim_start_seconds) / self.speed_factor
        return min(max(0.0, value), self.output_duration_seconds)


def transform_speaker_segments(
    *,
    segments: tuple[SpeakerSegment, ...],
    timeline: AudioTimelineTransform,
) -> tuple[SpeakerSegment, ...]:
    """Clip, rescale, and discard diarization segments outside retained audio."""

    transformed: list[SpeakerSegment] = []
    for segment in segments:
        interval = timeline.interval(
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
        )
        if interval is None:
            continue
        transformed.append(
            replace(
                segment,
                start_seconds=interval[0],
                end_seconds=interval[1],
            )
        )
    return tuple(transformed)


def transform_audio_annotation_mapping(
    *,
    value: Mapping[str, object],
    timeline: AudioTimelineTransform,
) -> dict[str, object]:
    """Transform canonical timed collections while preserving other metadata."""

    result: dict[str, object] = {}
    for raw_key, child in value.items():
        key = str(raw_key)
        normalized = key.strip().lower()
        if normalized in {
            "speaker_segments",
            "transcript_segments",
            "word_timestamps",
        }:
            result[key] = _transform_collection(value=child, timeline=timeline)
            continue
        if normalized in {
            "audio_duration_seconds",
            "media_duration_seconds",
        }:
            result[key] = timeline.output_duration_seconds
            continue
        if isinstance(child, Mapping):
            result[key] = transform_audio_annotation_mapping(
                value=child,
                timeline=timeline,
            )
            continue
        if isinstance(child, list):
            result[key] = [
                transform_audio_annotation_mapping(
                    value=item, timeline=timeline
                )
                if isinstance(item, Mapping)
                else item
                for item in child
            ]
            continue
        result[key] = child
    return result


def _transform_collection(
    *,
    value: object,
    timeline: AudioTimelineTransform,
) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("timed_audio_annotation_collection_must_be_array")
    transformed: list[object] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("timed_audio_annotation_item_must_be_object")
        mapped = _transform_timed_item(value=item, timeline=timeline)
        if mapped is not None:
            transformed.append(mapped)
    return transformed


def _transform_timed_item(
    *,
    value: Mapping[str, object],
    timeline: AudioTimelineTransform,
) -> dict[str, object] | None:
    result: dict[str, Any] = dict(value)

    interval = _read_interval_seconds(value)
    if interval is not None:
        mapped = timeline.interval(
            start_seconds=interval[0],
            end_seconds=interval[1],
        )
        if mapped is None:
            return None
        _write_interval(result=result, source=value, mapped=mapped)
        return result

    point = _read_point_seconds(value)
    if point is not None:
        mapped_point = timeline.point(seconds=point)
        if mapped_point is None:
            return None
        _write_point(result=result, source=value, mapped=mapped_point)
        return result

    raise ValueError("timed_audio_annotation_missing_time_bounds")


def _read_interval_seconds(
    value: Mapping[str, object],
) -> tuple[float, float] | None:
    pairs = (
        ("start_seconds", "end_seconds", 1.0),
        ("start_ms", "end_ms", 0.001),
        ("start", "end", 1.0),
    )
    for start_key, end_key, factor in pairs:
        if start_key in value or end_key in value:
            start = _finite_number(value.get(start_key), field=start_key)
            end = _finite_number(value.get(end_key), field=end_key)
            return start * factor, end * factor
    timestamp = value.get("timestamp")
    if isinstance(timestamp, Sequence) and not isinstance(
        timestamp, (str, bytes)
    ):
        if len(timestamp) != 2:
            raise ValueError("timestamp_interval_requires_two_values")
        return (
            _finite_number(timestamp[0], field="timestamp[0]"),
            _finite_number(timestamp[1], field="timestamp[1]"),
        )
    return None


def _read_point_seconds(value: Mapping[str, object]) -> float | None:
    for key, factor in (
        ("timestamp_seconds", 1.0),
        ("time_seconds", 1.0),
        ("timestamp_ms", 0.001),
        ("time_ms", 0.001),
    ):
        if key in value:
            return _finite_number(value.get(key), field=key) * factor
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
        result["start_ms"] = mapped[0] * 1000.0
        result["end_ms"] = mapped[1] * 1000.0
    elif "start" in source or "end" in source:
        result["start"], result["end"] = mapped
    elif "timestamp" in source:
        result["timestamp"] = [mapped[0], mapped[1]]


def _write_point(
    *,
    result: dict[str, Any],
    source: Mapping[str, object],
    mapped: float,
) -> None:
    if "timestamp_seconds" in source:
        result["timestamp_seconds"] = mapped
    elif "time_seconds" in source:
        result["time_seconds"] = mapped
    elif "timestamp_ms" in source:
        result["timestamp_ms"] = mapped * 1000.0
    elif "time_ms" in source:
        result["time_ms"] = mapped * 1000.0
    elif "timestamp" in source:
        result["timestamp"] = mapped


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        raise ValueError(f"{field} must be finite")
    return numeric
