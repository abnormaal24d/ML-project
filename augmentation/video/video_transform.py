"""Canonical video augmentation plans, receipts, and validation rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from augmentation.video.video_transform_backend import (
    AudioPolicy,
    ResizeMode,
    SpatialTransform,
    VideoProbe,
)


@dataclass(frozen=True, slots=True)
class TimelineTransform:
    """Temporal crop mapping that preserves playback speed."""

    crop_start_seconds: float
    crop_end_seconds: float
    output_duration_seconds: float
    source_fps: float
    output_fps: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VideoClipReceipt:
    """Actual clip properties and transform decisions."""

    source: VideoProbe
    output: VideoProbe
    timeline: TimelineTransform
    spatial: SpatialTransform
    audio_policy: AudioPolicy
    output_audio_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "output": self.output.to_dict(),
            "timeline": self.timeline.to_dict(),
            "spatial": self.spatial.to_dict(),
            "audio_policy": self.audio_policy,
            "output_audio_status": self.output_audio_status,
        }


@dataclass(frozen=True, slots=True)
class VideoKeyframeReceipt:
    """Timestamp and geometry of an extracted keyframe."""

    source: VideoProbe
    timestamp_seconds: float
    spatial: SpatialTransform
    output_width: int
    output_height: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "timestamp_seconds": self.timestamp_seconds,
            "spatial": self.spatial.to_dict(),
            "output_width": self.output_width,
            "output_height": self.output_height,
            "audio_policy": "not_applicable_keyframe",
        }


def build_clip_receipt(
    *,
    source: VideoProbe,
    output: VideoProbe,
    crop_start_seconds: float,
    crop_duration_seconds: float,
    output_fps: float,
    output_width: int,
    output_height: int,
    resize_mode: ResizeMode,
    audio_policy: AudioPolicy,
) -> VideoClipReceipt:
    """Create a canonical receipt after output probing."""

    crop_end = min(
        source.duration_seconds, crop_start_seconds + crop_duration_seconds
    )
    timeline = TimelineTransform(
        crop_start_seconds=crop_start_seconds,
        crop_end_seconds=crop_end,
        output_duration_seconds=max(0.0, crop_end - crop_start_seconds),
        source_fps=source.fps,
        output_fps=output_fps,
    )
    spatial = SpatialTransform.build(
        source_width=source.width,
        source_height=source.height,
        output_width=output_width,
        output_height=output_height,
        mode=resize_mode,
    )
    if audio_policy == "remove":
        audio_status = "removed"
    elif source.has_audio and output.has_audio:
        audio_status = "preserved_transcoded"
    elif not source.has_audio and not output.has_audio:
        audio_status = "source_absent"
    else:
        audio_status = "invalid"
    return VideoClipReceipt(
        source=source,
        output=output,
        timeline=timeline,
        spatial=spatial,
        audio_policy=audio_policy,
        output_audio_status=audio_status,
    )


def deterministic_crop_start(
    *,
    duration_seconds: float,
    clip_duration_seconds: float,
    seed_hex: str,
    mode: Literal["deterministic", "start", "center"],
) -> float:
    """Return a stable crop offset inside the source timeline."""

    maximum = max(0.0, duration_seconds - clip_duration_seconds)
    if maximum <= 0.0 or mode == "start":
        return 0.0
    if mode == "center":
        return maximum / 2.0
    if mode != "deterministic":
        raise ValueError(f"unsupported_video_crop_offset_mode:{mode}")
    fraction = int(seed_hex[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return maximum * fraction


def validate_clip_receipt(
    *,
    receipt: VideoClipReceipt,
    output_path: Path,
    output_max_bytes: int,
    duration_tolerance_seconds: float,
    fps_tolerance: float,
) -> dict[str, object]:
    """Validate decoded output properties and audio policy."""

    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise ValueError("generated_video_missing_or_empty")
    if output_path.stat().st_size > output_max_bytes:
        raise ValueError("generated_video_size_invalid")
    expected_duration = receipt.timeline.output_duration_seconds
    if (
        abs(receipt.output.duration_seconds - expected_duration)
        > duration_tolerance_seconds
    ):
        raise ValueError("generated_video_duration_mismatch")
    if abs(receipt.output.fps - receipt.timeline.output_fps) > fps_tolerance:
        raise ValueError("generated_video_fps_mismatch")
    if (
        receipt.output.width != receipt.spatial.output_width
        or receipt.output.height != receipt.spatial.output_height
    ):
        raise ValueError("generated_video_dimensions_mismatch")
    if receipt.audio_policy == "remove" and receipt.output.has_audio:
        raise ValueError("generated_video_audio_not_removed")
    if (
        receipt.audio_policy == "preserve"
        and receipt.source.has_audio
        and not receipt.output.has_audio
    ):
        raise ValueError("generated_video_audio_missing")
    if receipt.output_audio_status == "invalid":
        raise ValueError("generated_video_audio_policy_mismatch")
    expected_frame_count = max(
        1, int(round(expected_duration * receipt.timeline.output_fps))
    )
    if (
        receipt.output.frame_count is not None
        and abs(receipt.output.frame_count - expected_frame_count) > 1
    ):
        raise ValueError("generated_video_frame_count_mismatch")
    if receipt.audio_policy == "preserve" and receipt.source.has_audio:
        if receipt.output.audio_duration_seconds is None:
            raise ValueError("generated_video_audio_duration_missing")
        if (
            abs(receipt.output.audio_duration_seconds - expected_duration)
            > duration_tolerance_seconds
        ):
            raise ValueError("generated_video_audio_duration_mismatch")
    return {
        "byte_size": output_path.stat().st_size,
        "duration_seconds": receipt.output.duration_seconds,
        "fps": receipt.output.fps,
        "width": receipt.output.width,
        "height": receipt.output.height,
        "frame_count": receipt.output.frame_count,
        "has_audio": receipt.output.has_audio,
        "audio_status": receipt.output_audio_status,
        "audio_codec": receipt.output.audio_codec,
        "audio_sample_rate": receipt.output.audio_sample_rate,
        "audio_channels": receipt.output.audio_channels,
        "audio_duration_seconds": receipt.output.audio_duration_seconds,
        "decoded": True,
    }
