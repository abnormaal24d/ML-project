"""Shared contracts for deterministic video transforms and media adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

ResizeMode = Literal["letterbox", "center_crop"]
AudioPolicy = Literal["preserve", "remove"]


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """Decoded stream properties used to plan and validate transforms."""

    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int | None
    has_audio: bool
    video_codec: str | None = None
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    audio_duration_seconds: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpatialTransform:
    """Affine resize/crop mapping from source pixels to output pixels."""

    source_width: int
    source_height: int
    output_width: int
    output_height: int
    mode: ResizeMode
    scale: float
    offset_x: float
    offset_y: float

    @classmethod
    def build(
        cls,
        *,
        source_width: int,
        source_height: int,
        output_width: int,
        output_height: int,
        mode: ResizeMode,
    ) -> SpatialTransform:
        if min(source_width, source_height, output_width, output_height) <= 0:
            raise ValueError("video_dimensions_must_be_positive")
        if mode == "letterbox":
            scale = min(
                output_width / source_width, output_height / source_height
            )
        elif mode == "center_crop":
            scale = max(
                output_width / source_width, output_height / source_height
            )
        else:
            raise ValueError(f"unsupported_video_resize_mode:{mode}")
        scaled_width = source_width * scale
        scaled_height = source_height * scale
        return cls(
            source_width=source_width,
            source_height=source_height,
            output_width=output_width,
            output_height=output_height,
            mode=mode,
            scale=scale,
            offset_x=(output_width - scaled_width) / 2.0,
            offset_y=(output_height - scaled_height) / 2.0,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VideoTransformBackend(Protocol):
    """Injected backend for probing, transforming, and decoding video."""

    def is_available(self) -> bool: ...

    def probe(self, *, path: Path) -> VideoProbe: ...

    def render_clip(
        self,
        *,
        source_path: Path,
        output_path: Path,
        crop_start_seconds: float,
        crop_duration_seconds: float,
        output_fps: float,
        output_width: int,
        output_height: int,
        resize_mode: ResizeMode,
        audio_policy: AudioPolicy,
        timeout_seconds: float,
    ) -> None: ...

    def extract_keyframe(
        self,
        *,
        source_path: Path,
        output_path: Path,
        timestamp_seconds: float,
        output_width: int,
        output_height: int,
        resize_mode: ResizeMode,
        timeout_seconds: float,
    ) -> None: ...

    def decode_check(self, *, path: Path, timeout_seconds: float) -> None: ...

    def validate_keyframe(
        self,
        *,
        path: Path,
        expected_width: int,
        expected_height: int,
        output_max_bytes: int,
        timeout_seconds: float,
    ) -> dict[str, object]: ...
