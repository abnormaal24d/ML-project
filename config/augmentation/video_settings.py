"""Video augmentation settings."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from config.augmentation.common import (
    validate_operation_names,
    validate_output_directory,
)
from config.base.settings_model import SettingsModel

VideoResizeMode: TypeAlias = Literal["letterbox", "center_crop"]
VideoAudioPolicy: TypeAlias = Literal["preserve", "remove"]
VideoCropOffsetMode: TypeAlias = Literal["deterministic", "start", "center"]
VideoOperation: TypeAlias = Literal[
    "frame_sampling",
    "clip_extraction",
    "temporal_crop",
    "fps_normalization",
    "resolution_normalization",
    "thumbnail_extraction",
]
VIDEO_OPERATION_NAMES = frozenset(
    {
        "frame_sampling",
        "clip_extraction",
        "temporal_crop",
        "fps_normalization",
        "resolution_normalization",
        "thumbnail_extraction",
    }
)


class VideoAugmentationSettings(SettingsModel):
    """Video-transform augmentation configuration."""

    enabled: bool = False
    operations: tuple[VideoOperation, ...] = (
        "frame_sampling",
        "clip_extraction",
        "temporal_crop",
        "fps_normalization",
        "resolution_normalization",
        "thumbnail_extraction",
    )
    output_directory: str = "objects/video/augmented"
    metadata_policy: Literal["strip_all", "preserve_safe", "preserve_all"] = (
        "strip_all"
    )
    keyframe_output_directory: str = "objects/video/keyframes"

    output_max_bytes: int = Field(
        default=1_000_000_000,
        ge=1,
    )
    max_clip_duration_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=3600.0,
    )
    output_fps: float = Field(
        default=12.0,
        ge=1.0,
        le=240.0,
    )
    output_width: int = Field(
        default=512,
        ge=2,
        le=8192,
    )
    output_height: int = Field(
        default=512,
        ge=2,
        le=8192,
    )
    resize_mode: VideoResizeMode = "letterbox"
    audio_policy: VideoAudioPolicy = "preserve"
    temporal_crop_offset_mode: VideoCropOffsetMode = "deterministic"
    keyframe_timestamp_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    duration_tolerance_seconds: float = Field(
        default=0.15,
        ge=0.0,
        le=2.0,
    )
    fps_tolerance: float = Field(
        default=0.25,
        ge=0.0,
        le=5.0,
    )
    probe_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
    )
    command_timeout_seconds: float = Field(
        default=180.0,
        ge=1.0,
        le=3600.0,
    )

    @model_validator(mode="after")
    def validate_configuration(
        self,
    ) -> VideoAugmentationSettings:
        """Validate video operations and output configuration."""

        validate_operation_names(
            operations=self.operations,
            allowed=VIDEO_OPERATION_NAMES,
            media_type="video",
        )
        validate_output_directory(self.output_directory)
        validate_output_directory(self.keyframe_output_directory)
        if self.output_width % 2 or self.output_height % 2:
            raise ValueError(
                "video output_width and output_height must be even for yuv420p"
            )

        return self
