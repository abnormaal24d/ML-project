"""Audio augmentation settings."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from config.augmentation.common import (
    validate_operation_names,
    validate_output_directory,
)
from config.base.settings_model import SettingsModel
from config.environment.default_values import DEFAULT_AUDIO_SAMPLE_RATE_HZ

AudioOperation: TypeAlias = Literal[
    "gain_shift",
    "noise_injection",
    "trim",
    "speed_perturbation",
    "sample_rate_normalization",
    "channel_conversion",
]
AUDIO_OPERATION_NAMES = frozenset(
    {
        "gain_shift",
        "noise_injection",
        "trim",
        "speed_perturbation",
        "sample_rate_normalization",
        "channel_conversion",
    }
)


class AudioAugmentationSettings(SettingsModel):
    """Audio-transform augmentation configuration."""

    enabled: bool = False
    operations: tuple[AudioOperation, ...] = (
        "gain_shift",
        "noise_injection",
        "trim",
        "speed_perturbation",
        "sample_rate_normalization",
        "channel_conversion",
    )
    output_directory: str = "objects/audio/augmented"
    metadata_policy: Literal["strip_all", "preserve_safe", "preserve_all"] = (
        "strip_all"
    )

    output_max_bytes: int = Field(
        default=250_000_000,
        ge=1,
    )
    target_sample_rate: int = Field(
        default=DEFAULT_AUDIO_SAMPLE_RATE_HZ,
        ge=1,
        le=384_000,
    )
    target_channels: int = Field(
        default=1,
        ge=1,
        le=8,
    )
    chunk_frames: int = Field(
        default=65_536,
        ge=1,
    )
    gain_db: float = Field(
        default=-0.5,
        ge=-24.0,
        le=24.0,
    )
    noise_std_fraction: float = Field(
        default=0.00015,
        ge=0.0,
        le=0.05,
    )
    trim_silence_threshold_dbfs: float = Field(
        default=-55.0,
        ge=-100.0,
        le=-1.0,
    )
    trim_padding_seconds: float = Field(
        default=0.05,
        ge=0.0,
        le=5.0,
    )
    speed_factor: float = Field(
        default=1.05,
        ge=0.5,
        le=2.0,
    )
    max_clipping_fraction: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
    )
    duration_tolerance_seconds: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_configuration(
        self,
    ) -> AudioAugmentationSettings:
        """Validate audio operations and output configuration."""

        validate_operation_names(
            operations=self.operations,
            allowed=AUDIO_OPERATION_NAMES,
            media_type="audio",
        )
        validate_output_directory(self.output_directory)
        if (
            "sample_rate_normalization" in self.operations
            and self.target_sample_rate < 8_000
        ):
            raise ValueError(
                "target_sample_rate must be at least 8000 Hz when "
                "normalization is enabled"
            )

        return self
