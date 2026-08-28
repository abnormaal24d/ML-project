"""Model settings coordinator composing multimodal configuration slices."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel
from config.multimodal.encoder_settings import EncoderSettings
from config.multimodal.fusion_settings import FusionSettings
from config.multimodal.generation_settings import (
    AudioCodecSettings,
    GenerationModelSettings,
    ImageCodecSettings,
)
from config.multimodal.training_head_settings import TrainingHeadSettings


class ProjectionSettings(SettingsModel):
    """Shared embedding projection dimensions."""

    projection_dim: int = Field(default=256, gt=0)


ModelFamily = Literal["multimodal_model"]


class ModelSettings(
    FusionSettings,
    ProjectionSettings,
    TrainingHeadSettings,
    GenerationModelSettings,
    SettingsModel,
):
    """Validated multimodal model architecture settings."""

    artifact_version: str = "multimodal_model_step1.v1"
    model_family: ModelFamily = "multimodal_model"

    text: EncoderSettings = Field(
        default_factory=lambda: EncoderSettings(input_dim=512)
    )
    image: EncoderSettings = Field(
        default_factory=lambda: EncoderSettings(input_dim=512)
    )
    audio: EncoderSettings = Field(
        default_factory=lambda: EncoderSettings(input_dim=256)
    )
    document: EncoderSettings = Field(
        default_factory=lambda: EncoderSettings(input_dim=512)
    )
    video: EncoderSettings = Field(
        default_factory=lambda: EncoderSettings(input_dim=512)
    )
    image_codec: "ImageCodecSettings" = Field(
        default_factory=ImageCodecSettings
    )
    audio_codec: "AudioCodecSettings" = Field(
        default_factory=AudioCodecSettings
    )

    @property
    def feature_dimensions(self) -> dict[str, int]:
        return {
            "text": int(self.text.input_dim),
            "document": int(self.document.input_dim),
            "image": int(self.image.input_dim),
            "audio": int(self.audio.input_dim),
            "video": int(self.video.input_dim),
        }

    @model_validator(mode="after")
    def _validate_single_model_schema(self) -> ModelSettings:
        encoders = {
            "text": self.text,
            "document": self.document,
            "image": self.image,
            "audio": self.audio,
            "video": self.video,
        }
        output_dimensions = {
            encoders[modality].output_dim
            for modality in self.enabled_modalities
        }
        if len(output_dimensions) != 1:
            raise ValueError(
                "enabled modality encoder output_dim values must match"
            )

        encoder_output_dim = next(iter(output_dimensions))
        if encoder_output_dim != self.fusion_dim:
            raise ValueError(
                "fusion_dim must match enabled encoder output_dim values"
            )
        if self.text_decoder.enabled:
            if self.text_decoder.hidden_dim != self.fusion_dim:
                raise ValueError(
                    "text_decoder.hidden_dim must match fusion_dim for the "
                    "dense multimodal decoder"
                )
            if self.text_decoder.vocab_size != self.raw_text_vocab_size:
                raise ValueError(
                    "text_decoder.vocab_size must match raw_text_vocab_size"
                )
            total_tokens = (
                self.text_decoder.max_context_tokens
                + self.text_decoder.max_target_tokens
            )
            if total_tokens > self.runtime.max_batch_tokens:
                raise ValueError(
                    "text decoder context plus target budget exceeds "
                    "runtime.max_batch_tokens"
                )

        if self.image_generator.enabled or self.image_decoder.enabled:
            if self.raw_image_size != self.image_codec.input_resolution:
                raise ValueError(
                    "raw_image_size must match image_codec.input_resolution "
                    "when image generation is enabled"
                )

        return self
