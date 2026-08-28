"""Training head, decoder, and raw-input schema settings."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from config.base.settings_model import SettingsModel
from config.environment.default_values import (
    DEFAULT_MODEL_DROPOUT_PROBABILITY,
    DEFAULT_RAW_AUDIO_NUM_SAMPLES,
    DEFAULT_RAW_IMAGE_SIZE,
    DEFAULT_RAW_TEXT_MAX_TOKENS,
    DEFAULT_RAW_TEXT_VOCAB_SIZE,
    DEFAULT_RAW_VIDEO_FRAMES,
)
from config.multimodal.encoder_settings import SUPPORTED_MODALITIES, Modality
from schemas.multimodal_tasks import OutputModality

SUPPORTED_OUTPUT_MODALITIES: frozenset[OutputModality] = frozenset(
    {
        "text",
        "class",
        "json",
        "image",
        "audio",
        "video",
        "embedding",
        "code",
    }
)
OUTPUT_TEXT: tuple[OutputModality, ...] = ("text",)


class DecoderSettings(SettingsModel):
    """Text/code/JSON autoregressive decoder settings."""

    enabled: bool = False
    vocab_size: int = Field(default=8192, gt=128)
    hidden_dim: int = Field(default=256, gt=0)
    num_layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=4, gt=0)
    dropout: float = Field(
        default=DEFAULT_MODEL_DROPOUT_PROBABILITY,
        ge=0.0,
        lt=1.0,
    )
    max_target_tokens: int = Field(default=256, gt=8)
    max_context_tokens: int = Field(default=512, gt=0)
    max_text_context_tokens: int = Field(default=128, ge=0)
    max_document_context_tokens: int = Field(default=64, ge=0)
    max_image_context_tokens: int = Field(default=64, ge=0)
    max_audio_context_tokens: int = Field(default=64, ge=0)
    max_video_context_tokens: int = Field(default=64, ge=0)
    tie_input_output_embeddings: bool = True
    use_rotary_embeddings: bool = True
    rotary_base: float = Field(default=10000.0, gt=1.0)

    @model_validator(mode="after")
    def _validate_decoder_shape(self) -> DecoderSettings:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "text_decoder.hidden_dim must be divisible by num_heads"
            )
        if (
            self.use_rotary_embeddings
            and (self.hidden_dim // self.num_heads) % 2 != 0
        ):
            raise ValueError(
                "text_decoder rotary attention requires an even head dimension"
            )
        configured_budget = sum(
            (
                self.max_text_context_tokens,
                self.max_document_context_tokens,
                self.max_image_context_tokens,
                self.max_audio_context_tokens,
                self.max_video_context_tokens,
            )
        )
        if configured_budget <= 0:
            raise ValueError(
                "text_decoder requires at least one context token budget"
            )
        return self

    # pretrained_name and freeze are intentionally removed (scratch-only rules)


class TrainingHeadSettings(SettingsModel):
    """Classifier, decoder, and raw tensor schema settings."""

    text_decoder: DecoderSettings = Field(default_factory=DecoderSettings)
    num_classes: int = Field(default=2, gt=0)

    raw_text_vocab_size: int = Field(
        default=DEFAULT_RAW_TEXT_VOCAB_SIZE, gt=128
    )
    raw_text_max_tokens: int = Field(default=DEFAULT_RAW_TEXT_MAX_TOKENS, gt=8)
    raw_image_size: int = Field(default=DEFAULT_RAW_IMAGE_SIZE, gt=8)
    raw_audio_num_samples: int = Field(
        default=DEFAULT_RAW_AUDIO_NUM_SAMPLES,
        gt=256,
    )
    raw_video_frames: int = Field(default=DEFAULT_RAW_VIDEO_FRAMES, gt=1)

    enabled_modalities: tuple[Modality, ...] = (
        "text",
        "document",
        "image",
        "audio",
        "video",
    )
    output_modalities: tuple[OutputModality, ...] = Field(
        default_factory=lambda: OUTPUT_TEXT
    )

    freeze_components: tuple[str, ...] = ()
    # pretrained_backbones, adapter_dim, lora_rank removed: not supported in scratch-only training
    gradient_checkpointing: bool = False
    long_context_strategy: Literal[
        "none", "chunking", "memory", "retrieval"
    ] = "none"
    special_tokens: tuple[str, ...] = (
        "<image>",
        "<audio>",
        "<video>",
        "<doc_page>",
        "<ocr>",
        "<answer>",
        "<code>",
        "<speech_out>",
        "<image_out>",
        "<video_out>",
    )

    @field_validator("enabled_modalities")
    @classmethod
    def _validate_enabled_modalities(
        cls,
        value: tuple[Modality, ...],
    ) -> tuple[Modality, ...]:
        if not value:
            raise ValueError("enabled_modalities must not be empty")

        invalid_modalities = set(value) - SUPPORTED_MODALITIES
        if invalid_modalities:
            raise ValueError(
                f"Invalid modalities: {sorted(invalid_modalities)}"
            )

        return tuple(dict.fromkeys(value))

    @field_validator("output_modalities")
    @classmethod
    def _validate_output_modalities(
        cls,
        value: tuple[OutputModality, ...],
    ) -> tuple[OutputModality, ...]:
        if not value:
            raise ValueError("output_modalities must not be empty")

        invalid_modalities = set(value) - SUPPORTED_OUTPUT_MODALITIES
        if invalid_modalities:
            raise ValueError(
                f"Invalid output modalities: {sorted(invalid_modalities)}"
            )

        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def _reject_unsupported_training_options(self) -> TrainingHeadSettings:
        # pretrained_backbones / adapters / LoRA are not part of the scratch-only architecture
        return self
