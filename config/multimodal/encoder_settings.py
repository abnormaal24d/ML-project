"""Encoder dimension settings for multimodal input modalities."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel
from config.environment.default_values import DEFAULT_MODEL_DROPOUT_PROBABILITY

Modality = Literal["text", "document", "image", "audio", "video"]

SUPPORTED_MODALITIES: frozenset[Modality] = frozenset(
    cast(
        "tuple[Modality, ...]",
        cast(object, ("text", "document", "image", "audio", "video")),
    )
)


class EncoderSettings(SettingsModel):
    """Validated encoder dimensions for one modality encoder."""

    input_dim: int = Field(gt=0)
    hidden_dim: int = Field(default=256, gt=0)
    output_dim: int = Field(default=256, gt=0)
    dropout: float = Field(
        default=DEFAULT_MODEL_DROPOUT_PROBABILITY,
        ge=0.0,
        lt=1.0,
    )
    attention_heads: int = Field(default=8, gt=0)
    # pretrained_name and freeze are intentionally not supported for scratch-only models.

    @model_validator(mode="after")
    def _validate_attention_heads(self) -> EncoderSettings:
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError(
                "encoder hidden_dim must be divisible by attention_heads"
            )
        return self
