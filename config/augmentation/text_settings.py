"""Text augmentation settings."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from config.augmentation.common import normalize_nonempty_strings
from config.base.settings_model import SettingsModel


class TextAugmentationSettings(SettingsModel):
    """Fine-grained controls for variants that rewrite text fields."""

    enabled: bool = True
    title_prefix_enabled: bool = True
    context_prefix_enabled: bool = True
    context_prefix_task_types: tuple[str, ...] = (
        "audio_text_pair",
        "document_text_pair",
        "image_text_pair",
        "text_pretrain",
        "video_text_pair",
    )
    text_span_focus_enabled: bool = True

    minimum_text_length: int = Field(
        default=24,
        ge=1,
    )
    maximum_text_length: int | None = Field(
        default=8192,
        ge=1,
    )

    truncation_rules: Literal[
        "skip",
        "truncate",
    ] = "skip"

    max_variants_per_sample: int = Field(
        default=2,
        ge=0,
    )

    language_rules: Literal["preserve_source_language"] = (
        "preserve_source_language"
    )

    @model_validator(mode="after")
    def validate_text_lengths(self) -> TextAugmentationSettings:
        """Ensure the configured text-length range is internally valid."""

        if (
            self.maximum_text_length is not None
            and self.maximum_text_length < self.minimum_text_length
        ):
            raise ValueError(
                "maximum_text_length must be greater than or equal to "
                "minimum_text_length"
            )

        return self

    @model_validator(mode="after")
    def validate_context_task_types(self) -> TextAugmentationSettings:
        """Reject blank or duplicate context-prefix task types."""

        normalized = normalize_nonempty_strings(
            values=self.context_prefix_task_types,
            field_name="context_prefix_task_types",
        )

        if normalized != self.context_prefix_task_types:
            object.__setattr__(
                self,
                "context_prefix_task_types",
                normalized,
            )

        return self
