"""Root multimodal augmentation settings composition."""

from __future__ import annotations

from pydantic import Field, model_validator

from config.augmentation.audio_settings import AudioAugmentationSettings
from config.augmentation.common import (
    AugmentationModality,
    normalize_modalities,
    validate_output_directory,
)
from config.augmentation.document_settings import DocumentAugmentationSettings
from config.augmentation.image_settings import ImageAugmentationSettings
from config.augmentation.text_settings import TextAugmentationSettings
from config.augmentation.video_settings import VideoAugmentationSettings
from config.base.settings_model import SettingsModel


class AugmentationSettings(SettingsModel):
    """Root settings for multimodal augmentation."""

    enabled: bool = True

    cache_directory: str = "data/interim/augmentation_cache"
    cache_enabled: bool = True

    text_field_modalities: tuple[AugmentationModality, ...] = (
        "text",
        "document",
        "image",
        "audio",
        "video",
    )

    apply_to_modalities: tuple[AugmentationModality, ...] | None = None

    text: TextAugmentationSettings = Field(
        default_factory=TextAugmentationSettings,
    )
    document: DocumentAugmentationSettings = Field(
        default_factory=DocumentAugmentationSettings,
    )
    image: ImageAugmentationSettings = Field(
        default_factory=ImageAugmentationSettings,
    )
    audio: AudioAugmentationSettings = Field(
        default_factory=AudioAugmentationSettings,
    )
    video: VideoAugmentationSettings = Field(
        default_factory=VideoAugmentationSettings,
    )

    @model_validator(mode="after")
    def validate_configuration(
        self,
    ) -> AugmentationSettings:
        """Validate directories and normalize modality selections."""

        validate_output_directory(self.cache_directory)

        text_modalities = normalize_modalities(
            self.text_field_modalities,
            field_name="text_field_modalities",
            allow_empty=False,
        )

        if text_modalities != self.text_field_modalities:
            object.__setattr__(
                self,
                "text_field_modalities",
                text_modalities,
            )

        if self.apply_to_modalities is not None:
            applied_modalities = normalize_modalities(
                self.apply_to_modalities,
                field_name="apply_to_modalities",
                allow_empty=True,
            )

            if applied_modalities != self.apply_to_modalities:
                object.__setattr__(
                    self,
                    "apply_to_modalities",
                    applied_modalities,
                )

        return self

    @property
    def effective_text_field_modalities(
        self,
    ) -> tuple[AugmentationModality, ...]:
        """Return modalities whose text fields may receive variants."""

        if self.apply_to_modalities is not None:
            return self.apply_to_modalities

        return self.text_field_modalities
