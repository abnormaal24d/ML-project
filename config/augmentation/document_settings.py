"""Document augmentation settings."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from config.augmentation.common import (
    validate_operation_names,
    validate_output_directory,
)
from config.base.settings_model import SettingsModel

DocumentAugmentationMode: TypeAlias = Literal[
    "text_field_only", "document_media"
]
DocumentOperation: TypeAlias = Literal[
    "text_span", "page_image", "layout_preserving", "ocr_normalization"
]
DOCUMENT_OPERATION_NAMES = frozenset(
    {"text_span", "page_image", "layout_preserving", "ocr_normalization"}
)


class DocumentAugmentationSettings(SettingsModel):
    """Document-specific augmentation rules."""

    enabled: bool = True
    mode: DocumentAugmentationMode = "text_field_only"
    operations: tuple[DocumentOperation, ...] = (
        "text_span",
        "page_image",
        "layout_preserving",
        "ocr_normalization",
    )
    output_directory: str = "objects/document/augmented"
    output_max_bytes: int = Field(default=50_000_000, ge=1)
    page_max_pixels: int = Field(default=40_000_000, ge=1)
    page_resize_max_width: int = Field(default=2048, ge=1)
    page_resize_max_height: int = Field(default=2048, ge=1)
    page_output_format: Literal["webp", "png"] = "webp"
    page_webp_quality: int = Field(default=92, ge=1, le=100)

    @model_validator(mode="after")
    def validate_configuration(
        self,
    ) -> DocumentAugmentationSettings:
        """Validate document operations and output configuration."""

        validate_operation_names(
            operations=self.operations,
            allowed=DOCUMENT_OPERATION_NAMES,
            media_type="document",
        )
        validate_output_directory(self.output_directory)

        return self
