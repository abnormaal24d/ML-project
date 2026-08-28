"""Image augmentation settings."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, model_validator

from config.augmentation.common import (
    normalize_mime_types,
    validate_operation_names,
    validate_output_directory,
)
from config.base.settings_model import SettingsModel

ImageOperation: TypeAlias = Literal[
    "resize", "content_aware_crop", "compression", "color_jitter", "blur"
]
IMAGE_OPERATION_NAMES = frozenset(
    {"resize", "content_aware_crop", "compression", "color_jitter", "blur"}
)


class ImageAugmentationSettings(SettingsModel):
    """Image-transform augmentation configuration."""

    enabled: bool = False
    operations: tuple[ImageOperation, ...] = (
        "resize",
        "content_aware_crop",
        "compression",
        "color_jitter",
    )
    output_directory: str = "objects/image/augmented"
    metadata_policy: Literal[
        "strip_all", "preserve_color_profile", "preserve_safe", "preserve_all"
    ] = "strip_all"
    resize_max_width: int = Field(default=512, ge=1)
    resize_max_height: int = Field(default=512, ge=1)
    resize_allow_upscale: bool = False
    crop_width: int = Field(default=512, ge=1)
    crop_height: int = Field(default=512, ge=1)
    crop_strategy: Literal["annotation_aware", "entropy"] = "annotation_aware"
    crop_candidate_count: int = Field(default=9, ge=1, le=64)
    crop_variant_count: int = Field(default=3, ge=1, le=16)
    minimum_annotation_coverage: float = Field(default=0.75, ge=0.0, le=1.0)
    minimum_image_difference: float = Field(default=0.01, ge=0.0, le=1.0)
    brightness_min: float = Field(default=0.94, ge=0.1, le=3.0)
    brightness_max: float = Field(default=1.06, ge=0.1, le=3.0)
    contrast_min: float = Field(default=0.95, ge=0.1, le=3.0)
    contrast_max: float = Field(default=1.05, ge=0.1, le=3.0)
    blur_radius_min: float = Field(default=0.2, ge=0.0, le=20.0)
    blur_radius_max: float = Field(default=0.8, ge=0.0, le=20.0)
    compression_quality: int = Field(default=82, ge=1, le=100)
    compression_method: int = Field(default=4, ge=0, le=6)
    default_lossless: bool = True
    minimum_visible_box_fraction: float = Field(default=0.05, ge=0.0, le=1.0)

    max_pixels: int = Field(
        default=40_000_000,
        ge=1,
    )
    min_width: int = Field(
        default=64,
        ge=1,
    )
    min_height: int = Field(
        default=48,
        ge=1,
    )
    output_max_bytes: int = Field(
        default=50_000_000,
        ge=1,
    )

    allowed_output_mime_types: tuple[str, ...] = ("image/webp",)

    @model_validator(mode="after")
    def validate_configuration(
        self,
    ) -> ImageAugmentationSettings:
        """Validate image operations and output configuration."""

        validate_operation_names(
            operations=self.operations,
            allowed=IMAGE_OPERATION_NAMES,
            media_type="image",
        )
        validate_output_directory(self.output_directory)

        normalized_mime_types = normalize_mime_types(
            self.allowed_output_mime_types
        )

        if not normalized_mime_types:
            raise ValueError(
                "allowed_output_mime_types must contain at least one MIME type"
            )

        if normalized_mime_types != self.allowed_output_mime_types:
            object.__setattr__(
                self,
                "allowed_output_mime_types",
                normalized_mime_types,
            )
        for lower, upper, name in (
            (self.brightness_min, self.brightness_max, "brightness"),
            (self.contrast_min, self.contrast_max, "contrast"),
            (self.blur_radius_min, self.blur_radius_max, "blur_radius"),
        ):
            if lower > upper:
                raise ValueError(f"{name}_min must not exceed {name}_max")

        return self
