"""Canonical parameterized image augmentation operations."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentation.annotations.spatial_transform import SpatialTransform
from augmentation.image.content_aware_crop import CropWindow

if TYPE_CHECKING:
    from PIL.Image import Image

    from config.augmentation.image_settings import ImageAugmentationSettings

SUPPORTED_IMAGE_OPERATIONS = frozenset(
    {"resize", "content_aware_crop", "compression", "color_jitter", "blur"}
)


@dataclass(frozen=True, slots=True)
class ImageOperationResult:
    image: Image
    transform: SpatialTransform
    parameters: dict[str, object]
    save_options: dict[str, object]


def resolve_image_operations(names: tuple[str, ...]) -> tuple[str, ...]:
    unknown = set(names) - SUPPORTED_IMAGE_OPERATIONS
    if unknown:
        raise ValueError(
            f"unknown image augmentation operations: {sorted(unknown)}"
        )
    return names


def apply_image_operation(
    *,
    image: Image,
    operation: str,
    settings: ImageAugmentationSettings,
    seed_key: str,
    crop_window: CropWindow | None = None,
) -> ImageOperationResult:
    """Apply exactly one operation and return its complete output contract."""

    source_width, source_height = image.size
    # This seed exists solely to make augmentation reproducible per sample.
    rng = random.Random(_stable_seed(seed_key, operation))  # nosec: B311
    identity = SpatialTransform(
        source_width=source_width,
        source_height=source_height,
        output_width=source_width,
        output_height=source_height,
        minimum_visible_fraction=settings.minimum_visible_box_fraction,
    )
    if operation == "resize":
        target_width, target_height = _fit_dimensions(
            source_width,
            source_height,
            settings.resize_max_width,
            settings.resize_max_height,
            settings.resize_allow_upscale,
        )
        from PIL import Image as PILImage

        output = image.resize(
            (target_width, target_height), PILImage.Resampling.LANCZOS
        )
        transform = SpatialTransform(
            source_width=source_width,
            source_height=source_height,
            output_width=target_width,
            output_height=target_height,
            scale_x=target_width / source_width,
            scale_y=target_height / source_height,
            minimum_visible_fraction=settings.minimum_visible_box_fraction,
        )
        return ImageOperationResult(
            output,
            transform,
            {
                "mode": "fit_within",
                "max_width": settings.resize_max_width,
                "max_height": settings.resize_max_height,
                "allow_upscale": settings.resize_allow_upscale,
                "output_width": target_width,
                "output_height": target_height,
                "resampling": "lanczos",
            },
            _base_save_options(settings),
        )
    if operation == "content_aware_crop":
        if crop_window is None:
            raise ValueError(
                "content_aware_crop requires a selected crop window"
            )
        target_width, target_height = crop_window.width, crop_window.height
        left, top = crop_window.left, crop_window.top
        output = image.crop(
            (left, top, left + target_width, top + target_height)
        )
        transform = SpatialTransform(
            source_width=source_width,
            source_height=source_height,
            output_width=target_width,
            output_height=target_height,
            crop_left=float(left),
            crop_top=float(top),
            minimum_visible_fraction=settings.minimum_visible_box_fraction,
        )
        return ImageOperationResult(
            output,
            transform,
            {
                "left": left,
                "top": top,
                "width": target_width,
                "height": target_height,
                "selection_strategy": crop_window.strategy,
                "selection_score": crop_window.score,
                "annotation_coverage": crop_window.annotation_coverage,
            },
            _base_save_options(settings),
        )
    if operation == "color_jitter":
        from PIL import ImageEnhance

        brightness = rng.uniform(
            settings.brightness_min, settings.brightness_max
        )
        contrast = rng.uniform(settings.contrast_min, settings.contrast_max)
        output = ImageEnhance.Contrast(
            ImageEnhance.Brightness(image).enhance(brightness)
        ).enhance(contrast)
        return ImageOperationResult(
            output,
            identity,
            {
                "brightness_factor": brightness,
                "contrast_factor": contrast,
            },
            _base_save_options(settings),
        )
    if operation == "blur":
        from PIL import ImageFilter

        radius = rng.uniform(
            settings.blur_radius_min, settings.blur_radius_max
        )
        return ImageOperationResult(
            image.filter(ImageFilter.GaussianBlur(radius=radius)),
            identity,
            {"radius": radius},
            _base_save_options(settings),
        )
    if operation == "compression":
        return ImageOperationResult(
            image.copy(),
            identity,
            {
                "quality": settings.compression_quality,
                "method": settings.compression_method,
                "lossless": False,
            },
            {
                "quality": settings.compression_quality,
                "method": settings.compression_method,
            },
        )
    raise ValueError(f"unknown image operation: {operation}")


def _fit_dimensions(
    width: int,
    height: int,
    max_width: int,
    max_height: int,
    allow_upscale: bool,
) -> tuple[int, int]:
    ratio = min(max_width / width, max_height / height)
    if not allow_upscale:
        ratio = min(1.0, ratio)
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def _base_save_options(
    settings: ImageAugmentationSettings,
) -> dict[str, object]:
    return {
        "lossless": settings.default_lossless,
        "method": settings.compression_method,
    }


def _stable_seed(seed_key: str, operation: str) -> int:
    digest = hashlib.sha256(f"{seed_key}:{operation}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
