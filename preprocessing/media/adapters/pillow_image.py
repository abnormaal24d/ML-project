"""Pillow-backed image loading and training normalization."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from preprocessing.media.ports import (
    ImageNormalizationResult,
)
from shared.media.adapters.pillow_decode import (
    IMAGE_NORMALIZATION_SOFT_ERRORS,
    get_image_file_dimensions_with_guard,
    open_image_with_guard,
)


class PillowImageLoader:
    """Open and inspect images with Pillow."""

    def __init__(self, *, max_decode_pixels: int) -> None:
        self._max_decode_pixels = max_decode_pixels

    def open_image(self, *, body: bytes) -> Any | None:
        return open_image_with_guard(
            body=body,
            max_decode_pixels=self._max_decode_pixels,
        )


def inspect_image_dimensions(
    *,
    path: Path,
    max_decode_pixels: int,
) -> tuple[int, int] | None:
    """Verify an image file and return its decoded dimensions."""

    return get_image_file_dimensions_with_guard(
        path=path,
        max_decode_pixels=max_decode_pixels,
    )


def normalize_image_for_training(
    *,
    body: bytes,
    resize_images: bool,
    max_width: int,
    max_height: int,
    max_decode_pixels: int,
) -> ImageNormalizationResult:
    """Normalize an image payload into a stable training-friendly encoding."""

    if not body:
        return ImageNormalizationResult(
            normalized_bytes=None,
            error_type="empty_body",
        )
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        return ImageNormalizationResult(
            normalized_bytes=None,
            error_type=type(exc).__name__,
        )

    # First, verify header dimensions with decode-pixel guard.
    try:
        img = Image.open(io.BytesIO(body))
        img.verify()

        source_width, source_height = img.size
        if source_width <= 0 or source_height <= 0:
            return ImageNormalizationResult(
                normalized_bytes=None,
                error_type="invalid_dimensions",
            )

        if source_width * source_height > max_decode_pixels:
            return ImageNormalizationResult(
                normalized_bytes=None,
                error_type="image_decode_pixel_limit_exceeded",
            )
    except (OSError, ValueError) as exc:
        return ImageNormalizationResult(
            normalized_bytes=None,
            error_type=type(exc).__name__,
        )

    # Now perform full decode.
    try:
        opened_image = Image.open(io.BytesIO(body))
        opened_image.load()
        orientation = opened_image.getexif().get(274)
        image: Image.Image = ImageOps.exif_transpose(opened_image)
        was_oriented = orientation not in (None, 1)

        if resize_images:
            image.thumbnail(
                (max_width, max_height),
                Image.Resampling.LANCZOS,
            )

        was_converted = image.mode not in {"RGB", "L"}
        if was_converted:
            image = image.convert("RGB")

        width, height = image.size
        buffer = io.BytesIO()
        image_format = "JPEG"
        if image.mode == "L":
            image_format = "PNG"
            image.save(buffer, format=image_format)
        else:
            image.save(buffer, format=image_format, quality=90, optimize=True)

        return ImageNormalizationResult(
            normalized_bytes=buffer.getvalue(),
            format=image_format,
            width=int(width),
            height=int(height),
            mode=str(image.mode),
            was_oriented=was_oriented,
            was_converted=was_converted,
        )
    except IMAGE_NORMALIZATION_SOFT_ERRORS as exc:
        return ImageNormalizationResult(
            normalized_bytes=None,
            error_type=type(exc).__name__,
        )


__all__ = [
    "IMAGE_NORMALIZATION_SOFT_ERRORS",
    "ImageNormalizationResult",
    "PillowImageLoader",
    "inspect_image_dimensions",
    "normalize_image_for_training",
]
