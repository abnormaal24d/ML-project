"""Safety and quality validation for image augmentation."""

from __future__ import annotations

import mimetypes
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from augmentation.outcomes.media_validation_outcome import (
    MediaValidationOutcome,
)

if TYPE_CHECKING:
    from config.augmentation.image_settings import ImageAugmentationSettings


def validate_image_input(
    *,
    path: Path,
    declared_mime_type: str | None,
    declared_byte_size: int | None,
    settings: ImageAugmentationSettings,
    allowed_mime_types: frozenset[str],
    max_input_bytes: int,
) -> MediaValidationOutcome:
    signals: dict[str, object] = {
        "path": path.as_posix(),
        "declared_mime_type": declared_mime_type,
        "declared_byte_size": declared_byte_size,
        "exists": path.exists(),
    }
    if not path.is_file():
        return _rejected("missing_image_file", signals=signals)
    byte_size = path.stat().st_size
    signals["byte_size"] = byte_size
    if byte_size <= 0 or byte_size > max_input_bytes:
        return _rejected("invalid_image_size", signals=signals)
    mime_type = declared_mime_type or mimetypes.guess_type(path)[0]
    signals["mime_type"] = mime_type
    if mime_type not in allowed_mime_types:
        return _rejected("unsupported_image_mime_type", signals=signals)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                signals["width"] = width
                signals["height"] = height
                signals["mode"] = image.mode
                if width * height > settings.max_pixels:
                    return _rejected(
                        "image_too_many_pixels",
                        signals=signals,
                    )
                if width < settings.min_width or height < settings.min_height:
                    return _rejected(
                        "image_dimensions_too_small",
                        signals=signals,
                    )
                image.verify()
    except (OSError, ValueError, Image.DecompressionBombWarning) as exc:
        signals["error"] = type(exc).__name__
        return _rejected("image_decode_failed", signals=signals)
    return MediaValidationOutcome(None, signals)


def validate_image_output(
    *,
    path: Path,
    expected_mime_type: str,
    settings: ImageAugmentationSettings,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> MediaValidationOutcome:
    signals: dict[str, object] = {
        "path": path.as_posix(),
        "expected_mime_type": expected_mime_type,
        "exists": path.exists(),
    }
    if expected_mime_type not in settings.allowed_output_mime_types:
        return _rejected("unsupported_output_image_mime_type", signals=signals)
    if not path.is_file() or path.stat().st_size <= 0:
        return _rejected("missing_generated_image", signals=signals)
    byte_size = path.stat().st_size
    signals["byte_size"] = byte_size
    if byte_size > settings.output_max_bytes:
        return _rejected("generated_image_too_large", signals=signals)
    try:
        with Image.open(path) as image:
            signals["width"] = image.width
            signals["height"] = image.height
            if image.width * image.height > settings.max_pixels:
                return _rejected(
                    "generated_image_too_many_pixels", signals=signals
                )
            if image.width < 1 or image.height < 1:
                return _rejected(
                    "generated_image_dimensions_invalid", signals=signals
                )
            if expected_width is not None and image.width != expected_width:
                signals["expected_width"] = expected_width
                return _rejected(
                    "generated_image_width_mismatch", signals=signals
                )
            if expected_height is not None and image.height != expected_height:
                signals["expected_height"] = expected_height
                return _rejected(
                    "generated_image_height_mismatch", signals=signals
                )
            image.verify()
    except (OSError, ValueError) as exc:
        signals["error"] = type(exc).__name__
        return _rejected("generated_image_decode_failed", signals=signals)
    return MediaValidationOutcome(None, signals)


def _rejected(
    reason: str,
    *,
    signals: dict[str, object],
) -> MediaValidationOutcome:
    return MediaValidationOutcome(reason, signals)
