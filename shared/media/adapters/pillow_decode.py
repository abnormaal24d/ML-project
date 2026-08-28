"""Shared Pillow decode utilities with bounded decode-pixel guard."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

IMAGE_NORMALIZATION_SOFT_ERRORS = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    MemoryError,
)


def open_image_with_guard(
    *,
    body: bytes,
    max_decode_pixels: int,
) -> Any | None:
    """Open an image with a decode-pixel guard before full load.

    Inspects image header dimensions first, rejects if source pixels
    exceed the limit, then performs full decode. Returns the loaded
    image or None on any soft error.
    """
    if not body:
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        image = Image.open(io.BytesIO(body))
        # Verify header only - no full decode yet.
        image.verify()

        source_width, source_height = image.size
        if source_width <= 0 or source_height <= 0:
            return None

        if source_width * source_height > max_decode_pixels:
            return None

        # Reopen for actual load since verify() closes the image.
        image = Image.open(io.BytesIO(body))
        image.load()
        return image
    except IMAGE_NORMALIZATION_SOFT_ERRORS:
        return None


def get_image_file_dimensions_with_guard(
    *,
    path: Path,
    max_decode_pixels: int,
) -> tuple[int, int] | None:
    """Return decoded image file dimensions without full pixel load.

    Rejects images exceeding max_decode_pixels before full decode.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(path) as image:
            image.verify()

            source_width, source_height = image.size
            if source_width <= 0 or source_height <= 0:
                return None

            if source_width * source_height > max_decode_pixels:
                return None

            return int(source_width), int(source_height)
    except (OSError, ValueError):
        return None
