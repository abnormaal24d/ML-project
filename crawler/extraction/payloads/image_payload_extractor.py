"""Objective image payload metadata from already-fetched bytes.

No OCR, blur scoring, perceptual hashing, or quality judgment. URL
normalization and reference discovery live elsewhere.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from shared.media.adapters.pillow_decode import (
    open_image_with_guard,
)


@dataclass(frozen=True, slots=True)
class ImagePayloadExtractionResult:
    """Deterministic properties of one image payload."""

    width: int
    height: int
    format: str
    color_mode: str | None
    frame_count: int
    exif_orientation: int | None
    byte_size: int
    sha256: str


class ImagePayloadExtractor:
    """Extract objective image payload metadata from raw bytes."""

    def __init__(self, *, max_decode_pixels: int) -> None:
        self._max_decode_pixels = max_decode_pixels

    def extract(self, *, body: bytes) -> ImagePayloadExtractionResult | None:
        """Return payload metadata, or ``None`` when the image is unreadable.

        Requires a successful open with positive dimensions and a known
        format name. Empty bodies always return ``None``.
        """

        if not body:
            return None

        byte_size = len(body)
        sha256 = hashlib.sha256(body).hexdigest()
        image = open_image_with_guard(
            body=body,
            max_decode_pixels=self._max_decode_pixels,
        )
        if image is None:
            return None

        try:
            width, height = _image_size(image)
            if width is None or height is None or width <= 0 or height <= 0:
                return None

            format_name = _image_format(image)
            if not format_name:
                return None

            color_mode = _image_mode(image)
            frame_count = _frame_count(image)
            exif_orientation = _exif_orientation(image)

            return ImagePayloadExtractionResult(
                width=int(width),
                height=int(height),
                format=format_name,
                color_mode=color_mode,
                frame_count=frame_count,
                exif_orientation=exif_orientation,
                byte_size=byte_size,
                sha256=sha256,
            )
        finally:
            _close_image(image)


def _close_image(image: Any) -> None:
    close = getattr(image, "close", None)
    if not callable(close):
        return
    try:
        close()
    except (OSError, RuntimeError, TypeError, ValueError):
        return


def _image_size(image: Any) -> tuple[int | None, int | None]:
    size = getattr(image, "size", None)
    if not isinstance(size, tuple) or len(size) != 2:
        return None, None
    width, height = size
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        return None, None


def _image_format(image: Any) -> str | None:
    raw = getattr(image, "format", None)
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().upper()
    return normalized or None


def _image_mode(image: Any) -> str | None:
    raw = getattr(image, "mode", None)
    if not isinstance(raw, str):
        return None
    normalized = raw.strip()
    return normalized or None


def _frame_count(image: Any) -> int:
    """Return frame count; still images are always 1."""

    n_frames = getattr(image, "n_frames", None)
    if isinstance(n_frames, int) and n_frames > 0:
        return n_frames
    return 1


def _exif_orientation(image: Any) -> int | None:
    getter = getattr(image, "getexif", None)
    if not callable(getter):
        return None
    try:
        exif = getter() or {}
    except Exception:
        return None
    # TIFF/EXIF Orientation tag id.
    value = exif.get(274) if hasattr(exif, "get") else None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
