"""Tests for objective ImagePayloadExtractor (no OCR / scoring)."""

from __future__ import annotations

from io import BytesIO

import pytest

from crawler.extraction.payloads.image_payload_extractor import (
    ImagePayloadExtractionResult,
    ImagePayloadExtractor,
)

_MAX_DECODE_PIXELS = 10_000


def _png_bytes(
    *, width: int = 8, height: int = 4, color=(0, 128, 255)
) -> bytes:
    pytest.importorskip("PIL")
    from PIL import Image

    image = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _gif_animated_bytes() -> bytes:
    pytest.importorskip("PIL")
    from PIL import Image

    frames = [Image.new("RGB", (6, 6), color=(i * 40, 0, 0)) for i in range(3)]
    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
    return buffer.getvalue()


def test_extract_png_objective_fields() -> None:
    body = _png_bytes(width=12, height=7)
    result = ImagePayloadExtractor(
        max_decode_pixels=_MAX_DECODE_PIXELS,
    ).extract(body=body)
    assert isinstance(result, ImagePayloadExtractionResult)
    assert result.width == 12
    assert result.height == 7
    assert result.format == "PNG"
    assert result.color_mode == "RGB"
    assert result.frame_count == 1
    assert result.byte_size == len(body)
    assert len(result.sha256) == 64
    assert result.sha256 == __import__("hashlib").sha256(body).hexdigest()


def test_extract_empty_body_returns_none() -> None:
    assert (
        ImagePayloadExtractor(
            max_decode_pixels=_MAX_DECODE_PIXELS,
        ).extract(body=b"")
        is None
    )


def test_extract_garbage_returns_none() -> None:
    assert (
        ImagePayloadExtractor(
            max_decode_pixels=_MAX_DECODE_PIXELS,
        ).extract(body=b"not-an-image")
        is None
    )


def test_extract_animated_gif_frame_count() -> None:
    body = _gif_animated_bytes()
    result = ImagePayloadExtractor(
        max_decode_pixels=_MAX_DECODE_PIXELS,
    ).extract(body=body)
    assert result is not None
    assert result.format == "GIF"
    assert result.frame_count >= 2
    assert result.width == 6
    assert result.height == 6


def test_result_has_no_ocr_or_score_fields() -> None:
    body = _png_bytes()
    result = ImagePayloadExtractor(
        max_decode_pixels=_MAX_DECODE_PIXELS,
    ).extract(body=body)
    assert result is not None
    field_names = set(result.__dataclass_fields__)
    assert "extracted_text" not in field_names
    assert "blur_variance" not in field_names
    assert "quality_score" not in field_names
    assert "average_hash" not in field_names


def test_extract_enforces_configured_decode_pixel_limit() -> None:
    extractor = ImagePayloadExtractor(max_decode_pixels=100)

    assert extractor.extract(body=_png_bytes(width=10, height=10)) is not None
    assert extractor.extract(body=_png_bytes(width=11, height=10)) is None
