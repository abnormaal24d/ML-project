"""Image normalization preserves explicit orientation/conversion evidence."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from preprocessing.media.adapters.pillow_image import (
    PillowImageLoader,
    inspect_image_dimensions,
    normalize_image_for_training,
)


def _image_bytes(image: Image.Image, *, format: str, exif=None) -> bytes:
    output = BytesIO()
    image.save(output, format=format, exif=exif)
    return output.getvalue()


def test_normalization_records_color_conversion() -> None:
    body = _image_bytes(Image.new("RGBA", (2, 3)), format="PNG")

    result = normalize_image_for_training(
        body=body,
        resize_images=True,
        max_width=64,
        max_height=64,
        max_decode_pixels=100,
    )

    assert result.normalized_bytes
    assert result.format == "JPEG"
    assert result.mode == "RGB"
    assert result.was_converted is True
    assert result.was_oriented is False


def test_normalization_applies_exif_orientation() -> None:
    exif = Image.Exif()
    exif[274] = 6
    body = _image_bytes(
        Image.new("RGB", (2, 3)),
        format="JPEG",
        exif=exif,
    )

    result = normalize_image_for_training(
        body=body,
        resize_images=True,
        max_width=64,
        max_height=64,
        max_decode_pixels=100,
    )

    assert result.normalized_bytes
    assert (result.width, result.height) == (3, 2)
    assert result.was_oriented is True
    assert result.was_converted is False


def test_pillow_loader_respects_explicit_decode_pixel_limit() -> None:
    loader = PillowImageLoader(max_decode_pixels=100)

    accepted = loader.open_image(
        body=_image_bytes(Image.new("RGB", (10, 10)), format="PNG")
    )

    assert accepted is not None
    accepted.close()
    assert (
        loader.open_image(
            body=_image_bytes(Image.new("RGB", (11, 10)), format="PNG")
        )
        is None
    )


def test_normalization_respects_explicit_decode_pixel_limit() -> None:
    result = normalize_image_for_training(
        body=_image_bytes(Image.new("RGB", (11, 10)), format="PNG"),
        resize_images=True,
        max_width=64,
        max_height=64,
        max_decode_pixels=100,
    )

    assert result.normalized_bytes is None
    assert result.error_type == "image_decode_pixel_limit_exceeded"


def test_dimension_inspection_respects_explicit_decode_pixel_limit(
    tmp_path: Path,
) -> None:
    accepted_path = tmp_path / "accepted.png"
    rejected_path = tmp_path / "rejected.png"
    Image.new("RGB", (10, 10)).save(accepted_path, format="PNG")
    Image.new("RGB", (11, 10)).save(rejected_path, format="PNG")

    assert inspect_image_dimensions(
        path=accepted_path,
        max_decode_pixels=100,
    ) == (10, 10)
    assert (
        inspect_image_dimensions(
            path=rejected_path,
            max_decode_pixels=100,
        )
        is None
    )
