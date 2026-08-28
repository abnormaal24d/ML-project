"""Decode, prepare, encode, and clean image augmentation artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

_LOGGER = logging.getLogger(__name__)


def load_prepared_image(path: Path) -> tuple[Image, dict[str, object]]:
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        info = {
            key: value
            for key, value in image.info.items()
            if isinstance(key, str)
        }
        prepared = prepare_image(ImageOps.exif_transpose(image))
        prepared.load()
        return prepared, info


def prepare_image(image: Image) -> Image:
    from PIL import Image as PILImage

    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = PILImage.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def metadata_for_save(
    info: dict[str, object], policy: str
) -> dict[str, object]:
    if policy == "strip_all":
        return {}
    allowed = {"icc_profile"}
    if policy in {"preserve_safe", "preserve_all"}:
        allowed.add("dpi")
    if policy == "preserve_all":
        allowed |= {"exif", "xmp"}
    return {key: info[key] for key in allowed if key in info}


def atomic_save_webp(
    *,
    image: Image,
    output_path: Path,
    save_options: dict[str, object],
    metadata: dict[str, object],
) -> None:
    temporary = output_path.with_suffix(".tmp.webp")
    image.save(temporary, format="WEBP", **save_options, **metadata)
    temporary.replace(output_path)


def remove_artifact(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        _LOGGER.warning(
            "image_augmentation_cleanup_failed",
            extra={"path": str(path), "error_type": type(exc).__name__},
        )
