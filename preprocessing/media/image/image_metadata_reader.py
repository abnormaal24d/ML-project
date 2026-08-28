"""Image metadata extraction and assembly for preprocessing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from preprocessing.media.ports import (
        ImageHashCalculator,
        ImageLoader,
        ImageMetadataAssembler,
    )


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    format_name: str | None = None
    byte_size: int | None = None
    average_hash: str | None = None
    difference_hash: str | None = None
    phash: str | None = None
    sha256: str | None = None
    exif_orientation: int | None = None
    is_animated: bool = False
    frame_count: int | None = None
    icc_profile_sha256: str | None = None


class PillowImageMetadataAssembler:
    """Assemble ImageMetadata from PIL image + raw bytes + hashes."""

    def __init__(
        self,
        *,
        average_hash_calculator: ImageHashCalculator,
        difference_hash_calculator: ImageHashCalculator,
        phash_calculator: ImageHashCalculator,
    ) -> None:
        self._average_hash_calculator = average_hash_calculator
        self._difference_hash_calculator = difference_hash_calculator
        self._phash_calculator = phash_calculator

    def assemble(
        self,
        *,
        img: Any,
        body: bytes,
        fmt: str | None = None,
    ) -> ImageMetadata:
        byte_size = len(body)
        if img is None:
            return ImageMetadata(byte_size=byte_size)
        w, h = getattr(img, "size", (None, None))
        mode = getattr(img, "mode", None)
        f = fmt or getattr(img, "format", None)
        ah = self._average_hash_calculator.compute(img=img)
        dh = self._difference_hash_calculator.compute(img=img)
        ph = self._phash_calculator.compute(img=img)
        sha = hashlib.sha256(body).hexdigest()
        exif_orient = None
        try:
            exif = img.getexif() or {}
            exif_orient = exif.get(274)  # Orientation tag
        except (AttributeError, OSError, TypeError, ValueError):
            exif_orient = None
        animated = bool(getattr(img, "is_animated", False))
        frames = getattr(img, "n_frames", None) if animated else None
        icc = None
        try:
            prof = getattr(img, "info", {}).get("icc_profile")
            if prof:
                profile_bytes = (
                    bytes(prof)
                    if isinstance(prof, (bytes, bytearray))
                    else b""
                )
                icc = hashlib.sha256(profile_bytes).hexdigest()
        except (AttributeError, OSError, TypeError, ValueError):
            icc = None
        return ImageMetadata(
            width=w,
            height=h,
            mode=mode,
            format_name=f,
            byte_size=byte_size,
            average_hash=ah,
            difference_hash=dh,
            phash=ph,
            sha256=sha,
            exif_orientation=exif_orient,
            is_animated=animated,
            frame_count=frames,
            icc_profile_sha256=icc,
        )


class ImageMetadataReader:
    """Main reader used by crawlers and preprocessors."""

    def __init__(
        self,
        *,
        image_loader: ImageLoader,
        metadata_assembler: ImageMetadataAssembler,
    ) -> None:
        self._image_loader = image_loader
        self._metadata_assembler = metadata_assembler

    def read_metadata(self, *, body: bytes) -> ImageMetadata | None:
        if not body:
            return None
        img = self._image_loader.open_image(body=body)
        if img is None:
            return cast(
                ImageMetadata,
                self._metadata_assembler.assemble(img=None, body=body),
            )
        try:
            return cast(
                ImageMetadata,
                self._metadata_assembler.assemble(img=img, body=body),
            )
        except Exception:
            return ImageMetadata(byte_size=len(body))
        finally:
            _close_image(img)


def _close_image(image: Any) -> None:
    try:
        image.close()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return
