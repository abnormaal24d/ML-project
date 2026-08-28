"""Image blur estimation for preprocessing via injected media abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from preprocessing.media.ports import FrameProcessor, ImageLoader


@dataclass(frozen=True, slots=True)
class ImageBlurScore:
    """Simple blur metric container."""

    laplacian_variance: float | None = None


class PillowGrayscaleArrayConverter:
    """Convert a Pillow image to a grayscale array."""

    def to_grayscale_array(
        self,
        *,
        img: Any,
    ) -> Any | None:
        if img is None:
            return None
        converted = None
        try:
            grayscale_image = img
            if img.mode != "L":
                converted = img.convert("L")
                grayscale_image = converted
            import numpy as np

            return np.array(grayscale_image)
        except (
            AttributeError,
            ImportError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return None
        finally:
            if converted is not None and converted is not img:
                converted.close()


class ImageBlurEstimator:
    """Estimate blur through injected image and frame adapters."""

    def __init__(
        self,
        *,
        image_loader: ImageLoader,
        grayscale_converter: PillowGrayscaleArrayConverter,
        frame_processor: FrameProcessor,
    ) -> None:
        self._image_loader = image_loader
        self._grayscale_converter = grayscale_converter
        self._frame_processor = frame_processor

    def estimate_blur(self, *, body: bytes) -> ImageBlurScore | None:
        image = self._image_loader.open_image(body=body)
        if image is None:
            return None

        try:
            grayscale = self._grayscale_converter.to_grayscale_array(
                img=image,
            )
            if grayscale is None:
                return None

            variance = self._frame_processor.laplacian_variance(grayscale)
            if variance is None:
                return None

            return ImageBlurScore(laplacian_variance=variance)
        finally:
            image.close()
