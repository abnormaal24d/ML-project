"""Image hashing for preprocessing deduplication and near-duplicate detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.collection.processors import AverageHashSettings
    from preprocessing.media.ports import ImageLoader

_DEFAULT_HASH_SIZE = 8

_VALID_KINDS = frozenset({"average", "difference", "perceptual"})


class PillowImageHashCalculator:
    """Compute one Pillow hash with a shared load-and-swallow failure policy.

    The kind selects the algorithm; the public ``compute`` contract and
    failure semantics are identical across all three kinds.
    """

    def __init__(
        self,
        *,
        kind: str,
        image_loader: ImageLoader,
        settings: AverageHashSettings | None = None,
    ) -> None:
        if kind not in _VALID_KINDS:
            raise ValueError(f"unknown image hash kind: {kind}")
        if kind == "average" and settings is None:
            raise ValueError("average image hash settings are required")
        self._kind = kind
        self._settings = settings
        self._image_loader = image_loader

    def compute(
        self, *, body: bytes | None = None, img: Any = None
    ) -> str | None:
        if img is None:
            img = self._image_loader.open_image(body=body or b"")
        if img is None:
            return None
        try:
            if self._kind == "average":
                settings = self._settings
                if settings is None:
                    raise RuntimeError(
                        "average image hash settings unavailable"
                    )
                return _compute_average_hash(
                    img=img,
                    size=int(settings.size),
                )
            if self._kind == "difference":
                return _compute_difference_hash(img=img)
            return _compute_perceptual_hash(img=img)
        except Exception:
            return None


def _compute_difference_hash(*, img: Any) -> str:
    small = img.convert("L").resize((9, 8))
    pixels = list(small.getdata())
    bits = ""
    for row in range(8):
        for col in range(8):
            bits += (
                "1"
                if pixels[row * 9 + col] > pixels[row * 9 + col + 1]
                else "0"
            )
    return f"{int(bits, 2):016x}"


def _compute_perceptual_hash(*, img: Any) -> str:
    try:
        small = img.convert("L").resize((32, 32))
        import numpy as np

        arr = np.array(small, dtype=float)
        dct = np.abs(np.fft.fft2(arr)[:8, :8])
        med = np.median(dct)
        bits = "".join("1" if v > med else "0" for v in dct.flatten())
        return f"{int(bits, 2):016x}"
    except Exception:
        return _compute_average_hash(
            img=img,
            size=_DEFAULT_HASH_SIZE,
        )


def _compute_average_hash(*, img: Any, size: int) -> str:
    small = img.convert("L").resize((size, size))
    pixels = list(small.getdata())
    average = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel > average else "0" for pixel in pixels)
    hex_digits = (size * size + 3) // 4
    return f"{int(bits, 2):0{hex_digits}x}"


def near_duplicate_cluster_id_from_hashes(
    *,
    image_phash: str | None = None,
    image_dhash: str | None = None,
    image_ahash: str | None = None,
) -> str | None:
    h = image_phash or image_dhash or image_ahash
    if not h:
        return None
    return f"img-cluster:{h}"
