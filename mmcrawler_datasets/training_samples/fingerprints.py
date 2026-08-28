"""Typed training content fingerprint contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .common import _require_sha256


@dataclass(frozen=True, slots=True)
class ContentFingerprints:
    """Typed multimodal duplicate and leakage evidence."""

    emitted_text_sha256: str | None
    normalized_text_sha256: str | None
    text_shingle_profile: tuple[str, ...] | None
    image_ahash: str | None
    image_dhash: str | None
    image_phash: str | None
    audio_chromaprint: str | None
    video_keyframe_phashes: tuple[str, ...] | None
    document_layout_sha256: str | None
    document_page_phashes: tuple[str, ...] | None

    def __post_init__(self) -> None:
        for name in (
            "emitted_text_sha256",
            "normalized_text_sha256",
            "image_ahash",
            "image_dhash",
            "image_phash",
            "audio_chromaprint",
            "document_layout_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise TypeError(f"fingerprint {name} must be non-empty text")
        for name in (
            "text_shingle_profile",
            "video_keyframe_phashes",
            "document_page_phashes",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, tuple)
                or not all(
                    isinstance(item, str) and bool(item.strip())
                    for item in value
                )
            ):
                raise TypeError(
                    f"fingerprint {name} must be an immutable text tuple"
                )
        for name in (
            "emitted_text_sha256",
            "normalized_text_sha256",
            "document_layout_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(value, field_name=name)

    def populated(self) -> dict[str, object]:
        return {
            name: value
            for name, value in asdict(self).items()
            if value is not None
        }

    def normalized_items(self) -> tuple[tuple[str, str], ...]:
        """Return stable scalar keys for string and tuple fingerprints."""

        normalized: list[tuple[str, str]] = []
        for name, value in self.populated().items():
            if isinstance(value, str):
                scalar = value.strip().casefold()
                if scalar and scalar not in {"unknown", "none"}:
                    normalized.append((name, scalar))
                continue
            if isinstance(value, tuple):
                items = tuple(
                    item.strip().casefold()
                    for item in value
                    if isinstance(item, str) and item.strip()
                )
                if items:
                    normalized.append((name, "|".join(items)))
        return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ContentFingerprintInputs:
    """Verified modality evidence consumed by the sole fingerprint builder."""

    image_ahash: str | None = None
    image_dhash: str | None = None
    image_phash: str | None = None
    audio_chromaprint: str | None = None
    video_keyframe_phashes: tuple[str, ...] | None = None
    document_page_phashes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "image_ahash",
            "image_dhash",
            "image_phash",
            "audio_chromaprint",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise TypeError(
                    f"fingerprint input {name} must be non-empty text"
                )
        for name in (
            "video_keyframe_phashes",
            "document_page_phashes",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, tuple)
                or not all(
                    isinstance(item, str) and bool(item.strip())
                    for item in value
                )
            ):
                raise TypeError(
                    f"fingerprint input {name} must be an immutable text tuple"
                )
