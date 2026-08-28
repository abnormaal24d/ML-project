"""Modality-specific transport and preprocessing acceptance rules."""

from __future__ import annotations

from pydantic import Field

from config.base.settings_model import SettingsModel

DEFAULT_PAGE_MAX_BYTES = 8_000_000
DEFAULT_FEED_FETCH_MAX_BYTES = 10_000_000
DEFAULT_FEED_PREPROCESSING_MAX_BYTES = 25_000_000
DEFAULT_IMAGE_MAX_BYTES = 25_000_000
DEFAULT_AUDIO_MAX_BYTES = 16_000_000
DEFAULT_VIDEO_MAX_BYTES = 25_000_000
DEFAULT_DOCUMENT_MAX_BYTES = 100_000_000
DEFAULT_MAX_DECODE_PIXELS = 40_000_000


class ModalityAcceptanceSettings(SettingsModel):
    """Configurable acceptance rules for one crawler modality."""

    fetch_max_bytes: int = Field(ge=1)
    preprocessing_max_bytes: int = Field(ge=1)
    allow_training: bool = True
    allow_metadata_only_when_oversized: bool = False
    allow_streaming_when_oversized: bool = False
    allow_partial_when_oversized: bool = False


class ImageAcceptanceSettings(ModalityAcceptanceSettings):
    """Image-specific acceptance settings including decode safety limits."""

    max_decode_pixels: int = Field(default=DEFAULT_MAX_DECODE_PIXELS, ge=1)


def _default_page_acceptance() -> ModalityAcceptanceSettings:
    return ModalityAcceptanceSettings(
        fetch_max_bytes=DEFAULT_PAGE_MAX_BYTES,
        preprocessing_max_bytes=DEFAULT_PAGE_MAX_BYTES,
    )


def _default_feed_acceptance() -> ModalityAcceptanceSettings:
    return ModalityAcceptanceSettings(
        fetch_max_bytes=DEFAULT_FEED_FETCH_MAX_BYTES,
        preprocessing_max_bytes=DEFAULT_FEED_PREPROCESSING_MAX_BYTES,
    )


def _default_image_acceptance() -> ImageAcceptanceSettings:
    return ImageAcceptanceSettings(
        fetch_max_bytes=DEFAULT_IMAGE_MAX_BYTES,
        preprocessing_max_bytes=DEFAULT_IMAGE_MAX_BYTES,
    )


def _default_audio_acceptance() -> ModalityAcceptanceSettings:
    return ModalityAcceptanceSettings(
        fetch_max_bytes=DEFAULT_AUDIO_MAX_BYTES,
        preprocessing_max_bytes=DEFAULT_AUDIO_MAX_BYTES,
        allow_metadata_only_when_oversized=True,
        allow_partial_when_oversized=True,
    )


def _default_video_acceptance() -> ModalityAcceptanceSettings:
    return ModalityAcceptanceSettings(
        fetch_max_bytes=DEFAULT_VIDEO_MAX_BYTES,
        preprocessing_max_bytes=DEFAULT_VIDEO_MAX_BYTES,
        allow_metadata_only_when_oversized=True,
        allow_partial_when_oversized=True,
    )


def _default_document_acceptance() -> ModalityAcceptanceSettings:
    return ModalityAcceptanceSettings(
        fetch_max_bytes=DEFAULT_DOCUMENT_MAX_BYTES,
        preprocessing_max_bytes=DEFAULT_DOCUMENT_MAX_BYTES,
    )


class ModalityAcceptanceSettingsCatalog(SettingsModel):
    """Explicit transport and preprocessing rules for every media kind."""

    page: ModalityAcceptanceSettings = Field(
        default_factory=_default_page_acceptance
    )
    feed: ModalityAcceptanceSettings = Field(
        default_factory=_default_feed_acceptance
    )
    image: ImageAcceptanceSettings = Field(
        default_factory=_default_image_acceptance
    )
    audio: ModalityAcceptanceSettings = Field(
        default_factory=_default_audio_acceptance
    )
    video: ModalityAcceptanceSettings = Field(
        default_factory=_default_video_acceptance
    )
    document: ModalityAcceptanceSettings = Field(
        default_factory=_default_document_acceptance
    )
