"""Collection and media acceptance cross-section rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings.root import Settings


def _validate_media_limits(
    settings: Settings,
) -> None:
    """Reject non-positive configured media byte limits.

    Limits below the recommended defaults are allowed when no enabled task
    requires that modality. The task-dependent minimums are validated
    separately by multimodal task-domain validation.
    """

    acceptance = settings.collection.modality_acceptance

    configured_limits = (
        (
            "collection.modality_acceptance.image.fetch_max_bytes",
            acceptance.image.fetch_max_bytes,
        ),
        (
            "collection.modality_acceptance.image.preprocessing_max_bytes",
            acceptance.image.preprocessing_max_bytes,
        ),
        (
            "collection.modality_acceptance.audio.fetch_max_bytes",
            acceptance.audio.fetch_max_bytes,
        ),
        (
            "collection.modality_acceptance.video.fetch_max_bytes",
            acceptance.video.fetch_max_bytes,
        ),
    )

    for field_name, value in configured_limits:
        if value is not None and int(value) <= 0:
            raise ValueError(f"{field_name} must be greater than zero or null")
