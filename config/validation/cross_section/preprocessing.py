"""Cross-section preprocessing configuration rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings.root import Settings


def _validate_video_transcription_flags(settings: Settings) -> None:
    """Fail closed when video transcription intent cannot execute."""

    video = settings.collection.processors.video
    if not video.run_transcription:
        return
    if not video.generate_transcriptions:
        raise ValueError(
            "collection.processors.video.run_transcription=true requires "
            "collection.processors.video.generate_transcriptions=true"
        )
    if not video.extract_audio_track:
        raise ValueError(
            "collection.processors.video.run_transcription=true requires "
            "collection.processors.video.extract_audio_track=true"
        )


def _validate_preprocessing_configuration(settings: Settings) -> None:
    """Validate relationships between collection intent and preprocessing."""

    processors = settings.collection.processors
    transcription_requested = bool(
        processors.audio.run_transcription
        or processors.video.run_transcription
    )
    if transcription_requested != settings.preprocessing.transcription.enabled:
        raise ValueError(
            "collection processor run_transcription settings must match "
            "preprocessing.transcription.enabled"
        )
