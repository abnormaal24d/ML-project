"""Canonical rich ASR output models for derived preprocessing data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from preprocessing.provenance import ProducerProvenance


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        _validate_timing(self.start_seconds, self.end_seconds)
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One timestamped ASR segment with explicit provenance and label rules."""

    text: str
    start_seconds: float
    end_seconds: float
    language: str | None
    avg_logprob: float | None
    no_speech_probability: float | None
    confidence: float | None
    provenance: ProducerProvenance
    training_label_eligible: bool
    label_weight: float
    compression_ratio: float | None = None
    speaker_id: str | None = None
    words: tuple[TranscriptWord, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("transcript segment text must not be blank")
        _validate_timing(self.start_seconds, self.end_seconds)
        _validate_confidence(self.confidence)
        _validate_confidence(self.no_speech_probability)
        _validate_label_weight(self.label_weight)
        if not self.training_label_eligible and self.label_weight != 0.0:
            raise ValueError(
                "ineligible transcript segments require label_weight=0"
            )


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Rich ASR result persisted as derived dataset data.

    The external backend ends at this schema.  It is never retained as a
    trainable encoder or checkpoint component.
    """

    text: str
    confidence: float
    language: str | None
    language_probability: float | None
    segments: tuple[TranscriptSegment, ...]
    provenance: ProducerProvenance
    avg_logprob: float | None
    no_speech_probability: float | None
    compression_ratio: float | None
    training_label_eligible: bool
    label_weight: float
    reject_reason: str | None = None
    postprocessing: tuple[str, ...] = ("collapse_whitespace",)
    decode_settings: dict[str, Any] | None = None
    # Separate statuses per requirement (media object vs derived transcript)
    video_download_status: str = "success"  # or failure
    audio_extraction_status: str = "success"  # or failure
    transcription_status: str = "empty"  # empty / failure / success

    def __post_init__(self) -> None:
        if self.transcription_status == "empty":
            return
        if not self.text.strip():
            raise ValueError(
                "transcription text must not be blank for non-empty status"
            )
        _validate_confidence(self.confidence)
        _validate_confidence(self.language_probability)
        _validate_confidence(self.no_speech_probability)
        _validate_label_weight(self.label_weight)
        if not self.training_label_eligible and self.label_weight != 0.0:
            raise ValueError(
                "ineligible transcriptions require label_weight=0"
            )
        if not self.segments:
            raise ValueError("transcription requires at least one segment")


def _validate_timing(start_seconds: float, end_seconds: float) -> None:
    if start_seconds < 0.0:
        raise ValueError("start_seconds must be non-negative")
    if end_seconds < start_seconds:
        raise ValueError(
            "end_seconds must be greater than or equal to start_seconds"
        )


def _validate_confidence(value: float | None) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(
            "probabilities and confidences must be between 0 and 1"
        )


def _validate_label_weight(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("label_weight must be between 0 and 1")
