"""Data models for speaker diarization and segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from preprocessing.provenance import ProducerProvenance


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """A speaker turn with timing and metadata."""

    speaker_id: str
    start_seconds: float
    end_seconds: float
    confidence: float | None = None
    overlapping_speech: bool = False
    # Optional additional metadata
    embedding: list[float] | None = (
        None  # speaker embedding vector if available
    )


@dataclass(frozen=True, slots=True)
class SpeakerDiarizationResult:
    """Full result of speaker diarization analysis."""

    segments: tuple[SpeakerSegment, ...]
    speaker_count: int
    overlapping_speech: bool
    model_name: str | None = None
    model_version: str | None = None
    provenance: ProducerProvenance | None = None
    # Global speaker mapping if identity is known
    speaker_identities: dict[str, str] | None = (
        None  # local_id -> global_identity
    )


def _validate_segments(segments: list[dict[str, Any]]) -> list[SpeakerSegment]:
    """Convert raw dicts to validated SpeakerSegment objects."""
    validated = []
    for seg in segments:
        try:
            raw_speaker_id = seg.get("speaker_id")
            raw_start = seg.get("start_seconds")
            raw_end = seg.get("end_seconds")
            if (
                not isinstance(raw_speaker_id, str)
                or not raw_speaker_id.strip()
                or raw_start is None
                or raw_end is None
            ):
                continue
            speaker_id = raw_speaker_id.strip()
            start = float(raw_start)
            end = float(raw_end)
            raw_confidence = seg.get("confidence")
            conf = (
                float(raw_confidence) if raw_confidence is not None else None
            )
            overlap = bool(seg.get("overlapping_speech", False))
            emb = seg.get("embedding")
            if emb and isinstance(emb, (list, tuple)):
                emb = [float(x) for x in emb]
            else:
                emb = None
            validated.append(
                SpeakerSegment(
                    speaker_id=speaker_id,
                    start_seconds=max(0.0, start),
                    end_seconds=max(start + 0.01, end),
                    confidence=conf,
                    overlapping_speech=overlap,
                    embedding=emb,
                )
            )
        except (ValueError, TypeError):
            continue
    # Sort by start time
    validated.sort(key=lambda s: s.start_seconds)
    return validated
