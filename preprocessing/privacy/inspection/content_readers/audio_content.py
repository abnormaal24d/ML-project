"""Audio inspection input with explicit, fail-closed local evidence."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str
    speaker_id: str | None


@dataclass(frozen=True, slots=True)
class AudioContent:
    subject_bytes: bytes
    duration_ms: int
    transcript_segments: tuple[TranscriptSegment, ...]
    transcript_checked_ranges_ms: tuple[tuple[int, int], ...]
    transcript_analysis_completed: bool
    metadata: dict[str, str]
    metadata_analysis_completed: bool
    language: str | None
    country: str | None
    full_decode_completed: bool
    speaker_analysis_completed: bool
    background_speech_analysis_completed: bool
    voice_analysis_completed: bool
    voice_identity_detected: bool
    voice_identity_authorized: bool
    audio_fingerprint: str | None
    detector_versions: dict[str, str] = field(default_factory=dict)
    analysis_errors: tuple[str, ...] = ()
