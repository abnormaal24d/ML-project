"""Typed media outputs from preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from preprocessing.preprocessing_quality import PreprocessingQualityResult
    from preprocessing.privacy.clearance import PrivacyClearance


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TimedTranscriptSegment:
    text: str
    start_seconds: float | None
    end_seconds: float | None
    confidence: float | None = None
    speaker_id: str | None = None
    language: str | None = None
    overlap: bool = False
    words: tuple[TranscriptWord, ...] = ()


@dataclass(frozen=True, slots=True)
class TimedPrivacyInterval:
    start_seconds: float
    end_seconds: float
    reason: str
    action: str = "redact"


@dataclass(frozen=True, slots=True)
class VideoKeyframe:
    frame_path: str
    timestamp_seconds: float
    frame_index: int | None = None
    shot_id: str | None = None
    ocr_text: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class VideoShot:
    shot_id: str
    start_seconds: float
    end_seconds: float
    keyframe_ids: tuple[str, ...] = ()
    scene_id: str | None = None
    quality_score: float | None = None


@dataclass(frozen=True, slots=True)
class VideoObjectTrack:
    track_id: str
    label: str
    start_seconds: float
    end_seconds: float
    frame_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalEvent:
    event_type: str
    start_seconds: float
    end_seconds: float
    confidence: float | None = None


class _CanonicalVideoStructure(TypedDict):
    """Typed result of canonical video-structure normalization."""

    keyframes: tuple[VideoKeyframe, ...]
    shots: tuple[VideoShot, ...]
    object_tracks: tuple[VideoObjectTrack, ...]
    temporal_events: tuple[TemporalEvent, ...]


def canonical_transcript_segments(
    segments: tuple[dict[str, object], ...],
) -> tuple[TimedTranscriptSegment, ...]:
    rows: list[TimedTranscriptSegment] = []
    for segment in segments:
        words: list[TranscriptWord] = []
        raw_words = segment.get("words")
        if isinstance(raw_words, (list, tuple)):
            for word in raw_words:
                if not isinstance(word, dict):
                    continue
                text = str(word.get("text") or "").strip()
                start = _float(word.get("start_seconds"))
                end = _float(word.get("end_seconds"))
                if (
                    text
                    and start is not None
                    and end is not None
                    and end >= start
                ):
                    words.append(
                        TranscriptWord(
                            text, start, end, _float(word.get("confidence"))
                        )
                    )
        rows.append(
            TimedTranscriptSegment(
                text=str(segment.get("text") or ""),
                start_seconds=_float(segment.get("start_seconds")),
                end_seconds=_float(segment.get("end_seconds")),
                confidence=_float(segment.get("confidence")),
                speaker_id=_optional_text(segment.get("speaker_id")),
                language=_optional_text(segment.get("language")),
                overlap=bool(segment.get("overlap", False)),
                words=tuple(words),
            )
        )
    return tuple(rows)


def canonical_privacy_intervals(
    value: object,
) -> tuple[TimedPrivacyInterval, ...]:
    rows: list[TimedPrivacyInterval] = []
    if not isinstance(value, (list, tuple)):
        return ()
    for raw in value:
        if not isinstance(raw, dict):
            continue
        start = _float(raw.get("start_seconds"))
        end = _float(raw.get("end_seconds"))
        reason = _optional_text(raw.get("reason"))
        if start is None or end is None or end < start or reason is None:
            continue
        rows.append(
            TimedPrivacyInterval(
                start,
                end,
                reason,
                _optional_text(raw.get("action")) or "redact",
            )
        )
    return tuple(rows)


def canonical_video_structure(
    payload: dict[str, object],
) -> _CanonicalVideoStructure:
    keyframes: list[VideoKeyframe] = []
    raw_keyframes = payload.get("keyframes", ())
    if not isinstance(raw_keyframes, (list, tuple)):
        raw_keyframes = ()
    for raw in raw_keyframes:
        if not isinstance(raw, dict):
            continue
        path = _optional_text(raw.get("frame_path"))
        timestamp = _float(raw.get("timestamp_seconds"))
        if path is None or timestamp is None:
            continue
        keyframes.append(
            VideoKeyframe(
                path,
                timestamp,
                _int_or_none(raw.get("frame_index")),
                _optional_text(raw.get("shot_id")),
                _optional_text(raw.get("ocr_text")),
                _float(raw.get("confidence")),
            )
        )
    shots: list[VideoShot] = []
    raw_shots = payload.get("shots") or payload.get("scene_boundaries")
    if isinstance(raw_shots, (list, tuple)):
        for index, raw in enumerate(raw_shots):
            if not isinstance(raw, dict):
                continue
            start = _float(raw.get("start_seconds"))
            end = _float(raw.get("end_seconds"))
            if start is None or end is None or end < start:
                continue
            ids = raw.get("keyframe_ids")
            shots.append(
                VideoShot(
                    _optional_text(raw.get("shot_id")) or f"shot:{index}",
                    start,
                    end,
                    tuple(str(v) for v in ids)
                    if isinstance(ids, (list, tuple))
                    else (),
                    _optional_text(raw.get("scene_id")),
                    _float(raw.get("quality_score")),
                )
            )
    tracks: list[VideoObjectTrack] = []
    raw_tracks = payload.get("object_tracks")
    if isinstance(raw_tracks, (list, tuple)):
        for index, raw in enumerate(raw_tracks):
            if not isinstance(raw, dict):
                continue
            start, end = (
                _float(raw.get("start_seconds")),
                _float(raw.get("end_seconds")),
            )
            label = _optional_text(raw.get("label"))
            if start is None or end is None or end < start or label is None:
                continue
            frames = raw.get("frame_indices")
            tracks.append(
                VideoObjectTrack(
                    _optional_text(raw.get("track_id")) or f"track:{index}",
                    label,
                    start,
                    end,
                    tuple(int(v) for v in frames)
                    if isinstance(frames, (list, tuple))
                    else (),
                )
            )
    events: list[TemporalEvent] = []
    raw_events = payload.get("temporal_events")
    if isinstance(raw_events, (list, tuple)):
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            start, end = (
                _float(raw.get("start_seconds")),
                _float(raw.get("end_seconds")),
            )
            kind = _optional_text(raw.get("event_type"))
            if start is not None and end is not None and end >= start and kind:
                events.append(
                    TemporalEvent(
                        kind, start, end, _float(raw.get("confidence"))
                    )
                )
    return {
        "keyframes": tuple(keyframes),
        "shots": tuple(shots),
        "object_tracks": tuple(tracks),
        "temporal_events": tuple(events),
    }


def canonical_time_intervals(value: object) -> tuple[tuple[float, float], ...]:
    rows: list[tuple[float, float]] = []
    if not isinstance(value, (list, tuple)):
        return ()
    for raw in value:
        if isinstance(raw, dict):
            start, end = (
                _float(raw.get("start_seconds")),
                _float(raw.get("end_seconds")),
            )
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            start, end = _float(raw[0]), _float(raw[1])
        else:
            continue
        if start is not None and end is not None and end >= start:
            rows.append((start, end))
    return tuple(rows)


def _float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    """Validated image preprocessing result used by curation/training."""

    media_id: str
    source_id: str
    source_url: str
    normalized_url: str
    domain: str
    media_path: str
    mime_type: str | None
    width: int | None
    height: int | None
    normalized_media_path: str | None
    ocr_text: str | None
    ocr_confidence: float | None
    ocr_language: str | None
    ocr_quality_score: float | None
    quality: PreprocessingQualityResult
    safety_status: str = "unchecked"
    license: str | None = None
    dedupe_fingerprints: dict[str, str] = field(default_factory=dict)
    alignment_signals: dict[str, object] = field(default_factory=dict)
    privacy_clearance: PrivacyClearance | None = None
    privacy_evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreprocessedAudio:
    """Validated audio preprocessing result used by curation/training."""

    media_id: str
    source_id: str
    source_url: str
    normalized_url: str
    domain: str
    media_path: str
    mime_type: str | None
    duration_seconds: float | None
    transcript_text: str | None
    transcript_language: str | None
    transcript_segments: tuple[dict[str, object], ...]
    quality: PreprocessingQualityResult
    normalized_audio_path: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    loudness_lufs: float | None = None
    safety_status: str = "unchecked"
    license: str | None = None
    dedupe_fingerprints: dict[str, str] = field(default_factory=dict)
    alignment_signals: dict[str, object] = field(default_factory=dict)
    privacy_clearance: PrivacyClearance | None = None
    privacy_evidence: dict[str, object] = field(default_factory=dict)
    timed_segments: tuple[TimedTranscriptSegment, ...] = ()
    voice_activity_intervals: tuple[tuple[float, float], ...] = ()
    privacy_intervals: tuple[TimedPrivacyInterval, ...] = ()
    snr_db: float | None = None
    speech_ratio: float | None = None
    music_ratio: float | None = None
    overlap_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class PreprocessedVideo:
    """Validated video preprocessing result used by curation/training."""

    media_id: str
    source_id: str
    source_url: str
    normalized_url: str
    domain: str
    media_path: str
    mime_type: str | None
    duration_seconds: float | None
    width: int | None
    height: int | None
    transcript_text: str | None
    transcript_language: str | None
    transcript_segments: tuple[dict[str, object], ...]
    frame_ocr_text: str | None
    keyframe_paths: tuple[str, ...]
    quality: PreprocessingQualityResult
    normalized_video_path: str | None = None
    video_probe_metadata: dict[str, object] = field(default_factory=dict)
    safety_status: str = "unchecked"
    license: str | None = None
    dedupe_fingerprints: dict[str, str] = field(default_factory=dict)
    alignment_signals: dict[str, object] = field(default_factory=dict)
    privacy_clearance: PrivacyClearance | None = None
    privacy_evidence: dict[str, object] = field(default_factory=dict)
    timed_segments: tuple[TimedTranscriptSegment, ...] = ()
    keyframes: tuple[VideoKeyframe, ...] = ()
    shots: tuple[VideoShot, ...] = ()
    object_tracks: tuple[VideoObjectTrack, ...] = ()
    temporal_events: tuple[TemporalEvent, ...] = ()
    subtitle_segments: tuple[TimedTranscriptSegment, ...] = ()
    privacy_intervals: tuple[TimedPrivacyInterval, ...] = ()
