"""Normalize transcript segments produced by media analyzers."""

from __future__ import annotations

from preprocessing.media.media_input_validation import (
    as_optional_float as _as_optional_float,
)
from preprocessing.media.media_input_validation import (
    as_optional_text as _as_optional_text,
)

__all__ = (
    "estimate_text_tokens",
    "normalize_segments",
    "summarize_timeline",
)


def estimate_text_tokens(*, text: str) -> int:
    """Estimate token count for short transcript and OCR snippets."""

    return len([token for token in text.split() if token])


def normalize_segments(value: object) -> tuple[dict[str, object], ...]:
    """Normalize analyzer transcript segments into stable dictionaries."""

    if not isinstance(value, (list, tuple)):
        return ()

    cleaned: list[dict[str, object]] = []
    seen: set[tuple[str, float | None, float | None]] = set()
    for segment in value:
        if not isinstance(segment, dict):
            continue
        text = _as_optional_text(segment.get("text"))
        if text is None:
            continue
        start = _as_optional_float(segment.get("start_seconds"))
        end = _as_optional_float(segment.get("end_seconds"))
        if start is not None and start < 0:
            continue
        if end is not None and end < 0:
            continue
        if start is not None and end is not None and end < start:
            continue
        confidence = _as_optional_float(segment.get("confidence"))
        if confidence is not None and confidence < 0.01:
            continue
        key = (text.casefold(), start, end)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "text": text,
                "source": _as_optional_text(segment.get("source"))
                or "transcript_segment",
                "start_seconds": start,
                "end_seconds": end,
                "confidence": confidence,
                "speaker_id": _as_optional_text(segment.get("speaker_id")),
                "language": _as_optional_text(segment.get("language")),
                "overlap": bool(segment.get("overlap", False)),
                "words": _normalize_words(segment.get("words")),
            }
        )

    return tuple(
        sorted(
            cleaned,
            key=lambda item: (
                item.get("start_seconds") is None,
                item.get("start_seconds") or 0.0,
                item.get("end_seconds") or 0.0,
            ),
        )
    )


def summarize_timeline(
    *,
    segments: tuple[dict[str, object], ...],
    duration_seconds: float | None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Summarize transcript timing and analyzer evidence without new schema."""

    timed: list[tuple[float, float]] = []
    for segment in segments:
        start = _as_optional_float(segment.get("start_seconds"))
        end = _as_optional_float(segment.get("end_seconds"))
        if start is not None and end is not None:
            timed.append((start, end))
    covered_seconds = sum(max(0.0, end - start) for start, end in timed)
    starts = [start for start, _end in timed]
    ends = [end for _start, end in timed]
    confidences = [
        confidence
        for segment in segments
        if (confidence := _as_optional_float(segment.get("confidence")))
        is not None
    ]
    source = payload or {}
    summary: dict[str, object] = {
        "segment_count": len(segments),
        "timed_segment_count": len(timed),
        "timeline_start_seconds": min(starts) if starts else None,
        "timeline_end_seconds": max(ends) if ends else None,
        "timed_coverage_seconds": round(covered_seconds, 4),
        "timed_coverage_ratio": (
            round(min(1.0, covered_seconds / duration_seconds), 4)
            if duration_seconds is not None and duration_seconds > 0.0
            else None
        ),
        "word_count": sum(
            estimate_text_tokens(text=str(segment.get("text") or ""))
            for segment in segments
        ),
        "mean_segment_confidence": (
            round(sum(confidences) / len(confidences), 4)
            if confidences
            else None
        ),
        "word_timestamps_available": bool(source.get("word_timestamps")),
        "diarization_available": bool(
            source.get("diarization") or source.get("speaker_segments")
        ),
        "language_segments_available": bool(source.get("language_segments")),
    }
    for key in (
        "speaker_count",
        "snr_db",
        "speech_ratio",
        "music_ratio",
        "overlap_ratio",
    ):
        value = source.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            summary[key] = value
    return summary


def _normalize_words(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    rows: list[dict[str, object]] = []
    for word in value:
        if not isinstance(word, dict):
            continue
        text = _as_optional_text(word.get("text"))
        start = _as_optional_float(word.get("start_seconds"))
        end = _as_optional_float(word.get("end_seconds"))
        if text is None or start is None or end is None or end < start:
            continue
        rows.append(
            {
                "text": text,
                "start_seconds": start,
                "end_seconds": end,
                "confidence": _as_optional_float(word.get("confidence")),
            }
        )
    return tuple(rows)
