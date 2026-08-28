"""ASR quality scoring and training-label eligibility rules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from preprocessing.media.speech.transcription_result import (
        TranscriptSegment,
    )


def score_transcript_quality(
    *,
    transcript_text: str | None,
    transcript_segments: Sequence[TranscriptSegment],
    avg_logprob: float | None = None,
    no_speech_probability: float | None = None,
    compression_ratio: float | None = None,
    language_probability: float | None = None,
    audio_duration: float | None = None,
    vad_ratio: float | None = None,
    language: str | None = None,
    expected_language: str | None = None,
) -> tuple[float, str | None]:
    """Return a bounded quality score and stable reject reason."""

    if not transcript_text or not transcript_text.strip():
        return 0.0, "asr_empty"

    text = transcript_text.strip().lower()
    words = text.split()
    if len(words) > 4 and len(set(words)) / len(words) < 0.3:
        return 0.1, "asr_repeated_text_loop"

    if (
        expected_language
        and language
        and expected_language.casefold() != language.casefold()
        and (language_probability or 0.0) >= 0.8
    ):
        return 0.1, "asr_language_mismatch"

    resolved_logprob = _first_or_average(
        avg_logprob,
        tuple(segment.avg_logprob for segment in transcript_segments),
    )
    resolved_no_speech = _first_or_average(
        no_speech_probability,
        tuple(
            segment.no_speech_probability for segment in transcript_segments
        ),
    )
    resolved_compression = _first_or_average(
        compression_ratio,
        tuple(segment.compression_ratio for segment in transcript_segments),
    )

    if resolved_no_speech is not None and resolved_no_speech > 0.95:
        return 0.05, "asr_too_much_silence"

    score = 0.5
    if audio_duration is not None and audio_duration > 0.0:
        chars_per_second = len(text) / audio_duration
        if chars_per_second < 1.0 or chars_per_second > 30.0:
            score -= 0.15

    if resolved_logprob is not None:
        logprob_confidence = max(0.0, min(1.0, resolved_logprob + 1.0))
        score = 0.4 * logprob_confidence + 0.6 * score

    if resolved_no_speech is not None and resolved_no_speech > 0.8:
        score -= 0.25
    if resolved_compression is not None and resolved_compression > 2.5:
        score -= 0.1
    if language_probability is not None:
        score = 0.2 * language_probability + 0.8 * score
    if vad_ratio is not None and vad_ratio > 0.7:
        score -= 0.1
    if is_boilerplate(text):
        score -= 0.2

    score = round(max(0.0, min(1.0, score)), 4)
    return (score, "asr_low_confidence") if score < 0.25 else (score, None)


def training_label_rules(
    *,
    quality_score: float,
    reject_reason: str | None,
    minimum_quality: float,
) -> tuple[bool, float]:
    """Map quality evidence to explicit label eligibility and weight."""

    eligible = reject_reason is None and quality_score >= minimum_quality
    return eligible, quality_score if eligible else 0.0


def is_boilerplate(transcript: str) -> bool:
    normalized = transcript.lower().strip()
    boilerplate = {
        "thank you",
        "thanks for watching",
        "like and subscribe",
        "the end",
        "[music]",
    }
    return normalized in boilerplate or len(normalized) < 5


def _first_or_average(
    explicit: float | None,
    values: tuple[float | None, ...],
) -> float | None:
    if explicit is not None:
        return explicit
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None
