"""Tests for objective AudioPayloadExtractor (no transcription / scoring)."""

from __future__ import annotations

import math
import wave
from io import BytesIO

from crawler.extraction.payloads.audio_payload_extractor import (
    AudioPayloadExtractionResult,
    AudioPayloadExtractor,
)


def _wav_bytes(
    *,
    sample_rate: int = 8_000,
    channels: int = 1,
    duration_seconds: float = 0.25,
    sample_width: int = 2,
) -> bytes:
    frame_count = int(sample_rate * duration_seconds)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        # Silent PCM frames.
        frame = b"\x00" * (sample_width * channels)
        writer.writeframes(frame * frame_count)
    return buffer.getvalue()


def test_extract_wav_objective_fields() -> None:
    body = _wav_bytes(sample_rate=16_000, channels=2, duration_seconds=0.5)
    result = AudioPayloadExtractor().extract(body=body)
    assert isinstance(result, AudioPayloadExtractionResult)
    assert result.format == "WAV"
    assert result.sample_rate == 16_000
    assert result.channels == 2
    assert result.duration_seconds is not None
    assert math.isclose(result.duration_seconds, 0.5, rel_tol=0.05)
    assert result.bitrate == 16_000 * 2 * 2 * 8
    assert result.byte_size == len(body)
    assert len(result.sha256) == 64
    assert result.sha256 == __import__("hashlib").sha256(body).hexdigest()


def test_extract_empty_body_returns_none() -> None:
    assert AudioPayloadExtractor().extract(body=b"") is None


def test_extract_garbage_returns_none() -> None:
    assert AudioPayloadExtractor().extract(body=b"not-audio-data") is None


def test_result_has_no_transcript_or_score_fields() -> None:
    body = _wav_bytes()
    result = AudioPayloadExtractor().extract(body=body)
    assert result is not None
    fields = set(result.__dataclass_fields__)
    assert "transcript" not in fields
    assert "extracted_text" not in fields
    assert "quality_score" not in fields
    assert "events" not in fields


def test_mono_short_wav() -> None:
    body = _wav_bytes(sample_rate=8_000, channels=1, duration_seconds=0.1)
    result = AudioPayloadExtractor().extract(body=body)
    assert result is not None
    assert result.channels == 1
    assert result.sample_rate == 8_000
