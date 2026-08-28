"""Validation helpers for audio augmentation inputs and outputs."""

from __future__ import annotations

import mimetypes
import struct
import wave
from pathlib import Path

from augmentation.outcomes.media_validation_outcome import (
    MediaValidationOutcome,
)


def validate_audio_input(
    *,
    path: Path,
    declared_mime_type: str | None,
    declared_byte_size: int | None,
    allowed_mime_types: frozenset[str],
    max_input_bytes: int,
) -> MediaValidationOutcome:
    """Validate audio existence, byte size, and supported MIME type."""

    signals: dict[str, object] = {
        "path": path.as_posix(),
        "declared_mime_type": declared_mime_type,
        "declared_byte_size": declared_byte_size,
        "exists": path.exists(),
    }
    if not path.is_file():
        return _rejected("missing_audio_file", signals=signals)
    byte_size = path.stat().st_size
    signals["byte_size"] = byte_size
    if byte_size <= 0 or byte_size > max_input_bytes:
        return _rejected("invalid_audio_size", signals=signals)
    if declared_byte_size is not None and declared_byte_size != byte_size:
        signals["declared_byte_size_mismatch"] = True
    mime_type = declared_mime_type or mimetypes.guess_type(path)[0]
    signals["mime_type"] = mime_type
    if mime_type not in allowed_mime_types:
        return _rejected("unsupported_audio_mime_type", signals=signals)
    return MediaValidationOutcome(None, signals)


def validate_audio_output(
    *,
    path: Path,
    expected_sample_rate: int,
    expected_channels: int,
    expected_duration_seconds: float,
    output_max_bytes: int,
    max_clipping_fraction: float,
    duration_tolerance_seconds: float,
) -> MediaValidationOutcome:
    """Decode and strictly validate generated 16-bit PCM WAV output."""

    signals: dict[str, object] = {
        "path": path.as_posix(),
        "exists": path.exists(),
        "expected_sample_rate": expected_sample_rate,
        "expected_channels": expected_channels,
        "expected_duration_seconds": expected_duration_seconds,
    }
    if not path.is_file():
        return _rejected("missing_generated_audio", signals=signals)
    byte_size = path.stat().st_size
    signals["byte_size"] = byte_size
    if byte_size <= 44 or byte_size > output_max_bytes:
        return _rejected("generated_audio_size_invalid", signals=signals)

    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frame_count = reader.getnframes()
            compression_type = reader.getcomptype()
            duration = float(frame_count) / sample_rate if sample_rate else 0.0
            clipped = 0
            total = 0
            peak = 0
            while chunk := reader.readframes(65_536):
                if len(chunk) % 2:
                    return _rejected(
                        "generated_audio_frame_alignment_invalid",
                        signals=signals,
                    )
                count = len(chunk) // 2
                values = struct.unpack(f"<{count}h", chunk)
                total += count
                for value in values:
                    absolute = abs(int(value))
                    peak = max(peak, absolute)
                    if value in {-32768, 32767}:
                        clipped += 1
    except (OSError, EOFError, wave.Error) as exc:
        signals["decode_error"] = type(exc).__name__
        return _rejected("generated_audio_decode_failed", signals=signals)

    clipping_fraction = float(clipped) / total if total else 0.0
    signals.update(
        {
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width": sample_width,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "compression_type": compression_type,
            "peak_absolute_sample": peak,
            "clipped_sample_count": clipped,
            "total_samples": total,
            "clipping_fraction": clipping_fraction,
        }
    )
    if compression_type != "NONE" or sample_width != 2:
        return _rejected(
            "generated_audio_pcm_contract_invalid", signals=signals
        )
    if sample_rate != expected_sample_rate:
        return _rejected(
            "generated_audio_sample_rate_mismatch", signals=signals
        )
    if channels != expected_channels:
        return _rejected("generated_audio_channel_mismatch", signals=signals)
    if frame_count < 1 or duration <= 0.0:
        return _rejected("generated_audio_duration_invalid", signals=signals)
    tolerance = max(duration_tolerance_seconds, 1.0 / sample_rate)
    if abs(duration - expected_duration_seconds) > tolerance:
        return _rejected("generated_audio_duration_mismatch", signals=signals)
    if clipping_fraction > max_clipping_fraction:
        return _rejected("generated_audio_clipping_excessive", signals=signals)
    return MediaValidationOutcome(None, signals)


def _rejected(
    reason: str,
    *,
    signals: dict[str, object],
) -> MediaValidationOutcome:
    return MediaValidationOutcome(reason, signals)
