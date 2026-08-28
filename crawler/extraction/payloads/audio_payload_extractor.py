"""Objective audio payload metadata from already-fetched bytes.

No transcription, diarization, event detection, or quality scoring.
"""

from __future__ import annotations

import hashlib
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AudioPayloadExtractionResult:
    """Deterministic properties of one audio payload."""

    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    bitrate: int | None
    format: str | None
    byte_size: int
    sha256: str


class AudioPayloadExtractor:
    """Extract objective audio payload metadata from raw bytes."""

    def extract(self, *, body: bytes) -> AudioPayloadExtractionResult | None:
        """Return payload metadata, or ``None`` when audio is unreadable.

        Empty bodies always return ``None``. When a container opens but only
        partial stream fields are available, those fields may be ``None`` while
        ``byte_size`` and ``sha256`` remain set.
        """

        if not body:
            return None

        byte_size = len(body)
        sha256 = hashlib.sha256(body).hexdigest()

        wave_meta = _read_wave_metadata(body=body)
        if wave_meta is not None:
            return AudioPayloadExtractionResult(
                duration_seconds=wave_meta["duration_seconds"],
                sample_rate=wave_meta["sample_rate"],
                channels=wave_meta["channels"],
                bitrate=wave_meta["bitrate"],
                format=wave_meta["format"],
                byte_size=byte_size,
                sha256=sha256,
            )

        mutagen_meta = _read_mutagen_metadata(body=body)
        if mutagen_meta is None:
            return None

        return AudioPayloadExtractionResult(
            duration_seconds=mutagen_meta["duration_seconds"],
            sample_rate=mutagen_meta["sample_rate"],
            channels=mutagen_meta["channels"],
            bitrate=mutagen_meta["bitrate"],
            format=mutagen_meta["format"],
            byte_size=byte_size,
            sha256=sha256,
        )


def _read_wave_metadata(*, body: bytes) -> dict[str, Any] | None:
    """Parse RIFF/WAVE via the stdlib when possible."""

    if len(body) < 12 or body[0:4] != b"RIFF" or body[8:12] != b"WAVE":
        return None

    from io import BytesIO

    try:
        with wave.open(BytesIO(body), "rb") as reader:
            channels = int(reader.getnchannels())
            sample_rate = int(reader.getframerate())
            frame_count = int(reader.getnframes())
            sample_width = int(reader.getsampwidth())
    except Exception:
        return None

    if channels < 1 or sample_rate <= 0 or frame_count < 0:
        return None

    duration = frame_count / float(sample_rate) if sample_rate else None
    bitrate = None
    if sample_rate > 0 and channels > 0 and sample_width > 0:
        # PCM bitrate in bits/second.
        bitrate = int(sample_rate * channels * sample_width * 8)

    return {
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "bitrate": bitrate,
        "format": "WAV",
    }


def _read_mutagen_metadata(*, body: bytes) -> dict[str, Any] | None:
    try:
        from mutagen._file import (  # type: ignore
            File as MutagenFile,
        )
        from mutagen._util import (  # type: ignore
            MutagenError,
        )
    except ImportError:
        return None

    suffix = _guess_suffix(body=body)
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as handle:
            handle.write(body)
            temp_path = Path(handle.name)
    except OSError:
        return None

    try:
        try:
            parsed = MutagenFile(temp_path)
        except (OSError, ValueError, RuntimeError, MutagenError):
            return None
        if parsed is None:
            return None
        info = getattr(parsed, "info", None)
        if info is None:
            return None

        duration = _as_float(getattr(info, "length", None))
        sample_rate = _as_int(getattr(info, "sample_rate", None))
        channels = _as_int(getattr(info, "channels", None))
        bitrate = _as_int(getattr(info, "bitrate", None))
        format_name = _mutagen_format_name(parsed=parsed, path=temp_path)

        if (
            duration is None
            and sample_rate is None
            and channels is None
            and bitrate is None
        ):
            return None

        return {
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "bitrate": bitrate,
            "format": format_name,
        }
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _mutagen_format_name(*, parsed: Any, path: Path) -> str | None:
    mime = getattr(parsed, "mime", None)
    if isinstance(mime, (list, tuple)) and mime:
        first = str(mime[0]).strip().lower()
        if first.startswith("audio/"):
            return first.removeprefix("audio/").upper()
        if first:
            return first.upper()

    name = type(parsed).__name__.strip()
    if name and name not in {"FileType", "File"}:
        return name.upper()

    suffix = path.suffix.lstrip(".").upper()
    return suffix or None


def _guess_suffix(*, body: bytes) -> str:
    if body.startswith(b"ID3") or (
        len(body) > 2 and body[0] == 0xFF and (body[1] & 0xE0) == 0xE0
    ):
        return ".mp3"
    if body.startswith(b"OggS"):
        return ".ogg"
    if body.startswith(b"fLaC"):
        return ".flac"
    if body[4:8] == b"ftyp" if len(body) >= 8 else False:
        return ".m4a"
    if body.startswith(b"RIFF") and body[8:12] == b"WAVE":
        return ".wav"
    return ".bin"


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number < 0:
        return None
    return number


def _as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number < 0:
        return None
    return number
