"""WAV and SoundFile audio decode backends for augmentation."""

from __future__ import annotations

import io
import statistics
import struct
import wave
from collections.abc import Callable, Iterator
from importlib import import_module
from pathlib import Path

from preprocessing.media.ports import DecodedAudio


class WaveAudioDecodeBackend:
    """Decode PCM WAV files with the stdlib ``wave`` module."""

    def decode(
        self,
        *,
        path: Path,
        chunk_frames: int,
    ) -> DecodedAudio:
        handle = wave.open(str(path), "rb")
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        duration = (
            float(frame_count) / float(sample_rate) if sample_rate else 0.0
        )
        chunk = max(1, int(chunk_frames))

        def _frames() -> Iterator[bytes]:
            try:
                while True:
                    data = handle.readframes(chunk)
                    if not data:
                        break
                    yield data
            finally:
                handle.close()

        return DecodedAudio(
            channels=channels,
            sample_width=sample_width,
            sample_rate=sample_rate,
            duration_sec=duration,
            frames_iterator=_frames(),
        )


class SoundFileAudioDecodeBackend:
    """Decode non-WAV audio via SoundFile into PCM chunks."""

    def decode(
        self,
        *,
        path: Path,
        chunk_frames: int,
    ) -> DecodedAudio:
        try:
            import soundfile as sf  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("soundfile_unavailable") from exc

        with sf.SoundFile(str(path), mode="r") as handle:
            channels = int(handle.channels)
            sample_rate = int(handle.samplerate)
            frames = int(handle.frames)
            duration = (
                float(frames) / float(sample_rate) if sample_rate else 0.0
            )
            data = handle.read(dtype="int16", always_2d=True)

        chunk = max(1, int(chunk_frames))

        def _frames() -> Iterator[bytes]:
            total = data.shape[0]
            for start in range(0, total, chunk):
                block = data[start : start + chunk]
                yield block.tobytes()

        return DecodedAudio(
            channels=channels,
            sample_width=2,
            sample_rate=sample_rate,
            duration_sec=duration,
            frames_iterator=_frames(),
        )


class CompositeAudioDecodeBackend:
    """Prefer WAV decode, then SoundFile for other formats."""

    def __init__(
        self,
        *,
        wave_backend: WaveAudioDecodeBackend,
        soundfile_backend_factory: Callable[[], SoundFileAudioDecodeBackend]
        | type[SoundFileAudioDecodeBackend],
    ) -> None:
        self._wave = wave_backend
        self._soundfile_factory = soundfile_backend_factory

    def decode(
        self,
        *,
        path: Path,
        chunk_frames: int,
    ) -> DecodedAudio:
        suffix = path.suffix.lower()
        if suffix == ".wav":
            return self._wave.decode(path=path, chunk_frames=chunk_frames)
        backend = self._soundfile_factory()
        return backend.decode(path=path, chunk_frames=chunk_frames)


def decode_audio_samples(
    *,
    audio_bytes: bytes,
    sample_rate: int | None,
) -> tuple[tuple[float, ...], int] | None:
    """Decode audio bytes to mono samples for deterministic analysis."""

    decoded_wave = _decode_pcm_wave(audio_bytes=audio_bytes)
    if decoded_wave is not None:
        return decoded_wave
    decoded_soundfile = _decode_soundfile_audio(audio_bytes=audio_bytes)
    if decoded_soundfile is not None:
        return decoded_soundfile
    return _decode_pcm16_mono(
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
    )


def _decode_soundfile_audio(
    *, audio_bytes: bytes
) -> tuple[tuple[float, ...], int] | None:
    if not audio_bytes:
        return None
    try:
        soundfile = import_module("soundfile")
    except (ImportError, OSError):
        return None

    try:
        data, rate = soundfile.read(
            io.BytesIO(audio_bytes),
            always_2d=True,
            dtype="float32",
        )
    except (OSError, RuntimeError, ValueError):
        return None

    if rate <= 0 or len(data) <= 0:
        return None

    samples = tuple(
        float(statistics.fmean(float(value) for value in row))
        for row in data
        if len(row)
    )
    if not samples:
        return None
    return samples, int(rate)


def _decode_pcm_wave(
    *, audio_bytes: bytes
) -> tuple[tuple[float, ...], int] | None:
    if not audio_bytes:
        return None
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            frame_count = reader.getnframes()
            if (
                channels < 1
                or width not in {1, 2, 4}
                or rate <= 0
                or frame_count <= 0
            ):
                return None
            raw = reader.readframes(frame_count)
    except (EOFError, OSError, wave.Error):
        return None

    values = _unpack_pcm(raw=raw, sample_width=width, channels=channels)
    if not values:
        return None
    return values, rate


def _decode_pcm16_mono(
    *,
    audio_bytes: bytes,
    sample_rate: int | None,
) -> tuple[tuple[float, ...], int] | None:
    if sample_rate is None or sample_rate <= 0:
        return None
    if not audio_bytes or len(audio_bytes) < 2 or len(audio_bytes) % 2:
        return None

    count = len(audio_bytes) // 2
    values = tuple(
        value / 32768.0
        for value in struct.unpack(
            "<" + "h" * count,
            audio_bytes[: count * 2],
        )
    )
    if not values:
        return None
    return values, sample_rate


def _unpack_pcm(
    *, raw: bytes, sample_width: int, channels: int
) -> tuple[float, ...]:
    if sample_width == 1:
        values = [(byte - 128) / 128.0 for byte in raw]
    elif sample_width == 2:
        count = len(raw) // 2
        values = [
            value / 32768.0
            for value in struct.unpack("<" + "h" * count, raw[: count * 2])
        ]
    else:
        count = len(raw) // 4
        values = [
            value / 2147483648.0
            for value in struct.unpack("<" + "i" * count, raw[: count * 4])
        ]

    if channels == 1:
        return tuple(values)

    mono: list[float] = []
    for offset in range(0, len(values) - channels + 1, channels):
        mono.append(
            float(statistics.fmean(values[offset : offset + channels]))
        )
    return tuple(mono)
