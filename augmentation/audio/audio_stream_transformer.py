"""Deterministic PCM audio transforms with a shared timeline receipt."""

from __future__ import annotations

import math
import os
import wave
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from augmentation.audio.audio_operations import AudioTransformParameters

if TYPE_CHECKING:
    from preprocessing.media.ports import DecodedAudio

_INT16_MAX = 32767.0
_INT16_MIN = -32768.0


@dataclass(frozen=True, slots=True)
class AudioTransformReceipt:
    """Measured source/output properties and the exact timeline mapping."""

    input_sample_rate: int
    output_sample_rate: int
    input_channels: int
    output_channels: int
    input_frame_count: int
    output_frame_count: int
    input_duration_seconds: float
    trimmed_duration_seconds: float
    output_duration_seconds: float
    trim_start_seconds: float
    trim_end_seconds: float
    speed_factor: float
    peak_absolute_sample: int
    clipped_sample_count: int
    total_output_samples: int
    clipping_fraction: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe lineage payload."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparedAudioTransform:
    """Transformed interleaved PCM bytes plus its verified receipt."""

    pcm_bytes: bytes
    receipt: AudioTransformReceipt


def prepare_audio_transform(
    *,
    decoded_audio: DecodedAudio,
    parameters: AudioTransformParameters,
) -> PreparedAudioTransform:
    """Decode, trim, convert channels, resample, perturb speed, and mix."""

    np = _require_numpy()
    source = _read_pcm_int16(decoded_audio=decoded_audio)
    input_frames = int(source.shape[0])
    if input_frames < 1:
        raise ValueError("decoded_audio_empty")

    trim_start_frame, trim_end_frame = _trim_bounds(
        samples=source,
        sample_rate=decoded_audio.sample_rate,
        enabled=parameters.trim_silence,
        threshold_dbfs=parameters.trim_silence_threshold_dbfs,
        padding_seconds=parameters.trim_padding_seconds,
    )
    trimmed = source[trim_start_frame:trim_end_frame]
    if trimmed.shape[0] < 1:
        raise ValueError("audio_trim_removed_all_frames")

    output_channels = (
        parameters.target_channels
        if parameters.convert_channels
        else decoded_audio.channels
    )
    converted = _convert_channels(trimmed, target_channels=output_channels)

    output_sample_rate = (
        parameters.target_sample_rate
        if parameters.normalize_sample_rate
        else decoded_audio.sample_rate
    )
    speed_factor = parameters.speed_factor if parameters.perturb_speed else 1.0
    resampled = _resample_with_speed(
        samples=converted,
        input_sample_rate=decoded_audio.sample_rate,
        output_sample_rate=output_sample_rate,
        speed_factor=speed_factor,
    )

    transformed = resampled
    if parameters.shift_gain:
        transformed = transformed * (10.0 ** (parameters.gain_db / 20.0))
    if parameters.inject_noise and parameters.noise_std_fraction > 0.0:
        generator = np.random.default_rng(parameters.noise_seed)
        noise = generator.normal(
            loc=0.0,
            scale=parameters.noise_std_fraction * _INT16_MAX,
            size=transformed.shape,
        )
        transformed = transformed + noise

    clipped_count = int(
        np.count_nonzero(
            (transformed > _INT16_MAX) | (transformed < _INT16_MIN)
        )
    )
    clipped = np.clip(np.rint(transformed), _INT16_MIN, _INT16_MAX).astype(
        "<i2", copy=False
    )
    peak = int(np.max(np.abs(clipped.astype(np.int32)))) if clipped.size else 0
    total_samples = int(clipped.size)
    clipping_fraction = (
        float(clipped_count) / float(total_samples) if total_samples else 0.0
    )
    output_frames = int(clipped.shape[0])

    receipt = AudioTransformReceipt(
        input_sample_rate=decoded_audio.sample_rate,
        output_sample_rate=output_sample_rate,
        input_channels=decoded_audio.channels,
        output_channels=output_channels,
        input_frame_count=input_frames,
        output_frame_count=output_frames,
        input_duration_seconds=float(input_frames) / decoded_audio.sample_rate,
        trimmed_duration_seconds=float(trimmed.shape[0])
        / decoded_audio.sample_rate,
        output_duration_seconds=float(output_frames) / output_sample_rate,
        trim_start_seconds=float(trim_start_frame) / decoded_audio.sample_rate,
        trim_end_seconds=float(trim_end_frame) / decoded_audio.sample_rate,
        speed_factor=speed_factor,
        peak_absolute_sample=peak,
        clipped_sample_count=clipped_count,
        total_output_samples=total_samples,
        clipping_fraction=clipping_fraction,
    )
    return PreparedAudioTransform(
        pcm_bytes=clipped.tobytes(order="C"),
        receipt=receipt,
    )


def write_prepared_wav(
    *,
    prepared: PreparedAudioTransform,
    output_path: Path,
) -> None:
    """Atomically write a prepared 16-bit PCM transform to WAV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.wav")
    try:
        with wave.open(str(tmp_path), "wb") as writer:
            writer.setnchannels(prepared.receipt.output_channels)
            writer.setsampwidth(2)
            writer.setframerate(prepared.receipt.output_sample_rate)
            writer.writeframes(prepared.pcm_bytes)
        with tmp_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _require_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - dependency preflight
        raise RuntimeError("numpy_required_for_audio_augmentation") from exc
    return numpy


def _read_pcm_int16(*, decoded_audio: DecodedAudio) -> Any:
    np = _require_numpy()
    if decoded_audio.sample_width != 2:
        raise ValueError(
            f"unsupported decoded sample width: {decoded_audio.sample_width}"
        )
    if decoded_audio.channels < 1:
        raise ValueError("decoded_audio_channels_invalid")
    if decoded_audio.sample_rate < 1:
        raise ValueError("decoded_audio_sample_rate_invalid")

    frame_width = decoded_audio.channels * decoded_audio.sample_width
    chunks: list[bytes] = []
    byte_count = 0
    iterator = decoded_audio.frames_iterator
    try:
        for chunk in iterator:
            if len(chunk) % frame_width != 0:
                raise ValueError("decoded_audio_chunk_not_frame_aligned")
            chunks.append(chunk)
            byte_count += len(chunk)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    if byte_count == 0:
        raise ValueError("decoded_audio_empty")

    values = np.frombuffer(b"".join(chunks), dtype="<i2")
    return values.reshape((-1, decoded_audio.channels)).astype(np.float32)


def _trim_bounds(
    *,
    samples: Any,
    sample_rate: int,
    enabled: bool,
    threshold_dbfs: float,
    padding_seconds: float,
) -> tuple[int, int]:
    np = _require_numpy()
    frame_count = int(samples.shape[0])
    if not enabled:
        return 0, frame_count

    threshold = _INT16_MAX * (10.0 ** (threshold_dbfs / 20.0))
    active = np.flatnonzero(np.max(np.abs(samples), axis=1) > threshold)
    if active.size == 0:
        # Silence has no reliable content boundary. Preserve the complete input
        # rather than manufacturing an arbitrary clip or corrupting timestamps.
        return 0, frame_count
    padding_frames = int(round(padding_seconds * sample_rate))
    start = max(0, int(active[0]) - padding_frames)
    end = min(frame_count, int(active[-1]) + 1 + padding_frames)
    return start, end


def _convert_channels(
    samples: Any,
    *,
    target_channels: int,
) -> Any:
    np = _require_numpy()
    if target_channels < 1:
        raise ValueError("target_channels_invalid")
    source_channels = int(samples.shape[1])
    if source_channels == target_channels:
        return samples
    if target_channels == 1:
        return np.mean(samples, axis=1, keepdims=True)
    if source_channels == 1:
        return np.repeat(samples, target_channels, axis=1)
    if target_channels < source_channels:
        groups = np.array_split(np.arange(source_channels), target_channels)
        return np.column_stack(
            [np.mean(samples[:, group], axis=1) for group in groups]
        )
    indices = np.arange(target_channels) % source_channels
    return samples[:, indices]


def _resample_with_speed(
    *,
    samples: Any,
    input_sample_rate: int,
    output_sample_rate: int,
    speed_factor: float,
) -> Any:
    np = _require_numpy()
    if output_sample_rate < 1:
        raise ValueError("output_sample_rate_invalid")
    if not math.isfinite(speed_factor) or speed_factor <= 0.0:
        raise ValueError("speed_factor_invalid")
    input_frames = int(samples.shape[0])
    output_frames = max(
        1,
        int(
            round(
                input_frames
                * float(output_sample_rate)
                / float(input_sample_rate)
                / speed_factor
            )
        ),
    )
    if input_frames == 1:
        return np.repeat(samples, output_frames, axis=0)

    try:
        from scipy.signal import resample_poly  # type: ignore[import-untyped]
    except (
        ImportError
    ) as exc:  # pragma: no cover - dependency preflight covers this
        raise RuntimeError("scipy_required_for_audio_resampling") from exc

    ratio = Fraction(
        float(output_sample_rate) / (float(input_sample_rate) * speed_factor)
    ).limit_denominator(100_000)
    resampled = resample_poly(
        samples,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=0,
        padtype="line",
    )
    if resampled.shape[0] > output_frames:
        return resampled[:output_frames]
    if resampled.shape[0] < output_frames:
        missing = output_frames - int(resampled.shape[0])
        tail = np.repeat(resampled[-1:, :], missing, axis=0)
        return np.concatenate((resampled, tail), axis=0)
    return resampled
