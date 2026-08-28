"""Deterministic signal-processing primitives for prosody extraction."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _FrameStatistics:
    mean_rms: float
    pause_ratio: float


def _measure_frames(
    *,
    samples: Sequence[float],
    frame_size: int,
    hop_size: int,
    silence_rms_threshold: float,
) -> _FrameStatistics:
    offsets = _frame_offsets(
        sample_count=len(samples),
        frame_size=frame_size,
        hop_size=hop_size,
    )
    rms_total = 0.0
    silent_frames = 0
    frame_count = 0

    for offset in offsets:
        frame = samples[offset : offset + frame_size]
        if not frame:
            continue

        rms = _root_mean_square(frame)
        rms_total += rms
        frame_count += 1
        if rms <= silence_rms_threshold:
            silent_frames += 1

    if frame_count == 0:
        rms = _root_mean_square(samples)
        return _FrameStatistics(
            mean_rms=rms,
            pause_ratio=float(rms <= silence_rms_threshold),
        )

    return _FrameStatistics(
        mean_rms=rms_total / frame_count,
        pause_ratio=silent_frames / frame_count,
    )


def _measure_pitch(
    *,
    samples: Sequence[float],
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    silence_rms_threshold: float,
    min_pitch_hz: float,
    max_pitch_hz: float,
    correlation_threshold: float,
    max_frames: int,
    target_sample_rate: int,
) -> tuple[float, ...]:
    offsets = _sampled_frame_offsets(
        sample_count=len(samples),
        frame_size=frame_size,
        hop_size=hop_size,
        maximum=max_frames,
    )
    measurements: list[float] = []

    for offset in offsets:
        frame = samples[offset : offset + frame_size]
        if not frame:
            continue
        if _root_mean_square(frame) <= silence_rms_threshold:
            continue

        measurement = _estimate_frame_pitch(
            frame=frame,
            sample_rate=sample_rate,
            min_pitch_hz=min_pitch_hz,
            max_pitch_hz=max_pitch_hz,
            correlation_threshold=correlation_threshold,
            target_sample_rate=target_sample_rate,
        )
        if measurement is not None:
            measurements.append(measurement)

    return tuple(measurements)


def _estimate_frame_pitch(
    *,
    frame: Sequence[float],
    sample_rate: int,
    min_pitch_hz: float,
    max_pitch_hz: float,
    correlation_threshold: float,
    target_sample_rate: int,
) -> float | None:
    stride = max(1, round(sample_rate / target_sample_rate))
    sampled = tuple(float(value) for value in frame[::stride])
    effective_sample_rate = sample_rate / stride

    if len(sampled) < 3:
        return None

    mean = statistics.fmean(sampled)
    centered = tuple(value - mean for value in sampled)
    if _root_mean_square(centered) <= 1e-6:
        return None

    minimum_lag = max(1, math.floor(effective_sample_rate / max_pitch_hz))
    maximum_lag = min(
        len(centered) - 2,
        math.ceil(effective_sample_rate / min_pitch_hz),
    )
    if maximum_lag < minimum_lag:
        return None

    correlations = tuple(
        (
            lag,
            _normalized_autocorrelation(
                samples=centered,
                lag=lag,
            ),
        )
        for lag in range(minimum_lag, maximum_lag + 1)
    )
    if not correlations:
        return None

    best_correlation = max(correlation for _, correlation in correlations)
    required_correlation = max(
        correlation_threshold,
        best_correlation * 0.8,
    )
    best_lag = _first_strong_local_peak(
        correlations=correlations,
        required_correlation=required_correlation,
    )
    if best_lag is None:
        return None

    pitch_hz = effective_sample_rate / best_lag
    if not min_pitch_hz <= pitch_hz <= max_pitch_hz:
        return None
    return pitch_hz


def _first_strong_local_peak(
    *,
    correlations: Sequence[tuple[int, float]],
    required_correlation: float,
) -> int | None:
    if len(correlations) == 1:
        lag, correlation = correlations[0]
        return lag if correlation >= required_correlation else None

    for index, (lag, correlation) in enumerate(correlations):
        previous = correlations[index - 1][1] if index > 0 else -1.0
        following = (
            correlations[index + 1][1]
            if index + 1 < len(correlations)
            else -1.0
        )
        if (
            correlation >= required_correlation
            and correlation >= previous
            and correlation >= following
        ):
            return lag

    return None


def _normalized_autocorrelation(
    *,
    samples: Sequence[float],
    lag: int,
) -> float:
    left = samples[:-lag]
    right = samples[lag:]
    if not left or not right:
        return 0.0

    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_energy = sum(value * value for value in left)
    right_energy = sum(value * value for value in right)
    denominator = math.sqrt(left_energy * right_energy)
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _measure_tempo(
    *,
    transcript: str | None,
    duration_seconds: float,
) -> float | None:
    if transcript is None:
        return None

    normalized = transcript.strip()
    if not normalized:
        return None

    word_count = len(normalized.split())
    return word_count / duration_seconds


def _frame_offsets(
    *,
    sample_count: int,
    frame_size: int,
    hop_size: int,
) -> range | tuple[int, ...]:
    if sample_count <= frame_size:
        return (0,)
    return range(0, sample_count - frame_size + 1, hop_size)


def _sampled_frame_offsets(
    *,
    sample_count: int,
    frame_size: int,
    hop_size: int,
    maximum: int,
) -> tuple[int, ...]:
    if sample_count <= frame_size:
        return (0,)

    frame_count = 1 + ((sample_count - frame_size) // hop_size)
    if frame_count <= maximum:
        return tuple(index * hop_size for index in range(frame_count))

    if maximum == 1:
        return (0,)

    indexes = {
        round(position * (frame_count - 1) / (maximum - 1))
        for position in range(maximum)
    }
    return tuple(index * hop_size for index in sorted(indexes))


def _root_mean_square(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def _positive_float(value: object, *, field_name: str) -> float:
    parsed = _bounded_float(
        value,
        field_name=field_name,
        minimum=0.0,
        maximum=None,
    )
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return parsed


def _bounded_float(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float | None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")

    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    if parsed < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return parsed


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))
