"""Deterministic prosody extraction."""

from __future__ import annotations

import math
import statistics
from typing import Final

from preprocessing.media.adapters.audio_decode import decode_audio_samples
from preprocessing.media.speech.prosody_contracts import (
    ProsodyAnalysisResult,
    ProsodyFeatures,
    ProsodyStatus,
)
from preprocessing.media.speech.prosody_dsp import (
    _bounded_float,
    _clamp_unit,
    _measure_frames,
    _measure_pitch,
    _measure_tempo,
    _positive_float,
    _positive_int,
)

_DEFAULT_FRAME_DURATION_SECONDS: Final[float] = 0.04
_DEFAULT_HOP_DURATION_SECONDS: Final[float] = 0.02
_DEFAULT_MIN_PITCH_HZ: Final[float] = 60.0
_DEFAULT_MAX_PITCH_HZ: Final[float] = 500.0
_DEFAULT_SILENCE_RMS_THRESHOLD: Final[float] = 0.01
_DEFAULT_PITCH_CORRELATION_THRESHOLD: Final[float] = 0.35
_DEFAULT_MAX_PITCH_FRAMES: Final[int] = 64
_DEFAULT_TARGET_PITCH_SAMPLE_RATE: Final[int] = 8_000


class ProsodyExtractor:
    """Measure deterministic prosody features from audio samples."""

    def __init__(
        self,
        *,
        frame_duration_seconds: float = _DEFAULT_FRAME_DURATION_SECONDS,
        hop_duration_seconds: float = _DEFAULT_HOP_DURATION_SECONDS,
        min_pitch_hz: float = _DEFAULT_MIN_PITCH_HZ,
        max_pitch_hz: float = _DEFAULT_MAX_PITCH_HZ,
        silence_rms_threshold: float = _DEFAULT_SILENCE_RMS_THRESHOLD,
        pitch_correlation_threshold: float = (
            _DEFAULT_PITCH_CORRELATION_THRESHOLD
        ),
        max_pitch_frames: int = _DEFAULT_MAX_PITCH_FRAMES,
        target_pitch_sample_rate: int = (_DEFAULT_TARGET_PITCH_SAMPLE_RATE),
    ) -> None:
        self._frame_duration_seconds = _positive_float(
            frame_duration_seconds,
            field_name="frame_duration_seconds",
        )
        self._hop_duration_seconds = _positive_float(
            hop_duration_seconds,
            field_name="hop_duration_seconds",
        )
        self._min_pitch_hz = _positive_float(
            min_pitch_hz,
            field_name="min_pitch_hz",
        )
        self._max_pitch_hz = _positive_float(
            max_pitch_hz,
            field_name="max_pitch_hz",
        )
        if self._max_pitch_hz <= self._min_pitch_hz:
            raise ValueError("max_pitch_hz must exceed min_pitch_hz")

        self._silence_rms_threshold = _bounded_float(
            silence_rms_threshold,
            field_name="silence_rms_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        self._pitch_correlation_threshold = _bounded_float(
            pitch_correlation_threshold,
            field_name="pitch_correlation_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        self._max_pitch_frames = _positive_int(
            max_pitch_frames,
            field_name="max_pitch_frames",
        )
        self._target_pitch_sample_rate = _positive_int(
            target_pitch_sample_rate,
            field_name="target_pitch_sample_rate",
        )

    def extract(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int | None = None,
        transcript: str | None = None,
    ) -> ProsodyAnalysisResult:
        """Extract pitch, energy, tempo, and pause measurements."""

        decoded = decode_audio_samples(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
        )
        if decoded is None:
            return ProsodyAnalysisResult(
                status=ProsodyStatus.UNAVAILABLE,
                reasons=("unsupported_or_empty_audio",),
            )

        samples, decoded_sample_rate = decoded
        if not samples or decoded_sample_rate <= 0:
            return ProsodyAnalysisResult(
                status=ProsodyStatus.UNAVAILABLE,
                reasons=("no_decodable_audio_samples",),
            )

        if any(not math.isfinite(sample) for sample in samples):
            return ProsodyAnalysisResult(
                status=ProsodyStatus.FAILED,
                reasons=("non_finite_audio_samples",),
            )

        duration_seconds = len(samples) / decoded_sample_rate
        frame_size = max(
            1,
            round(decoded_sample_rate * self._frame_duration_seconds),
        )
        hop_size = max(
            1,
            round(decoded_sample_rate * self._hop_duration_seconds),
        )

        frame_statistics = _measure_frames(
            samples=samples,
            frame_size=frame_size,
            hop_size=hop_size,
            silence_rms_threshold=self._silence_rms_threshold,
        )
        pitch_measurements = _measure_pitch(
            samples=samples,
            sample_rate=decoded_sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            silence_rms_threshold=self._silence_rms_threshold,
            min_pitch_hz=self._min_pitch_hz,
            max_pitch_hz=self._max_pitch_hz,
            correlation_threshold=self._pitch_correlation_threshold,
            max_frames=self._max_pitch_frames,
            target_sample_rate=self._target_pitch_sample_rate,
        )

        pitch_hz = (
            statistics.fmean(pitch_measurements)
            if pitch_measurements
            else None
        )
        pitch_std_hz = (
            statistics.pstdev(pitch_measurements)
            if pitch_measurements
            else None
        )
        tempo = _measure_tempo(
            transcript=transcript,
            duration_seconds=duration_seconds,
        )

        features = ProsodyFeatures(
            pitch_hz=pitch_hz,
            energy=_clamp_unit(frame_statistics.mean_rms),
            tempo=tempo,
            pause_ratio=_clamp_unit(frame_statistics.pause_ratio),
        )

        reasons: list[str] = []
        if pitch_hz is None:
            reasons.append("pitch_unavailable")
        if tempo is None:
            reasons.append("transcript_unavailable_for_tempo")

        status = (
            ProsodyStatus.AVAILABLE
            if features.is_complete
            else ProsodyStatus.PARTIAL
        )

        return ProsodyAnalysisResult(
            features=features,
            pitch_std_hz=pitch_std_hz,
            duration_seconds=duration_seconds,
            status=status,
            reasons=tuple(reasons),
        )


__all__ = ["ProsodyExtractor"]
