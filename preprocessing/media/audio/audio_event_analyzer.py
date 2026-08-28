"""Preprocessing helper for deterministic audio activity measurement."""

from __future__ import annotations

import math
import statistics
from typing import Any

from preprocessing.media.adapters.audio_decode import decode_audio_samples


class AudioEventAnalyzer:
    def __init__(
        self, *, model_name: str = "deterministic-signal-activity-v1"
    ) -> None:
        self.model_name = model_name

    def analyze(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        decoded = decode_audio_samples(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
        )
        if decoded is None:
            return _audio_event_payload(
                background_noise_label=None,
                acoustic_scene_label=None,
                sound_events=[],
                confidence=None,
                analysis_status="unavailable",
                analysis_reasons=["unsupported_or_empty_audio"],
                model_name=self.model_name,
            )

        samples, rate = decoded
        windows = _window_features(samples=samples, sample_rate=rate)
        if not windows:
            return _audio_event_payload(
                background_noise_label=None,
                acoustic_scene_label=None,
                sound_events=[],
                confidence=None,
                analysis_status="unavailable",
                analysis_reasons=["no_decodable_audio_frames"],
                model_name=self.model_name,
            )

        rms_values = [item[2] for item in windows]
        zcr_values = [item[3] for item in windows]
        mean_rms = statistics.fmean(rms_values)
        mean_zcr = statistics.fmean(zcr_values)
        silence_threshold = max(
            0.005,
            sorted(rms_values)[max(0, int(len(rms_values) * 0.2) - 1)] * 2,
        )
        active_ratio = sum(
            1 for value in rms_values if value > silence_threshold
        ) / len(rms_values)

        if active_ratio < 0.05:
            scene = None
            background = "silence"
            confidence = 0.90
        elif mean_zcr > 0.18 and mean_rms > 0.03:
            scene = None
            background = None
            confidence = 0.65
        elif active_ratio > 0.65 and mean_zcr < 0.12:
            scene = None
            background = None
            confidence = 0.55
        else:
            scene = None
            background = None
            confidence = 0.50

        return _audio_event_payload(
            background_noise_label=background,
            acoustic_scene_label=scene,
            sound_events=_merge_window_events(
                windows=windows, silence_threshold=silence_threshold
            ),
            confidence=confidence,
            analysis_status="passed",
            analysis_reasons=[
                "taxonomy_model_unavailable",
                "deterministic_signal_features",
            ],
            model_name=self.model_name,
        )


def _audio_event_payload(
    *,
    background_noise_label: str | None,
    acoustic_scene_label: str | None,
    sound_events: list[dict[str, Any]],
    confidence: float | None,
    analysis_status: str,
    analysis_reasons: list[str],
    model_name: str,
) -> dict[str, Any]:
    return {
        "background_noise_label": background_noise_label,
        "acoustic_scene_label": acoustic_scene_label,
        "sound_events": sound_events,
        "model_name": model_name,
        "confidence": confidence,
        "backend": "deterministic_signal_features",
        "classifier_kind": "measured_audio_activity",
        "signal_analysis_status": analysis_status,
        "taxonomy_classification_status": "not_run",
        "taxonomy_model_available": False,
        "taxonomy_status": "unavailable",
        "label_semantics": "measured_signal_activity",
        "analysis_status": analysis_status,
        "analysis_reasons": analysis_reasons,
    }


def _window_features(
    *,
    samples: tuple[float, ...],
    sample_rate: int,
) -> tuple[tuple[float, float, float, float], ...]:
    window_size = max(1, int(sample_rate * 0.5))
    features: list[tuple[float, float, float, float]] = []
    for offset in range(0, len(samples), window_size):
        window = samples[offset : offset + window_size]
        if not window:
            continue
        start = offset / sample_rate
        end = min(len(samples), offset + len(window)) / sample_rate
        rms = math.sqrt(
            sum(sample * sample for sample in window) / len(window)
        )
        zcr = _zero_crossing_rate(samples=window)
        features.append((start, end, rms, zcr))
    return tuple(features)


def _zero_crossing_rate(*, samples: tuple[float, ...]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = 0
    previous = samples[0]
    for sample in samples[1:]:
        if previous < 0 <= sample or previous >= 0 > sample:
            crossings += 1
        previous = sample
    return crossings / (len(samples) - 1)


def _merge_window_events(
    *,
    windows: tuple[tuple[float, float, float, float], ...],
    silence_threshold: float,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    current_label: str | None = None
    current_start = 0.0
    current_end = 0.0
    current_conf = 0.0

    for start, end, rms, zcr in windows:
        if rms <= silence_threshold:
            label = "silence"
            confidence = 0.90
        elif zcr > 0.18:
            label = "high_zero_crossing_activity"
            confidence = 0.60
        else:
            label = "low_zero_crossing_activity"
            confidence = 0.55

        if label != current_label:
            if current_label is not None:
                merged.append(
                    {
                        "label": current_label,
                        "label_semantics": "measured_signal_activity",
                        "start": current_start,
                        "end": current_end,
                        "confidence": current_conf,
                    }
                )
            current_label = label
            current_start = start
            current_conf = confidence
        current_end = end
        current_conf = max(current_conf, confidence)

    if current_label is not None:
        merged.append(
            {
                "label": current_label,
                "label_semantics": "measured_signal_activity",
                "start": current_start,
                "end": current_end,
                "confidence": current_conf,
            }
        )
    return merged
