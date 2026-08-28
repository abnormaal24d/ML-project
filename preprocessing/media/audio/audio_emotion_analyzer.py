"""Preprocessing helper for conservative affect analysis from prosody."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from preprocessing.media.speech.prosody_contracts import (
    ProsodyFeatures,
    ProsodyStatus,
)
from preprocessing.media.speech.prosody_extractor import (
    ProsodyExtractor,
)

__all__ = [
    "AudioEmotionAnalyzer",
    "AudioEmotionResult",
    "AudioEmotionStatus",
]

_DEFAULT_MODEL_NAME: Final[str] = "deterministic-prosody-affect-v2"


class AudioEmotionStatus(StrEnum):
    """Outcome of conservative audio-affect analysis."""

    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioEmotionResult:
    """Affect metadata backed only by measured audio signals."""

    emotion_label: str | None = None
    emotion_confidence: float | None = None
    arousal: float | None = None
    valence: float | None = None
    dominance: float | None = None
    prosody: ProsodyFeatures | None = None
    model_name: str = _DEFAULT_MODEL_NAME
    analysis_status: AudioEmotionStatus = AudioEmotionStatus.UNAVAILABLE
    analysis_reasons: tuple[str, ...] = ()


class AudioEmotionAnalyzer:
    """Derive conservative affect metadata from measured prosody."""

    def __init__(
        self,
        *,
        prosody_extractor: ProsodyExtractor,
        model_name: str = _DEFAULT_MODEL_NAME,
    ) -> None:
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("model_name must not be empty")

        self._prosody_extractor = prosody_extractor
        self._model_name = normalized_model_name

    def analyze(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int | None = None,
        transcript: str | None = None,
    ) -> AudioEmotionResult:
        """Analyze affect-related audio measurements.

        No emotion label, valence, or dominance score is emitted without a
        dedicated trained classifier. Arousal is a bounded signal heuristic.
        """

        prosody_result = self._prosody_extractor.extract(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            transcript=transcript,
        )

        features = prosody_result.features
        arousal = _estimate_arousal(features)
        reasons = list(prosody_result.reasons)
        reasons.append("emotion_classifier_unavailable")

        if arousal is not None:
            reasons.append("arousal_is_signal_heuristic")

        return AudioEmotionResult(
            emotion_label=None,
            emotion_confidence=None,
            arousal=arousal,
            valence=None,
            dominance=None,
            prosody=features,
            model_name=self._model_name,
            analysis_status=_emotion_status(
                prosody_status=prosody_result.status,
            ),
            analysis_reasons=_unique_reasons(reasons),
        )


def _emotion_status(
    *,
    prosody_status: ProsodyStatus,
) -> AudioEmotionStatus:
    if prosody_status is ProsodyStatus.FAILED:
        return AudioEmotionStatus.FAILED

    if prosody_status in {
        ProsodyStatus.AVAILABLE,
        ProsodyStatus.PARTIAL,
    }:
        return AudioEmotionStatus.PARTIAL

    return AudioEmotionStatus.UNAVAILABLE


def _estimate_arousal(
    features: ProsodyFeatures | None,
) -> float | None:
    if features is None or features.energy is None:
        return None

    energy_component = _clamp(features.energy * 8.0)
    voiced_component = (
        _clamp(1.0 - features.pause_ratio)
        if features.pause_ratio is not None
        else 0.0
    )

    return _clamp((0.7 * energy_component) + (0.3 * voiced_component))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _unique_reasons(reasons: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()

    for reason in reasons:
        cleaned_reason = reason.strip()
        if not cleaned_reason or cleaned_reason in seen:
            continue
        seen.add(cleaned_reason)
        normalized.append(cleaned_reason)

    return tuple(normalized)
