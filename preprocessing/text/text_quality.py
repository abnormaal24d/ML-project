"""Text quality scoring algorithm."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from preprocessing.preprocessing_quality import PreprocessingQualityResult

if TYPE_CHECKING:
    from config.preprocessing.text_settings import TextQualityScorerSettings
    from preprocessing.preprocessing_input import LanguageEvidence
    from preprocessing.text.text_language import LanguageDetector


class TextQualityScorer:
    """Score extracted text with configurable production-oriented gates."""

    def __init__(
        self,
        *,
        settings: TextQualityScorerSettings,
        language_detector: LanguageDetector,
    ) -> None:
        self._settings = settings
        self._language_detector = language_detector

    def score(
        self,
        *,
        text: str,
        boilerplate_ratio: float,
        language_evidence: LanguageEvidence | None,
    ) -> PreprocessingQualityResult:
        """Return quality score and rejection metadata for extracted text."""

        from preprocessing.text.text_language import verify_or_detect_language

        detection = verify_or_detect_language(
            detector=self._language_detector,
            text=text,
            evidence=language_evidence,
            trusted_confidence=self._settings.trusted_language_confidence,
            mismatch_confidence=self._settings.language_mismatch_confidence,
        )
        signals = _quality_signals(
            text=text,
            language=detection.language,
            script=detection.script,
        )
        char_count = signals.char_count
        token_count = signals.token_count
        normalized_boilerplate_ratio = max(0.0, min(1.0, boilerplate_ratio))

        rejection_reason: str | None = None
        if char_count < self._settings.min_text_chars:
            rejection_reason = "text_too_short"
        elif char_count > self._settings.max_text_chars:
            rejection_reason = "text_too_long"
        elif (
            normalized_boilerplate_ratio > self._settings.max_boilerplate_ratio
            and char_count < self._settings.min_text_chars * 3
        ):
            rejection_reason = "boilerplate_heavy"
        elif not detection.language:
            rejection_reason = "language_missing"

        score = _score_signal_mix(
            char_count=char_count,
            ascii_ratio=signals.ascii_ratio,
            control_char_ratio=signals.control_char_ratio,
            boilerplate_ratio=normalized_boilerplate_ratio,
            unique_terms=signals.unique_terms,
            repetition_ratio=signals.repetition_ratio,
            code_text_ratio=signals.code_text_ratio,
            language_confidence=detection.language_confidence,
            script=detection.script,
            settings=self._settings,
        )
        if rejection_reason is not None:
            bucket = "reject"
            score = min(score, self._settings.rejected_score_cap)
        elif score >= self._settings.gold_threshold:
            bucket = "gold"
        elif score >= self._settings.silver_threshold:
            bucket = "silver"
        elif score >= self._settings.bronze_threshold:
            bucket = "bronze"
        else:
            bucket = "reject"
            rejection_reason = "quality_below_threshold"

        signal_payload: dict[str, float | int | bool | str | None] = {
            "char_count": char_count,
            "token_count_estimate": token_count,
            "ascii_ratio": round(signals.ascii_ratio, 4),
            "unicode_ratio": round(signals.unicode_ratio, 4),
            "control_char_ratio": round(signals.control_char_ratio, 4),
            "boilerplate_ratio": round(normalized_boilerplate_ratio, 4),
            "language": detection.language,
            "language_inferred": not detection.used_crawler_evidence,
            "language_confidence": detection.language_confidence,
            "language_source": detection.language_source,
            "language_verification_mismatch": detection.verification_mismatch,
            "language_detector_version": detection.language_detector_version,
            "script": detection.script,
            "is_mixed_language": detection.is_mixed_language,
            "repetition_ratio": round(signals.repetition_ratio, 4),
            "code_text_ratio": round(signals.code_text_ratio, 4),
        }
        return PreprocessingQualityResult(
            score=round(score, 4),
            bucket=bucket,
            rejection_reason=rejection_reason,
            token_count_estimate=token_count,
            modality="text",
            language=detection.language,
            alignment_score=score,
            signals=signal_payload,
        )


@dataclass(frozen=True, slots=True)
class _QualitySignals:
    char_count: int
    token_count: int
    ascii_ratio: float
    unicode_ratio: float
    control_char_ratio: float
    unique_terms: int
    repetition_ratio: float
    code_text_ratio: float


def _quality_signals(
    *,
    text: str,
    language: str | None,
    script: str,
) -> _QualitySignals:
    char_count = len(text)
    tokens = _iter_tokens(text=text, language=language, script=script)
    unique_terms = {token.casefold() for token in tokens}
    ascii_ratio = _ascii_ratio(text=text, char_count=char_count)
    return _QualitySignals(
        char_count=char_count,
        token_count=len(tokens),
        ascii_ratio=ascii_ratio,
        unicode_ratio=1.0 - ascii_ratio,
        control_char_ratio=_control_char_ratio(
            text=text,
            char_count=char_count,
        ),
        unique_terms=len(unique_terms),
        repetition_ratio=_repetition_ratio(tokens=tokens),
        code_text_ratio=_code_text_ratio(text=text),
    )


def _ascii_ratio(*, text: str, char_count: int | None = None) -> float:
    if not text:
        return 0.0
    resolved_char_count = len(text) if char_count is None else char_count
    if text.isascii():
        return 1.0
    ascii_count = sum(1 for char in text if ord(char) < 128)
    return ascii_count / resolved_char_count


def _score_signal_mix(
    *,
    char_count: int,
    ascii_ratio: float,
    control_char_ratio: float,
    boilerplate_ratio: float,
    unique_terms: int,
    repetition_ratio: float,
    code_text_ratio: float,
    language_confidence: float,
    script: str,
    settings: TextQualityScorerSettings,
) -> float:
    score = 1.0
    if char_count < settings.short_text_chars:
        score -= settings.short_text_penalty
    if char_count < settings.medium_text_chars:
        score -= settings.medium_text_penalty
    if script == "Latin" and ascii_ratio < settings.min_ascii_ratio:
        score -= settings.ascii_penalty
    if control_char_ratio > 0.02:
        score -= settings.ascii_penalty
    if boilerplate_ratio > settings.boilerplate_penalty_threshold:
        score -= settings.boilerplate_penalty
    if unique_terms < settings.min_unique_terms:
        score -= settings.unique_terms_penalty
    if repetition_ratio > 0.45:
        score -= 0.15
    if code_text_ratio > 0.35:
        score -= 0.15
    if language_confidence < 0.35:
        score -= 0.1
    return max(0.0, min(1.0, score))


def _iter_tokens(
    *,
    text: str,
    language: str | None,
    script: str,
) -> tuple[str, ...]:
    if script in {"Han", "Hiragana", "Katakana"}:
        return tuple(char for char in text if char.isalpha())
    del language
    return tuple(match.group(0) for match in re.finditer(r"\w+", text))


def _control_char_ratio(*, text: str, char_count: int) -> float:
    if not text or char_count <= 0:
        return 0.0
    control_count = sum(
        1
        for char in text
        if unicodedata.category(char).startswith("C")
        and char not in {"\n", "\r", "\t"}
    )
    return control_count / char_count


def _repetition_ratio(*, tokens: tuple[str, ...]) -> float:
    if not tokens:
        return 0.0
    repeated = 0
    previous = None
    for token in tokens:
        folded = token.casefold()
        if folded == previous:
            repeated += 1
        previous = folded
    return repeated / len(tokens)


def _code_text_ratio(*, text: str) -> float:
    if not text:
        return 0.0
    code_markers = sum(
        text.count(marker)
        for marker in ("{", "}", ";", "=>", "</", "def ", "class ")
    )
    token_count = max(1, len(text.split()))
    return min(1.0, code_markers / token_count)
