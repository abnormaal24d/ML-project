"""Preprocessing heuristic language detection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from preprocessing.preprocessing_input import LanguageEvidence

LANGUAGE_DETECTOR_VERSION = "preprocessing-heuristic-v1"


@dataclass(frozen=True, slots=True)
class LanguageDetectionResult:
    """Normalized output from the preprocessing language detector.

    Attributes:
        language_code: ISO-style language code when evidence is available.
        confidence: Detector confidence in ``[0.0, 1.0]``.
        detector_name: Provenance label such as ``latin_marker`` or
            ``unicode_range``.
        text_length: Length of the analyzed text after stripping whitespace.
    """

    language_code: str | None
    confidence: float
    detector_name: str
    text_length: int


if TYPE_CHECKING:
    from collections.abc import Mapping


class LanguageDetector:
    """Detect likely language using script and token heuristics."""

    def detect(
        self,
        *,
        text: str,
        language_hint: str | None = None,
        stopwords_by_language: Mapping[str, tuple[str, ...]] | None = None,
        min_text_length: int = 20,
        min_best_score: float = 2.0,
        require_clear_winner: bool = True,
        use_stopword_matching: bool = True,
        use_unicode_ranges: bool = True,
    ) -> LanguageDetectionResult:
        """Return language code, confidence, detector name and analyzed length."""

        stripped = text.strip()
        text_length = len(stripped)

        if language_hint and language_hint.strip():
            return LanguageDetectionResult(
                language_code=language_hint.strip().lower(),
                confidence=1.0,
                detector_name="language_hint",
                text_length=text_length,
            )

        if text_length < min_text_length:
            return _no_evidence(text_length=text_length)

        script_counts = _script_counts(text=stripped)
        primary_script = _primary_script(script_counts=script_counts)

        if primary_script == "Latin":
            language, confidence, detector_name = _detect_latin_language(
                text=stripped,
                stopwords_by_language=stopwords_by_language,
                min_text_length=min_text_length,
                min_best_score=min_best_score,
                require_clear_winner=require_clear_winner,
                use_stopword_matching=use_stopword_matching,
            )
            if language is not None:
                return LanguageDetectionResult(
                    language_code=language,
                    confidence=confidence,
                    detector_name=detector_name,
                    text_length=text_length,
                )

        if use_unicode_ranges:
            unicode_language, unicode_confidence = (
                _detect_unicode_range_language(
                    text=stripped,
                )
            )
            if unicode_language is not None:
                return LanguageDetectionResult(
                    language_code=unicode_language,
                    confidence=unicode_confidence,
                    detector_name="unicode_range",
                    text_length=text_length,
                )

        language = _language_for_script(script=primary_script)
        if language is not None:
            return LanguageDetectionResult(
                language_code=language,
                confidence=0.65,
                detector_name="script_inference",
                text_length=text_length,
            )

        return _no_evidence(text_length=text_length)


def is_mixed_latin_language(*, text: str) -> bool:
    """Return whether multiple Latin languages score similarly in marker analysis."""

    tokens = tuple(token.casefold() for token in _iter_tokens(text=text))
    if not tokens:
        return False

    scores = {
        language: sum(1 for token in tokens if token in markers)
        for language, markers in _LATIN_MARKERS.items()
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_score = ordered[0][1]
    second_score = ordered[1][1] if len(ordered) > 1 else 0
    return second_score > 0 and second_score >= best_score * 0.65


def detect_script_profile(*, text: str) -> tuple[str, bool]:
    """Return primary script and whether multiple scripts are materially present."""

    script_counts = _script_counts(text=text)
    primary_script = _primary_script(script_counts=script_counts)
    mixed_script = _is_mixed_script(script_counts=script_counts)
    return primary_script, mixed_script


def _no_evidence(*, text_length: int) -> LanguageDetectionResult:
    return LanguageDetectionResult(
        language_code=None,
        confidence=0.0,
        detector_name="none",
        text_length=text_length,
    )


def _detect_latin_language(
    *,
    text: str,
    stopwords_by_language: Mapping[str, tuple[str, ...]] | None,
    min_text_length: int,
    min_best_score: float,
    require_clear_winner: bool,
    use_stopword_matching: bool,
) -> tuple[str | None, float, str]:
    tokens = tuple(token.casefold() for token in _iter_tokens(text=text))
    if not tokens:
        return None, 0.0, "none"

    scores: dict[str, float] = {}

    if use_stopword_matching and stopwords_by_language:
        scored_tokens = tokens[:400]
        for language, stopwords in stopwords_by_language.items():
            language_stopwords = set(stopwords)
            scores[language] = float(
                sum(
                    1 for token in scored_tokens if token in language_stopwords
                )
            )

    marker_scores = {
        language: sum(1 for token in tokens if token in markers)
        for language, markers in _LATIN_MARKERS.items()
    }
    for language, marker_score in marker_scores.items():
        scores[language] = scores.get(language, 0.0) + float(marker_score)

    if not scores:
        return None, 0.0, "none"

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_language, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0

    if best_score < min_best_score:
        if max(marker_scores.values(), default=0.0) <= 0:
            return "und", 0.2, "latin_marker"
        return None, 0.0, "none"

    if (
        require_clear_winner
        and second_score > 0
        and best_score == second_score
    ):
        return None, 0.0, "none"

    marker_hits = sum(marker_scores.values())
    if marker_hits > 0 and best_score == marker_scores.get(best_language, 0.0):
        confidence = min(0.99, max(0.35, best_score / max(1, marker_hits)))
        confidence = max(
            confidence,
            min(0.95, best_score / max(6, len(tokens))),
        )
        mixed_language = second_score > 0 and second_score >= best_score * 0.65
        if mixed_language:
            confidence = min(confidence, 0.75)
        return best_language, round(confidence, 4), "latin_marker"

    confidence = min(1.0, best_score / max(min_best_score, 1.0))
    return best_language, round(confidence, 4), "stopword"


def _detect_unicode_range_language(*, text: str) -> tuple[str | None, float]:
    scores = _unicode_range_scores(text=text.lower())
    if not scores:
        return None, 0.0
    best_language, best_score = max(scores.items(), key=lambda item: item[1])
    confidence = min(1.0, best_score / 4.0)
    return best_language, round(confidence, 4)


def _unicode_range_scores(*, text: str) -> dict[str, float]:
    ranges = {
        "ar": ((0x0600, 0x06FF), (0x0750, 0x077F)),
        "ru": ((0x0400, 0x04FF),),
        "zh": ((0x4E00, 0x9FFF),),
        "ja": ((0x3040, 0x30FF),),
    }
    scores: dict[str, float] = {}
    for char in text:
        codepoint = ord(char)
        for language, language_ranges in ranges.items():
            if any(
                start <= codepoint and codepoint <= end
                for start, end in language_ranges
            ):
                scores[language] = scores.get(language, 0.0) + 1.0
    return {
        language: score for language, score in scores.items() if score >= 2.0
    }


def _iter_tokens(*, text: str) -> tuple[str, ...]:
    tokens = re.findall(r"[^\W\d_]{2,}", text, flags=re.UNICODE)
    if tokens:
        return tuple(tokens)
    return tuple(match.group(0) for match in re.finditer(r"\w+", text))


def _script_counts(*, text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        script = _script_for_char(char=char)
        counts[script] = counts.get(script, 0) + 1
    return counts


def _primary_script(*, script_counts: dict[str, int]) -> str:
    if not script_counts:
        return "Unknown"
    return max(script_counts.items(), key=lambda item: item[1])[0]


def _is_mixed_script(*, script_counts: dict[str, int]) -> bool:
    total = sum(script_counts.values())
    if total <= 0 or len(script_counts) <= 1:
        return False
    primary_count = max(script_counts.values())
    return (total - primary_count) / total > 0.2


def _script_for_char(*, char: str) -> str:
    name = unicodedata.name(char, "")
    for script in (
        "LATIN",
        "CYRILLIC",
        "ARABIC",
        "DEVANAGARI",
        "HIRAGANA",
        "KATAKANA",
        "HANGUL",
    ):
        if script in name:
            return script.title()
    if "CJK UNIFIED" in name or "IDEOGRAPH" in name:
        return "Han"
    return "Other"


def _language_for_script(*, script: str) -> str | None:
    if script == "Cyrillic":
        return "ru"
    if script == "Han":
        return "zh"
    if script in {"Hiragana", "Katakana"}:
        return "ja"
    if script == "Arabic":
        return "ar"
    if script == "Devanagari":
        return "hi"
    if script == "Hangul":
        return "ko"
    return None


_LATIN_MARKERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        "the and of to in for with from that this are was were have has "
        "project page contact normal extraction ordinary words".split()
    ),
    "nl": frozenset(
        "de het een en van in is dat voor met op zijn niet als door deze "
        "tekst pagina project vandaag verwerking".split()
    ),
    "de": frozenset(
        "der die das und ist nicht mit den von zu ein eine für auf im dem "
        "diese seite projekt verarbeitung".split()
    ),
    "fr": frozenset(
        "le la les des et est dans pour avec une un que qui sur par cette "
        "page projet traitement".split()
    ),
    "es": frozenset(
        "el la los las de y en que para con una un por esta página proyecto "
        "procesamiento".split()
    ),
    "pt": frozenset(
        "o a os as de e em que para com uma um por esta página projeto "
        "processamento".split()
    ),
    "it": frozenset(
        "il la lo gli le di e in che per con una un questa pagina progetto "
        "elaborazione".split()
    ),
}


@dataclass(frozen=True, slots=True)
class PreprocessingLanguageDetectionResult:
    """Resolved language evidence for the final released text."""

    language: str | None
    language_confidence: float
    script: str
    is_mixed_language: bool
    language_source: str
    used_crawler_evidence: bool
    verification_mismatch: bool
    language_detector_version: str | None


def build_language_annotation(
    *,
    detection: LanguageDetectionResult,
    text: str,
    script: str,
    mixed_script: bool,
    language_source: str | None = None,
    used_crawler_evidence: bool = False,
    verification_mismatch: bool = False,
    language_detector_version: str | None = None,
) -> PreprocessingLanguageDetectionResult:
    """Map detection output to preprocessing annotation format."""

    is_mixed_language = mixed_script
    if script == "Latin" and detection.detector_name in {
        "latin_marker",
        "stopword",
        "language_hint",
    }:
        is_mixed_language = is_mixed_language or is_mixed_latin_language(
            text=text,
        )

    return PreprocessingLanguageDetectionResult(
        language=detection.language_code,
        language_confidence=detection.confidence,
        script=script,
        is_mixed_language=is_mixed_language,
        language_source=language_source or detection.detector_name,
        used_crawler_evidence=used_crawler_evidence,
        verification_mismatch=verification_mismatch,
        language_detector_version=language_detector_version,
    )


def verify_or_detect_language(
    *,
    detector: LanguageDetector,
    text: str,
    evidence: LanguageEvidence | None,
    trusted_confidence: float = 0.9,
    mismatch_confidence: float = 0.65,
) -> PreprocessingLanguageDetectionResult:
    """Verify trusted crawler evidence and detect only as fallback/override."""

    script, mixed_script = detect_script_profile(text=text)
    detected = detector.detect(text=text)
    crawler_language = (
        evidence.language.strip().lower()
        if evidence is not None and evidence.language
        else None
    )
    crawler_confidence = (
        evidence.confidence
        if evidence is not None and evidence.confidence is not None
        else 0.0
    )

    if crawler_language is None:
        return build_language_annotation(
            detection=detected,
            text=text,
            script=script,
            mixed_script=mixed_script,
            language_source=f"preprocessing_fallback:{detected.detector_name}",
            language_detector_version=LANGUAGE_DETECTOR_VERSION,
        )

    mismatch = bool(
        detected.language_code
        and detected.language_code != crawler_language
        and detected.confidence >= mismatch_confidence
    )
    if mismatch or crawler_confidence < trusted_confidence:
        if detected.language_code is not None:
            return build_language_annotation(
                detection=detected,
                text=text,
                script=script,
                mixed_script=mixed_script,
                language_source=(
                    f"preprocessing_override:{detected.detector_name}"
                    if mismatch
                    else f"preprocessing_fallback:{detected.detector_name}"
                ),
                used_crawler_evidence=False,
                verification_mismatch=mismatch,
                language_detector_version=LANGUAGE_DETECTOR_VERSION,
            )

    accepted = LanguageDetectionResult(
        language_code=crawler_language,
        confidence=crawler_confidence,
        detector_name=(
            evidence.source
            if evidence is not None and evidence.source
            else "crawler_language"
        ),
        text_length=len(text.strip()),
    )
    return build_language_annotation(
        detection=accepted,
        text=text,
        script=script,
        mixed_script=mixed_script,
        language_source=accepted.detector_name,
        used_crawler_evidence=True,
        verification_mismatch=False,
        language_detector_version=(
            evidence.detector_version if evidence is not None else None
        ),
    )
