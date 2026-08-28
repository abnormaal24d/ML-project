"""Final text language handling verifies crawler evidence and falls back."""

from __future__ import annotations

from preprocessing.preprocessing_input import LanguageEvidence
from preprocessing.text.text_language import (
    LanguageDetector,
    verify_or_detect_language,
)


def test_trusted_matching_crawler_language_is_retained() -> None:
    result = verify_or_detect_language(
        detector=LanguageDetector(),
        text=(
            "This is an English article about software systems and data "
            "processing with reliable operational controls."
        ),
        evidence=LanguageEvidence(
            language="en",
            confidence=0.98,
            source="crawler_fasttext",
            detector_version="fasttext-lid-v1",
        ),
    )

    assert result.language == "en"
    assert result.used_crawler_evidence is True
    assert result.verification_mismatch is False
    assert result.language_detector_version == "fasttext-lid-v1"
    assert result.language_source == "crawler_fasttext"


def test_conflicting_crawler_language_is_overridden() -> None:
    result = verify_or_detect_language(
        detector=LanguageDetector(),
        text=(
            "Dit is een Nederlands artikel over systemen en gegevens. "
            "Het project gebruikt betrouwbare verwerking voor de gebruiker."
        ),
        evidence=LanguageEvidence(
            language="en",
            confidence=0.99,
            source="html_lang",
        ),
        mismatch_confidence=0.35,
    )

    assert result.language == "nl"
    assert result.used_crawler_evidence is False
    assert result.verification_mismatch is True
    assert result.language_detector_version == "preprocessing-heuristic-v1"
    assert result.language_source.startswith("preprocessing_override:")


def test_missing_crawler_evidence_uses_final_text_fallback() -> None:
    result = verify_or_detect_language(
        detector=LanguageDetector(),
        text=(
            "This is an English article about a project and the processing "
            "of information for users."
        ),
        evidence=None,
    )

    assert result.language == "en"
    assert result.used_crawler_evidence is False
    assert result.verification_mismatch is False
    assert result.language_detector_version == "preprocessing-heuristic-v1"
    assert result.language_source.startswith("preprocessing_fallback:")
