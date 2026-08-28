"""Raw preprocessing input models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PreprocessingModality = Literal["text", "image", "audio", "video", "document"]


@dataclass(frozen=True, slots=True)
class ExtractedTextContent:
    """Crawler-owned structural page text used by preprocessing.

    For pages this is loaded from a versioned extraction sidecar. Preprocessing
    must not re-parse HTML to reconstruct these fields.
    """

    text: str
    markdown: str
    headings: tuple[str, ...]
    code_block_count: int
    boilerplate_ratio: float
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LanguageEvidence:
    """Crawler language decision with confidence and detector provenance."""

    language: str | None
    confidence: float | None = None
    source: str | None = None
    detector_version: str | None = None

    def __post_init__(self) -> None:
        if all(
            value is None
            for value in (
                self.language,
                self.confidence,
                self.source,
                self.detector_version,
            )
        ):
            raise ValueError(
                "language evidence must contain at least one evidence field"
            )

        if self.language is not None:
            if not self.language:
                raise ValueError("language must not be empty")
            if self.language != self.language.strip().lower():
                raise ValueError(
                    "language must be a normalized lowercase code"
                )

        for field_name, value in (
            ("source", self.source),
            ("detector_version", self.detector_version),
        ):
            if value is None:
                continue
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            if value != value.strip():
                raise ValueError(f"{field_name} must be stripped")

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("language confidence must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class PreprocessingInput:
    """Raw preprocessing input with modality-aware payload metadata."""

    source_id: str
    source_url: str
    normalized_url: str
    domain: str
    path: str
    encoding: str | None = None
    title: str | None = None
    modality: PreprocessingModality = "text"
    mime_type: str | None = None
    media_path: str | None = None
    byte_size: int | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    transcript_text: str | None = None
    ocr_text: str | None = None
    extracted_text_content: ExtractedTextContent | None = None
    payload: dict[str, object] = field(default_factory=dict)
    language_evidence: LanguageEvidence | None = None

    def resolved_language_evidence(self) -> LanguageEvidence | None:
        """Return structured crawler language evidence."""

        return self.language_evidence

    def resolved_language(self) -> str | None:
        """Return the resolved language from the crawler evidence."""

        evidence = self.language_evidence
        if evidence is None:
            return None
        return evidence.language
