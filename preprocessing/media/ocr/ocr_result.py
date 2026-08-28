"""Canonical OCR result and span schemas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from preprocessing.provenance import ProducerProvenance


class OcrOrigin(StrEnum):
    """Auditable origin of text represented as an OCR span."""

    NATIVE_TEXT_LAYER = "native_text_layer"
    SIDECAR = "sidecar"
    TESSERACT = "tesseract"
    RAPIDOCR = "rapidocr"
    HUMAN_VERIFIED = "human_verified"


@dataclass(frozen=True, slots=True)
class OcrSpan:
    text: str
    confidence: float | None
    origin: OcrOrigin
    producer_revision: str
    box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("OCR span text must not be blank")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("OCR span confidence must be between 0 and 1")
        if not self.producer_revision.strip():
            raise ValueError("OCR span producer_revision must not be blank")
        if self.box is not None and len(self.box) != 4:
            raise ValueError("OCR span box must contain four coordinates")


@dataclass(frozen=True, slots=True)
class OcrWord(OcrSpan):
    """Word-level OCR span."""


@dataclass(frozen=True, slots=True)
class OcrLine(OcrSpan):
    """Line-level OCR span with optional constituent words."""

    words: tuple[OcrWord, ...] = ()


@dataclass(frozen=True, slots=True)
class OpticalCharacterRecognitionResult:
    """OCR payload coupled to canonical producer provenance."""

    text: str
    confidence: float | None
    origin: OcrOrigin
    provenance: ProducerProvenance
    language: str | None = None
    lines: tuple[OcrLine, ...] = ()
    words: tuple[OcrWord, ...] = ()
    frame_results: tuple[dict[str, Any], ...] = ()
    engine: str = "unknown"
    raw_layout: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("OCR result text must not be blank")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("OCR result confidence must be between 0 and 1")
        if not self.lines and not self.words:
            raise ValueError("OCR result requires at least one auditable span")
        for span in (*self.lines, *self.words):
            if span.origin is not self.origin:
                raise ValueError("OCR result and span origins must match")

    @property
    def spans(self) -> tuple[OcrSpan, ...]:
        return (*self.lines, *self.words)

    @property
    def producer_revision(self) -> str:
        return (
            self.provenance.model_revision or self.provenance.producer_version
        )
