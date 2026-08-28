"""OCR orchestration: engine selection, result builders, and merging."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from preprocessing.media.ocr.ocr_backend_execution import (
    OcrBackendFailure,
    OcrBackendUnavailable,
)
from preprocessing.media.ocr.ocr_result import (
    OcrLine,
    OcrOrigin,
    OpticalCharacterRecognitionResult,
)
from preprocessing.provenance import (
    ProducerProvenance,
    ProducerType,
    hash_parameters,
    hash_text,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "OcrBackendFailure",
    "OcrBackendUnavailable",
    "OcrEngine",
    "build_text_result",
    "merge_ocr_results",
)


class OcrEngine(Protocol):
    """One OCR implementation chosen by composition."""

    def extract(
        self,
        *,
        image_bytes: bytes,
    ) -> OpticalCharacterRecognitionResult | None:
        """Return OCR text, or None when the backend finds no text."""

    def extract_pil(
        self,
        *,
        image: object,
        source_hash: str,
    ) -> OpticalCharacterRecognitionResult | None:
        """OCR a PIL-like image, or None when no text is found."""


def build_text_result(
    *,
    text: str,
    origin: OcrOrigin,
    producer_revision: str,
    confidence: float | None = None,
    source_hash: str | None = None,
) -> OpticalCharacterRecognitionResult:
    """Build source/native/human OCR-shaped text with an explicit origin."""

    normalized = " ".join(text.split())
    resolved_source_hash = source_hash or hash_text(normalized)
    producer_type = {
        OcrOrigin.HUMAN_VERIFIED: ProducerType.HUMAN,
        OcrOrigin.SIDECAR: ProducerType.SOURCE,
        OcrOrigin.NATIVE_TEXT_LAYER: ProducerType.SOURCE,
    }.get(origin, ProducerType.EXTERNAL_MODEL)
    provenance = ProducerProvenance(
        producer_type=producer_type,
        producer_name=origin.value,
        producer_version=producer_revision,
        model_id=None,
        model_revision=None,
        artifact_hash=None,
        parameters_hash=hash_parameters({"origin": origin.value}),
        confidence=confidence,
        warnings=(),
        source_hash=resolved_source_hash,
        output_hash=hash_text(normalized),
    )
    line = OcrLine(
        text=normalized,
        confidence=confidence,
        origin=origin,
        producer_revision=producer_revision,
    )
    return OpticalCharacterRecognitionResult(
        text=normalized,
        confidence=confidence,
        origin=origin,
        provenance=provenance,
        lines=(line,),
        words=(),
        engine=origin.value,
    )


def merge_ocr_results(
    *,
    results: Sequence[OpticalCharacterRecognitionResult],
) -> OpticalCharacterRecognitionResult | None:
    """Merge multiple OCR results into one concatenated text result."""

    usable = [result for result in results if result.text.strip()]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]

    text = " ".join(result.text.strip() for result in usable)
    confidences = [
        result.confidence for result in usable if result.confidence is not None
    ]
    confidence = sum(confidences) / len(confidences) if confidences else None
    lines = tuple(line for result in usable for line in result.lines)
    words = tuple(word for result in usable for word in result.words)
    primary = usable[0]
    return replace(
        primary,
        text=text,
        confidence=confidence,
        lines=lines,
        words=words,
        provenance=replace(
            primary.provenance,
            output_hash=hash_text(text),
            confidence=confidence,
            warnings=tuple(
                warning
                for result in usable
                for warning in result.provenance.warnings
            ),
        ),
    )
