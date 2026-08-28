"""Shared execution boundary for configured OCR backends."""

from __future__ import annotations

from collections.abc import Buffer, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any, SupportsFloat, SupportsIndex, cast

from config.preprocessing.media_settings import OcrBackendSettings
from preprocessing.media.ocr.ocr_result import (
    OcrLine,
    OcrOrigin,
    OcrWord,
    OpticalCharacterRecognitionResult,
)
from preprocessing.provenance import (
    ProducerProvenance,
    ProducerType,
    hash_parameters,
)


class OcrBackendUnavailable(RuntimeError):
    """Configured OCR backend is not installed or cannot be started."""


class OcrBackendFailure(RuntimeError):
    """OCR backend failed while processing an otherwise valid input."""


def _build_backend_result(
    *,
    text: str,
    confidence: float | None,
    origin: OcrOrigin,
    producer_name: str,
    producer_version: str,
    model_id: str,
    model_revision: str | None,
    artifact_hash: str | None,
    source_hash: str,
    lines: tuple[OcrLine, ...],
    words: tuple[OcrWord, ...],
    timeout_seconds: float,
) -> OpticalCharacterRecognitionResult:
    output_hash = hash_parameters(
        {
            "text": text,
            "lines": [line.text for line in lines],
            "origin": origin.value,
        }
    )
    provenance = ProducerProvenance(
        producer_type=ProducerType.EXTERNAL_MODEL,
        producer_name=producer_name,
        producer_version=producer_version,
        model_id=model_id,
        model_revision=model_revision,
        artifact_hash=artifact_hash,
        parameters_hash=hash_parameters(
            {"backend": producer_name, "timeout_seconds": timeout_seconds}
        ),
        confidence=confidence,
        warnings=(),
        source_hash=source_hash,
        output_hash=output_hash,
    )
    return OpticalCharacterRecognitionResult(
        text=text,
        confidence=confidence,
        origin=origin,
        provenance=provenance,
        lines=lines,
        words=words,
        engine=producer_name,
    )


def _tesseract_words(
    *,
    data: dict[str, Sequence[Any]],
    revision: str,
) -> tuple[OcrWord, ...]:
    texts = data.get("text", ())
    words: list[OcrWord] = []
    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue
        words.append(
            OcrWord(
                text=text,
                confidence=_normalized_confidence(
                    _at(data.get("conf", ()), index)
                ),
                origin=OcrOrigin.TESSERACT,
                producer_revision=revision,
                box=(
                    _as_float(_at(data.get("left", ()), index)),
                    _as_float(_at(data.get("top", ()), index)),
                    _as_float(_at(data.get("width", ()), index)),
                    _as_float(_at(data.get("height", ()), index)),
                ),
            )
        )
    return tuple(words)


def _lines_from_words(
    *,
    words: tuple[OcrWord, ...],
    origin: OcrOrigin,
    producer_revision: str,
) -> tuple[OcrLine, ...]:
    return (
        OcrLine(
            text=" ".join(word.text for word in words),
            confidence=_average(tuple(word.confidence for word in words)),
            origin=origin,
            producer_revision=producer_revision,
            words=words,
        ),
    )


def _verify_backend_version(
    *,
    settings: OcrBackendSettings,
    observed: str,
) -> None:
    expected = settings.backend_version
    if expected is not None and expected != observed:
        raise OcrBackendFailure("ocr_backend_version_mismatch")


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def _normalized_confidence(value: object) -> float | None:
    try:
        confidence = float(
            cast(str | Buffer | SupportsFloat | SupportsIndex, value)
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if confidence < 0.0:
        return None
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def _rapidocr_box(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    points = [
        point
        for point in value
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    if not points:
        return None
    xs = [_as_float(point[0]) for point in points]
    ys = [_as_float(point[1]) for point in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _average(values: tuple[float | None, ...]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _at(values: Sequence[Any], index: int) -> object:
    return values[index] if index < len(values) else None


def _as_float(value: object) -> float:
    try:
        return float(cast(str | Buffer | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def package_version(package: str) -> str:
    return _package_version(package)


def verify_backend_version(
    *, settings: OcrBackendSettings, observed: str
) -> None:
    return _verify_backend_version(settings=settings, observed=observed)
