"""Concrete Tesseract OCR backend adapter."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from config.errors import RuntimeDependencyError
from config.preprocessing.media_settings import OcrBackendSettings
from preprocessing.media.ocr import ocr_backend_execution
from preprocessing.media.ocr.ocr_backend_execution import (
    OcrBackendFailure,
    OcrBackendUnavailable,
    _average,
    _build_backend_result,
    _lines_from_words,
    _tesseract_words,
)
from preprocessing.media.ocr.ocr_result import (
    OcrOrigin,
    OpticalCharacterRecognitionResult,
)
from preprocessing.provenance import hash_bytes, hash_file


def validate_tesseract_artifact(*, settings: OcrBackendSettings) -> None:
    """Validate the configured production Tesseract model artifact."""

    if not settings.production_mode or settings.backend == "disabled":
        return
    expected_hash = settings.model_artifact_hash
    configured_path = settings.model_artifact_path
    if expected_hash is None or configured_path is None:
        raise RuntimeDependencyError(
            "tesseract_artifact_pin_missing",
            setting="preprocessing.ocr.model_artifact_path",
            issue="tesseract_artifact_pin_missing",
        )
    artifact_path = Path(configured_path)
    if not artifact_path.is_absolute() or not artifact_path.is_file():
        raise RuntimeDependencyError(
            "tesseract_artifact_missing",
            setting="preprocessing.ocr.model_artifact_path",
            required_artifact=artifact_path.name or "model_artifact",
            issue="tesseract_artifact_missing",
        )
    if hash_file(artifact_path) != expected_hash:
        raise RuntimeDependencyError(
            "tesseract_artifact_hash_mismatch",
            setting="preprocessing.ocr.model_artifact_hash",
            required_artifact=artifact_path.name,
            issue="tesseract_artifact_hash_mismatch",
        )


class TesseractOcrEngine:
    """Tesseract adapter: None means no text, exception means backend failure."""

    def __init__(self, *, settings: OcrBackendSettings) -> None:
        self._settings = settings
        try:
            validate_tesseract_artifact(settings=settings)
        except RuntimeDependencyError as error:
            raise OcrBackendUnavailable(error.issue) from error

    def extract(
        self,
        *,
        image_bytes: bytes,
    ) -> OpticalCharacterRecognitionResult | None:
        if not image_bytes:
            return None
        return extract_image_with_pytesseract(
            body=image_bytes,
            settings=self._settings,
        )

    def extract_pil(
        self,
        *,
        image: object,
        source_hash: str,
    ) -> OpticalCharacterRecognitionResult | None:
        return extract_pil_with_tesseract(
            image=image,
            source_hash=source_hash,
            settings=self._settings,
        )


def extract_image_with_pytesseract(
    *,
    body: bytes,
    settings: OcrBackendSettings,
) -> OpticalCharacterRecognitionResult | None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OcrBackendUnavailable("pillow_unavailable") from exc
    try:
        image = Image.open(BytesIO(body))
        return extract_pil_with_tesseract(
            image=image,
            source_hash=hash_bytes(body),
            settings=settings,
        )
    except (OcrBackendUnavailable, OcrBackendFailure):
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise OcrBackendFailure("tesseract_image_failed") from exc


def extract_pil_with_tesseract(
    *,
    image: object,
    source_hash: str,
    settings: OcrBackendSettings,
) -> OpticalCharacterRecognitionResult | None:
    try:
        import pytesseract  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OcrBackendUnavailable("pytesseract_unavailable") from exc

    try:
        revision = str(pytesseract.get_tesseract_version())
        ocr_backend_execution.verify_backend_version(
            settings=settings, observed=revision
        )
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            timeout=settings.timeout_seconds,
        )
    except RuntimeError as exc:
        raise OcrBackendFailure(
            "tesseract_timeout_or_runtime_failure"
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise OcrBackendFailure("tesseract_failed") from exc

    words = _tesseract_words(data=data, revision=revision)
    if not words:
        return None
    lines = _lines_from_words(
        words=words,
        origin=OcrOrigin.TESSERACT,
        producer_revision=revision,
    )
    text = " ".join(line.text for line in lines)
    confidence = _average(tuple(word.confidence for word in words))
    return _build_backend_result(
        text=text,
        confidence=confidence,
        origin=OcrOrigin.TESSERACT,
        producer_name="tesseract",
        producer_version=revision,
        model_id=settings.model_id or "tesseract-language-data",
        model_revision=settings.model_revision,
        artifact_hash=settings.model_artifact_hash,
        source_hash=source_hash,
        lines=lines,
        words=words,
        timeout_seconds=settings.timeout_seconds,
    )
