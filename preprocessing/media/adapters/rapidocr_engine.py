"""Concrete RapidOCR backend adapter."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from io import BytesIO
from threading import BoundedSemaphore, Thread
from time import monotonic
from typing import Protocol, cast

from config.preprocessing.media_settings import OcrBackendSettings
from preprocessing.media.ocr import ocr_backend_execution
from preprocessing.media.ocr.ocr_backend_execution import (
    OcrBackendFailure,
    OcrBackendUnavailable,
    _average,
    _build_backend_result,
    _normalized_confidence,
    _rapidocr_box,
)
from preprocessing.media.ocr.ocr_result import (
    OcrLine,
    OcrOrigin,
    OpticalCharacterRecognitionResult,
)
from preprocessing.provenance import hash_bytes


class _RapidOcrBackend(Protocol):
    def __call__(
        self,
        body: bytes,
    ) -> tuple[Sequence[Sequence[object]] | None, object]: ...


class RapidOcrEngine:
    """RapidOCR adapter: None means no text, exception means backend failure."""

    def __init__(self, *, settings: OcrBackendSettings) -> None:
        self._settings = settings
        self._backend: _RapidOcrBackend | None = None
        self._backend_revision: str | None = None
        self._execution_slot = BoundedSemaphore(value=1)

    def extract(
        self,
        *,
        image_bytes: bytes,
    ) -> OpticalCharacterRecognitionResult | None:
        if not image_bytes:
            return None
        return self._run_bounded(body=image_bytes)

    def _run_bounded(
        self,
        *,
        body: bytes,
    ) -> OpticalCharacterRecognitionResult | None:
        timeout_seconds = self._settings.timeout_seconds
        deadline = monotonic() + timeout_seconds
        if not self._execution_slot.acquire(timeout=timeout_seconds):
            raise OcrBackendFailure("rapidocr_timeout")

        result: Future[OpticalCharacterRecognitionResult | None] = Future()

        def execute() -> None:
            try:
                result.set_result(self._extract_image(body=body))
            except Exception as exc:  # noqa: BLE001 — propagate backend error
                result.set_exception(exc)
            finally:
                self._execution_slot.release()

        try:
            Thread(
                target=execute,
                name="rapidocr-worker",
                daemon=True,
            ).start()
        except RuntimeError as exc:
            self._execution_slot.release()
            raise OcrBackendFailure("rapidocr_worker_start_failed") from exc

        remaining_seconds = max(0.0, deadline - monotonic())
        try:
            return result.result(timeout=remaining_seconds)
        except FutureTimeout as exc:
            raise OcrBackendFailure("rapidocr_timeout") from exc

    def extract_pil(
        self,
        *,
        image: object,
        source_hash: str,
    ) -> OpticalCharacterRecognitionResult | None:
        del source_hash
        output = BytesIO()
        save = getattr(image, "save", None)
        if not callable(save):
            raise OcrBackendFailure("rapidocr_image_serialization_failed")
        save(output, format="PNG")
        return self.extract(image_bytes=output.getvalue())

    def _extract_image(
        self,
        *,
        body: bytes,
    ) -> OpticalCharacterRecognitionResult | None:
        backend, revision = self._initialized_backend()
        return _extract_image_with_backend(
            body=body,
            settings=self._settings,
            backend=backend,
            revision=revision,
        )

    def _initialized_backend(self) -> tuple[_RapidOcrBackend, str]:
        backend = self._backend
        revision = self._backend_revision
        if backend is None or revision is None:
            backend, revision = _load_backend(settings=self._settings)
            self._backend = backend
            self._backend_revision = revision
        return backend, revision


def extract_image_with_rapidocr(
    *,
    body: bytes,
    settings: OcrBackendSettings,
) -> OpticalCharacterRecognitionResult | None:
    backend, revision = _load_backend(settings=settings)
    return _extract_image_with_backend(
        body=body,
        settings=settings,
        backend=backend,
        revision=revision,
    )


def _load_backend(
    *,
    settings: OcrBackendSettings,
) -> tuple[_RapidOcrBackend, str]:
    try:
        from rapidocr_onnxruntime import (  # type: ignore[import-untyped]
            RapidOCR,
        )
    except ImportError as exc:
        raise OcrBackendUnavailable("rapidocr_unavailable") from exc

    revision = ocr_backend_execution.package_version("rapidocr-onnxruntime")
    ocr_backend_execution.verify_backend_version(
        settings=settings, observed=revision
    )
    try:
        backend = cast(_RapidOcrBackend, RapidOCR())
    except Exception as exc:  # noqa: BLE001 — native OCR runtime failures
        raise OcrBackendUnavailable("rapidocr_initialization_failed") from exc
    return backend, revision


def _extract_image_with_backend(
    *,
    body: bytes,
    settings: OcrBackendSettings,
    backend: _RapidOcrBackend,
    revision: str,
) -> OpticalCharacterRecognitionResult | None:
    try:
        raw_result, _ = backend(body)
    except Exception as exc:  # noqa: BLE001 — native OCR runtime failures
        raise OcrBackendFailure("rapidocr_execution_failed") from exc
    if not raw_result:
        return None

    lines: list[OcrLine] = []
    for item in raw_result:
        if len(item) < 2:
            continue
        text = str(item[1]).strip()
        if not text:
            continue
        confidence = _normalized_confidence(item[2] if len(item) > 2 else None)
        lines.append(
            OcrLine(
                text=text,
                confidence=confidence,
                origin=OcrOrigin.RAPIDOCR,
                producer_revision=revision,
                box=_rapidocr_box(item[0]),
            )
        )
    if not lines:
        return None

    text = " ".join(line.text for line in lines)
    confidence = _average(tuple(line.confidence for line in lines))
    return _build_backend_result(
        text=text,
        confidence=confidence,
        origin=OcrOrigin.RAPIDOCR,
        producer_name="rapidocr",
        producer_version=revision,
        model_id=settings.model_id or "rapidocr-onnxruntime-default",
        model_revision=settings.model_revision,
        artifact_hash=settings.model_artifact_hash,
        source_hash=hash_bytes(body),
        lines=tuple(lines),
        words=(),
        timeout_seconds=settings.timeout_seconds,
    )
