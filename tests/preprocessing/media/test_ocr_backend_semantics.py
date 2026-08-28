"""OCR engines report empty text as None, not as backend failure."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from config.preprocessing.media_settings import OcrBackendSettings
from orchestration.composition.preprocessing_dependencies import (
    build_ocr_engine,
)
from preprocessing.media.adapters.rapidocr_engine import (
    RapidOcrEngine,
    extract_image_with_rapidocr,
)
from preprocessing.media.adapters.tesseract_engine import (
    TesseractOcrEngine,
    extract_pil_with_tesseract,
)
from preprocessing.media.ocr.ocr_engine import (
    OcrBackendFailure,
    OcrBackendUnavailable,
)


def test_rapidocr_no_detections_returns_none() -> None:
    settings = OcrBackendSettings(backend="rapidocr")

    class _FakeRapid:
        def __call__(self, body: bytes):
            del body
            return None, None

    with (
        patch.dict(
            "sys.modules",
            {
                "rapidocr_onnxruntime": SimpleNamespace(
                    RapidOCR=lambda: _FakeRapid()
                )
            },
        ),
        patch(
            "preprocessing.media.ocr.ocr_backend_execution.package_version",
            return_value="1.4.4",
        ),
        patch(
            "preprocessing.media.ocr.ocr_backend_execution.verify_backend_version",
        ),
    ):
        result = extract_image_with_rapidocr(
            body=b"fake-image-bytes",
            settings=settings,
        )
    assert result is None


def test_rapidocr_runtime_exception_raises_backend_failure() -> None:
    settings = OcrBackendSettings(backend="rapidocr")

    class _Boom:
        def __call__(self, body: bytes):
            del body
            raise RuntimeError("model crashed")

    with (
        patch.dict(
            "sys.modules",
            {
                "rapidocr_onnxruntime": SimpleNamespace(
                    RapidOCR=lambda: _Boom()
                )
            },
        ),
        patch(
            "preprocessing.media.ocr.ocr_backend_execution.package_version",
            return_value="1.4.4",
        ),
        patch(
            "preprocessing.media.ocr.ocr_backend_execution.verify_backend_version",
        ),
        pytest.raises(OcrBackendFailure, match="rapidocr_execution_failed"),
    ):
        extract_image_with_rapidocr(
            body=b"fake-image-bytes",
            settings=settings,
        )


def test_rapidocr_engine_reuses_initialized_backend() -> None:
    settings = OcrBackendSettings(backend="rapidocr")
    initialization_count = 0

    class _FakeRapid:
        def __call__(self, body: bytes):
            assert body
            return None, None

    def _build_backend() -> _FakeRapid:
        nonlocal initialization_count
        initialization_count += 1
        return _FakeRapid()

    with (
        patch.dict(
            "sys.modules",
            {"rapidocr_onnxruntime": SimpleNamespace(RapidOCR=_build_backend)},
        ),
        patch(
            "preprocessing.media.ocr.ocr_backend_execution.package_version",
            return_value="1.4.4",
        ),
        patch(
            "preprocessing.media.ocr.ocr_backend_execution.verify_backend_version",
        ),
    ):
        engine = RapidOcrEngine(settings=settings)
        assert engine.extract(image_bytes=b"first") is None
        assert engine.extract(image_bytes=b"second") is None

    assert initialization_count == 1


def test_rapidocr_timeout_does_not_start_overlapping_native_work() -> None:
    settings = OcrBackendSettings(
        backend="rapidocr",
        timeout_seconds=0.05,
    )
    engine = RapidOcrEngine(settings=settings)
    started = Event()
    release = Event()
    completed = Event()

    def _blocked_extract(*, body: bytes):
        assert body
        started.set()
        try:
            release.wait(timeout=2.0)
        finally:
            completed.set()
        return None

    with patch.object(
        engine,
        "_extract_image",
        side_effect=_blocked_extract,
    ) as extract:
        with pytest.raises(OcrBackendFailure, match="rapidocr_timeout"):
            engine.extract(image_bytes=b"first")
        assert started.is_set()

        with pytest.raises(OcrBackendFailure, match="rapidocr_timeout"):
            engine.extract(image_bytes=b"second")
        assert extract.call_count == 1

        release.set()
        assert completed.wait(timeout=1.0)


def test_tesseract_no_words_returns_none() -> None:
    settings = OcrBackendSettings(backend="tesseract")
    fake_data = {
        "text": ["", "  ", ""],
        "conf": ["-1", "-1", "-1"],
        "left": [0, 0, 0],
        "top": [0, 0, 0],
        "width": [0, 0, 0],
        "height": [0, 0, 0],
    }
    fake_pytesseract = SimpleNamespace(
        get_tesseract_version=lambda: "5.0.0",
        image_to_data=lambda *a, **k: fake_data,
        Output=SimpleNamespace(DICT="dict"),
    )
    with (
        patch.dict("sys.modules", {"pytesseract": fake_pytesseract}),
        patch(
            "preprocessing.media.ocr.ocr_backend_execution.verify_backend_version",
        ),
    ):
        result = extract_pil_with_tesseract(
            image=object(),
            source_hash="a" * 64,
            settings=settings,
        )
    assert result is None


def test_tesseract_uses_backend_native_timeout() -> None:
    settings = OcrBackendSettings(backend="tesseract", timeout_seconds=7.5)
    observed_timeout: float | None = None

    def _image_to_data(*args, **kwargs):
        del args
        nonlocal observed_timeout
        observed_timeout = kwargs["timeout"]
        return {"text": []}

    fake_pytesseract = SimpleNamespace(
        get_tesseract_version=lambda: "5.0.0",
        image_to_data=_image_to_data,
        Output=SimpleNamespace(DICT="dict"),
    )
    with (
        patch.dict("sys.modules", {"pytesseract": fake_pytesseract}),
        patch(
            "preprocessing.media.ocr.ocr_backend_execution.verify_backend_version",
        ),
    ):
        result = extract_pil_with_tesseract(
            image=object(),
            source_hash="a" * 64,
            settings=settings,
        )

    assert result is None
    assert observed_timeout == 7.5


def test_build_ocr_engine_selects_concrete_implementations() -> None:
    rapid = build_ocr_engine(settings=OcrBackendSettings(backend="rapidocr"))
    tess = build_ocr_engine(settings=OcrBackendSettings(backend="tesseract"))
    disabled = build_ocr_engine(
        settings=OcrBackendSettings(backend="disabled")
    )
    assert isinstance(rapid, RapidOcrEngine)
    assert isinstance(tess, TesseractOcrEngine)
    assert disabled is None


def test_production_tesseract_rejects_wrong_artifact_hash(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "eng.traineddata"
    artifact.write_bytes(b"not-the-approved-model")
    settings = OcrBackendSettings(
        backend="tesseract",
        backend_version="5.3.0",
        model_id="tesseract-eng",
        model_revision="1:4.1.0-2",
        model_artifact_hash="0" * 64,
        model_artifact_path=str(artifact),
        production_mode=True,
    )

    with pytest.raises(
        OcrBackendUnavailable,
        match="tesseract_artifact_hash_mismatch",
    ):
        TesseractOcrEngine(settings=settings)


def test_production_rapidocr_rejects_unbound_composite_artifacts() -> None:
    with pytest.raises(
        ValidationError,
        match="detector, classifier, and recognizer artifacts",
    ):
        OcrBackendSettings(
            backend="rapidocr",
            backend_version="1.4.4",
            model_id="rapidocr-default",
            model_revision="1",
            model_artifact_hash="0" * 64,
            model_artifact_path="/models/rapidocr.onnx",
            production_mode=True,
        )


def test_diarization_auto_is_not_valid_config_value() -> None:
    from config.preprocessing.media_settings import DiarizationSettings

    with pytest.raises(ValidationError):
        DiarizationSettings(enabled=True, backend="auto")  # type: ignore[arg-type]
