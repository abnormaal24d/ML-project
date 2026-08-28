"""Payload extractor cutover for image/audio analyzers."""

from __future__ import annotations

import asyncio
import hashlib
import io
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from crawler.analysis.enrichment.audio.audio_analyzer import AudioAnalyzer
from crawler.analysis.enrichment.image.image_analyzer import (
    ImageAnalysisAssembler,
    ImageAnalyzer,
)
from crawler.classification.media_kind import MediaKind
from crawler.extraction.payloads.audio_payload_extractor import (
    AudioPayloadExtractor,
)
from crawler.extraction.payloads.image_payload_extractor import (
    ImagePayloadExtractionResult,
    ImagePayloadExtractor,
)
from crawler.fetching.errors.exceptions import IgnoredFetchError
from crawler.fetching.results.payload import FetchedPayload
from crawler.fetching.results.result import FetchResult
from tests.support.logging import TEST_LOGGER

_MAX_DECODE_PIXELS = 10_000


def _png_bytes(*, width: int = 4, height: int = 3) -> bytes:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        pytest.skip("Pillow not installed")
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def _wav_bytes(*, frames: int = 800, sample_rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


def _fetch_result(
    *,
    tmp_path: Path,
    body: bytes,
    kind: MediaKind,
    name: str,
    mime_type: str | None = None,
) -> FetchResult:
    path = tmp_path / name
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    resolved_mime = mime_type or {
        MediaKind.IMAGE: "image/png",
        MediaKind.AUDIO: "audio/wav",
        MediaKind.VIDEO: "video/mp4",
        MediaKind.DOCUMENT: "application/pdf",
        MediaKind.PAGE: "text/html",
        MediaKind.FEED: "application/rss+xml",
    }.get(kind, "application/octet-stream")
    return FetchResult(
        url=f"https://example.test/{name}",
        final_url=f"https://example.test/{name}",
        status_code=200,
        headers={"content-type": resolved_mime},
        fetched_at="2024-01-01T00:00:00Z",
        content_type=resolved_mime,
        mime_type=resolved_mime,
        encoding=None,
        language=None,
        kind=kind,
        payload=FetchedPayload(
            temp_path=path,
            byte_size=len(body),
            sha256_hex=digest,
            sniff_bytes=body[:64],
            chunk_count=1,
        ),
        body_sha256=digest,
    )


class _NoBlur:
    def estimate_blur(self, *, body: bytes):
        return SimpleNamespace(laplacian_variance=100.0)


class _UnexpectedBlur:
    def estimate_blur(self, *, body: bytes):
        del body
        raise AssertionError("blur estimator was called while disabled")


class _NoOcr:
    def extract(self, *, image_bytes: bytes):
        del image_bytes
        return None

    def extract_pil(self, *, image: object, source_hash: str):
        del image, source_hash
        return None


class _Resolver:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def resolve_path(self, *, result, suffix: str = ".audio"):
        del result, suffix
        return self._path

    def cleanup_owned_path(self, path: Path) -> None:
        del path


class _NoTranscription:
    async def transcribe_if_allowed(self, **kwargs):
        del kwargs
        return None


class _NoDiarization:
    def diarize(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            segments=(),
            speaker_count=0,
            overlapping_speech=False,
            model_name=None,
            model_version=None,
        )


class _NoEvents:
    def analyze(self, **kwargs):
        del kwargs
        return {
            "sound_events": (),
            "analysis_status": "not_run",
            "analysis_reasons": (),
        }


class _NoEmotion:
    def analyze(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            prosody=None,
            emotion_label=None,
            emotion_confidence=None,
            arousal=None,
            valence=None,
            dominance=None,
            analysis_status="not_run",
            analysis_reasons=(),
            model_name=None,
        )


def test_image_assembler_uses_payload_fields() -> None:
    payload = ImagePayloadExtractionResult(
        width=640,
        height=480,
        format="JPEG",
        color_mode="RGB",
        frame_count=1,
        exif_orientation=1,
        byte_size=100,
        sha256="b" * 64,
    )
    analysis = ImageAnalysisAssembler().build(
        payload=payload,
        blur_score=SimpleNamespace(laplacian_variance=42.5),
        extracted_text=None,
        ocr_result=None,
        fingerprint=None,
    )
    assert analysis.width == 640
    assert analysis.height == 480
    assert analysis.orientation == "landscape"
    assert analysis.metadata["image_format"] == "JPEG"
    assert analysis.metadata["sha256"] == "b" * 64
    assert analysis.blur_variance == 42.5


def test_image_analyzer_uses_payload_extractor(tmp_path: Path) -> None:
    body = _png_bytes(width=8, height=6)
    result = _fetch_result(
        tmp_path=tmp_path,
        body=body,
        kind=MediaKind.IMAGE,
        name="pic.png",
    )
    settings = SimpleNamespace(
        run_ocr=False,
        max_ocr_bytes=0,
        extract_metadata=False,
        detect_blur=True,
    )
    analyzer = ImageAnalyzer(
        settings=settings,  # type: ignore[arg-type]
        payload_extractor=ImagePayloadExtractor(
            max_decode_pixels=_MAX_DECODE_PIXELS,
        ),
        metadata_reader=None,
        blur_estimator=_NoBlur(),  # type: ignore[arg-type]
        ocr_engine=_NoOcr(),  # type: ignore[arg-type]
        logger=TEST_LOGGER,
    )
    analysis = asyncio.run(analyzer.analyze(result=result))
    assert analysis.width == 8
    assert analysis.height == 6
    assert analysis.metadata["image_format"] == "PNG"
    assert analysis.metadata["image_payload_bytes"] == len(body)
    assert analysis.blur_variance == 100.0


def test_image_analyzer_rejects_unreadable_payload(tmp_path: Path) -> None:
    result = _fetch_result(
        tmp_path=tmp_path,
        body=b"not-an-image",
        kind=MediaKind.IMAGE,
        name="bad.bin",
        mime_type="image/png",
    )
    settings = SimpleNamespace(
        run_ocr=False,
        max_ocr_bytes=0,
        extract_metadata=False,
        detect_blur=False,
    )
    analyzer = ImageAnalyzer(
        settings=settings,  # type: ignore[arg-type]
        payload_extractor=ImagePayloadExtractor(
            max_decode_pixels=_MAX_DECODE_PIXELS,
        ),
        metadata_reader=None,
        blur_estimator=_NoBlur(),  # type: ignore[arg-type]
        ocr_engine=_NoOcr(),  # type: ignore[arg-type]
        logger=TEST_LOGGER,
    )
    with pytest.raises(IgnoredFetchError, match="image_metadata_unreadable"):
        asyncio.run(analyzer.analyze(result=result))


def test_image_analyzer_rejects_unsupported_svg_mime(tmp_path: Path) -> None:
    result = _fetch_result(
        tmp_path=tmp_path,
        body=b"<svg xmlns='http://www.w3.org/2000/svg'></svg>",
        kind=MediaKind.IMAGE,
        name="icon.svg",
        mime_type="image/svg+xml",
    )
    settings = SimpleNamespace(
        run_ocr=False,
        max_ocr_bytes=0,
        extract_metadata=False,
        detect_blur=False,
    )
    analyzer = ImageAnalyzer(
        settings=settings,  # type: ignore[arg-type]
        payload_extractor=ImagePayloadExtractor(
            max_decode_pixels=_MAX_DECODE_PIXELS,
        ),
        metadata_reader=None,
        blur_estimator=_NoBlur(),  # type: ignore[arg-type]
        ocr_engine=_NoOcr(),  # type: ignore[arg-type]
        logger=TEST_LOGGER,
    )
    with pytest.raises(IgnoredFetchError, match="unsupported_image_format"):
        asyncio.run(analyzer.analyze(result=result))


def test_image_without_ocr_text_is_still_analyzed(tmp_path: Path) -> None:
    body = _png_bytes(width=8, height=6)
    result = _fetch_result(
        tmp_path=tmp_path,
        body=body,
        kind=MediaKind.IMAGE,
        name="blank.png",
        mime_type="image/png",
    )
    settings = SimpleNamespace(
        run_ocr=True,
        max_ocr_bytes=10_000_000,
        extract_metadata=False,
        detect_blur=False,
    )

    class _EmptyOcr:
        def extract(self, *, image_bytes: bytes):
            del image_bytes
            return None

        def extract_pil(self, *, image: object, source_hash: str):
            del image, source_hash
            return None

    analyzer = ImageAnalyzer(
        settings=settings,  # type: ignore[arg-type]
        payload_extractor=ImagePayloadExtractor(
            max_decode_pixels=_MAX_DECODE_PIXELS,
        ),
        metadata_reader=None,
        blur_estimator=_UnexpectedBlur(),  # type: ignore[arg-type]
        ocr_engine=_EmptyOcr(),  # type: ignore[arg-type]
        logger=TEST_LOGGER,
    )
    analysis = asyncio.run(analyzer.analyze(result=result))
    assert analysis.width == 8
    assert analysis.height == 6
    assert analysis.extracted_text is None
    assert analysis.ocr_result is None


def test_audio_analyzer_uses_payload_extractor(tmp_path: Path) -> None:
    body = _wav_bytes(frames=1600, sample_rate=8000)
    path = tmp_path / "clip.wav"
    path.write_bytes(body)
    result = _fetch_result(
        tmp_path=tmp_path,
        body=body,
        kind=MediaKind.AUDIO,
        name="clip.wav",
    )
    settings = SimpleNamespace(
        run_transcription=False,
        transcription_language=None,
        max_duration_seconds=0,
        analysis_timeout_seconds=30.0,
        max_transcription_bytes=20 * 1024 * 1024,
    )
    analyzer = AudioAnalyzer(
        settings=settings,  # type: ignore[arg-type]
        media_file_resolver=_Resolver(path),  # type: ignore[arg-type]
        payload_extractor=AudioPayloadExtractor(),
        diarization_service=_NoDiarization(),  # type: ignore[arg-type]
        transcription_executor=_NoTranscription(),  # type: ignore[arg-type]
        event_analyzer=_NoEvents(),  # type: ignore[arg-type]
        emotion_analyzer=_NoEmotion(),  # type: ignore[arg-type]
        logger=TEST_LOGGER,
    )
    analysis = asyncio.run(analyzer.analyze(result=result))
    assert analysis.metadata_status == "extracted"
    assert analysis.sample_rate == 8000
    assert analysis.channels == 1
    assert analysis.duration_seconds == pytest.approx(0.2, rel=1e-3)
    assert analysis.metadata["duration_seconds"] == analysis.duration_seconds
