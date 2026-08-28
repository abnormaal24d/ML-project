from __future__ import annotations

from config.settings.classification import ContentTypeDetectorSettings
from crawler.classification.mime_type_resolver import (
    MimeTypeResolver,
    _mime_type_from_filename_or_url,
)


class _Logger:
    def debug(self, *_args: object, **_kwargs: object) -> None:
        return None


class _SignatureDetector:
    def detect(self, *, sample: bytes) -> str | None:
        return "image/png" if sample.startswith(b"PNG") else None


def _resolver(*, detect_from_extension: bool = True) -> MimeTypeResolver:
    settings = ContentTypeDetectorSettings(
        detect_from_extension=detect_from_extension,
    )
    return MimeTypeResolver(
        settings=settings,
        logger=_Logger(),  # type: ignore[arg-type]
        mime_signature_detector=_SignatureDetector(),  # type: ignore[arg-type]
    )


def test_mime_type_from_url_uses_only_path_and_decodes_it() -> None:
    assert (
        _mime_type_from_filename_or_url(
            url="https://example.test/files/report%20final.PDF?download=1"
        )
        == "application/pdf"
    )


def test_extension_is_used_after_a_generic_header() -> None:
    result = _resolver().detect_resolution(
        url="https://example.test/media/song.MP3?raw=1",
        content_type_header="application/octet-stream",
        sample=b"",
    )

    assert result.mime_type == "audio/mpeg"
    assert result.source == "extension"


def test_signature_wins_over_extension_evidence() -> None:
    result = _resolver().detect_resolution(
        url="https://example.test/download/not-really.pdf",
        content_type_header=None,
        sample=b"PNG-data",
    )

    assert result.mime_type == "image/png"
    assert result.source == "signature"


def test_signature_wins_over_a_specific_conflicting_header() -> None:
    result = _resolver().detect_resolution(
        url="https://example.test/download/file.txt",
        content_type_header="text/plain",
        sample=b"PNG-data",
    )

    assert result.mime_type == "image/png"
    assert result.source == "signature"
    assert result.exact_conflict is True
    assert result.major_conflict is True


def test_extension_detection_can_be_disabled() -> None:
    result = _resolver(detect_from_extension=False).detect_resolution(
        url="https://example.test/files/report.pdf",
        content_type_header=None,
        sample=b"",
    )

    assert result.mime_type == "application/octet-stream"
    assert result.source == "fallback"
