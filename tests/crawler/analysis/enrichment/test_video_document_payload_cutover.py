"""Analyzer cutover tests for video/document payload extractors."""

from __future__ import annotations

import asyncio
import hashlib
import struct
from pathlib import Path

from crawler.analysis.enrichment.documents.document_analyzer import (
    DocumentAnalyzer,
)
from crawler.classification.media_kind import MediaKind
from crawler.extraction.payloads.document_payload_extractor import (
    DocumentPayloadExtractor,
)
from crawler.extraction.payloads.video_payload_extractor import (
    VideoPayloadExtractor,
)
from crawler.fetching.results.payload import FetchedPayload
from crawler.fetching.results.result import FetchResult


def _atom(atom_type: bytes, payload: bytes) -> bytes:
    size = 8 + len(payload)
    return struct.pack(">I", size) + atom_type + payload


def _minimal_mp4(
    *,
    timescale: int = 1000,
    duration: int = 1000,
    width: int = 128,
    height: int = 72,
) -> bytes:
    mvhd_body = bytearray(100)
    struct.pack_into(">I", mvhd_body, 12, timescale)
    struct.pack_into(">I", mvhd_body, 16, duration)
    struct.pack_into(">I", mvhd_body, 20, 0x00010000)
    struct.pack_into(">H", mvhd_body, 24, 0x0100)
    tkhd_body = bytearray(84)
    struct.pack_into(">I", tkhd_body, 76, width << 16)
    struct.pack_into(">I", tkhd_body, 80, height << 16)
    moov = _atom(
        b"moov",
        _atom(b"mvhd", bytes(mvhd_body))
        + _atom(b"trak", _atom(b"tkhd", bytes(tkhd_body))),
    )
    ftyp = _atom(b"ftyp", b"isom" + b"\x00\x00\x00\x00" + b"isom")
    return ftyp + moov


def _simple_pdf() -> bytes:
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >>\n"
        b"endobj\n"
        b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )


def _fetch_result(
    *,
    tmp_path: Path,
    body: bytes,
    kind: MediaKind,
    name: str,
    mime_type: str,
) -> FetchResult:
    path = tmp_path / name
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    return FetchResult(
        url=f"https://example.test/{name}",
        final_url=f"https://example.test/{name}",
        status_code=200,
        headers={"content-type": mime_type},
        fetched_at="2024-01-01T00:00:00Z",
        content_type=mime_type,
        mime_type=mime_type,
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


class _Resolver:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.cleaned = False

    async def resolve_path(self, *, result, suffix: str = ".bin"):
        del result, suffix
        return self._path

    def cleanup_owned_path(self, path: Path) -> None:
        del path
        self.cleaned = True


class _DocumentTextReader:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, bool, str | None, str | None]] = []

    def read_text(
        self,
        *,
        path: Path,
        is_pdf: bool,
        format_name: str | None,
        encoding: str | None,
    ) -> str:
        self.calls.append((path, is_pdf, format_name, encoding))
        return "document native text"


class _NoOcr:
    def extract(self, *, path: Path):
        del path
        return None


def test_video_payload_extractor_reads_path_metadata(
    tmp_path: Path,
) -> None:
    body = _minimal_mp4(duration=2000, timescale=1000, width=320, height=180)
    path = tmp_path / "clip.mp4"
    path.write_bytes(body)
    extraction = VideoPayloadExtractor().extract_from_path(path=path)
    assert extraction is not None
    metadata = extraction.as_metadata_dict()
    assert metadata["duration_seconds"] == 2.0
    assert metadata["width"] == 320
    assert metadata["height"] == 180
    assert metadata["format"] == "MP4"


def test_document_analyzer_uses_payload_extractor(tmp_path: Path) -> None:
    body = _simple_pdf()
    path = tmp_path / "doc.pdf"
    path.write_bytes(body)
    result = _fetch_result(
        tmp_path=tmp_path,
        body=body,
        kind=MediaKind.DOCUMENT,
        name="doc.pdf",
        mime_type="application/pdf",
    )
    resolver = _Resolver(path)
    text_reader = _DocumentTextReader()
    analyzer = DocumentAnalyzer(
        extract_text=True,
        run_ocr=False,
        max_ocr_bytes=0,
        media_file_resolver=resolver,  # type: ignore[arg-type]
        payload_extractor=DocumentPayloadExtractor(),
        document_text_reader=text_reader,  # type: ignore[arg-type]
        text_extraction_service=_NoOcr(),  # type: ignore[arg-type]
    )
    analysis = asyncio.run(analyzer.analyze(result=result))
    assert analysis.page_count == 1
    assert analysis.document_text.text == "document native text"
    assert analysis.document_text.source == "native"
    assert text_reader.calls == [(path, True, "pdf", None)]
    assert analysis.metadata["format"] == "pdf"
    assert analysis.metadata["is_pdf"] is True
    assert analysis.metadata["document_text_source"] == "native"
    assert analysis.metadata["sha256"] == hashlib.sha256(body).hexdigest()
    assert resolver.cleaned is True
