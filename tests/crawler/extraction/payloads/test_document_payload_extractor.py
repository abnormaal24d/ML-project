"""Tests for objective DocumentPayloadExtractor (no OCR / scoring)."""

from __future__ import annotations

import hashlib
import sys

from crawler.extraction.payloads.document_payload_extractor import (
    DocumentPayloadExtractionResult,
    DocumentPayloadExtractor,
)


def _simple_pdf(*, pages: int = 2) -> bytes:
    """Minimal PDF with leaf /Type /Page objects for heuristic counting."""

    kids = " ".join(f"{i + 3} 0 R" for i in range(pages))
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        (
            f"2 0 obj\n<< /Type /Pages /Kids [{kids}] "
            f"/Count {pages} >>\nendobj\n"
        ),
    ]
    for index in range(pages):
        objects.append(
            f"{index + 3} 0 obj\n"
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>\n"
            "endobj\n"
        )
    body = (
        b"%PDF-1.4\n"
        + "".join(objects).encode("ascii")
        + b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )
    return body


def test_extract_pdf_page_count_and_format() -> None:
    body = _simple_pdf(pages=3)
    result = DocumentPayloadExtractor().extract(
        body=body,
        suffix=".pdf",
        mime_type="application/pdf",
    )
    assert isinstance(result, DocumentPayloadExtractionResult)
    assert result.is_pdf is True
    assert result.format == "pdf"
    assert result.page_count == 3
    assert result.file_suffix == ".pdf"
    assert result.byte_size == len(body)
    assert result.sha256 == hashlib.sha256(body).hexdigest()


def test_extract_plain_text_format() -> None:
    body = b"hello document payload\nline two\n"
    result = DocumentPayloadExtractor().extract(
        body=body,
        suffix=".txt",
        mime_type="text/plain",
    )
    assert result is not None
    assert result.is_pdf is False
    assert result.format == "text"
    assert result.page_count is None
    assert result.byte_size == len(body)


def test_extract_empty_body_returns_none() -> None:
    assert DocumentPayloadExtractor().extract(body=b"") is None


def test_extract_pdf_magic_without_suffix() -> None:
    body = _simple_pdf(pages=1)
    result = DocumentPayloadExtractor().extract(body=body)
    assert result is not None
    assert result.is_pdf is True
    assert result.format == "pdf"
    assert result.page_count == 1


def test_pdf_magic_must_be_complete_and_at_byte_zero() -> None:
    result = DocumentPayloadExtractor().extract(body=b"%PDF-1.7\n")

    assert result is not None
    assert result.is_pdf is True
    assert result.format == "pdf"


def test_malformed_or_offset_pdf_magic_is_not_a_pdf() -> None:
    for body in (
        b"%PDF",
        b"%PDFx-1.7\n",
        b"\xef\xbb\xbf%PDF-1.7\n",
        b" %PDF-1.7\n",
        b"not-a-pdf%PDF-1.7\n",
    ):
        result = DocumentPayloadExtractor().extract(body=body, suffix=".pdf")

        assert result is not None
        assert result.is_pdf is False
        assert result.page_count is None


def test_pdf_mime_type_does_not_override_non_pdf_body() -> None:
    result = DocumentPayloadExtractor().extract(
        body=b"plain text, despite the claimed MIME type",
        suffix=".pdf",
        mime_type="application/pdf",
    )

    assert result is not None
    assert result.is_pdf is False
    assert result.page_count is None


def test_pdf_page_count_falls_back_when_pypdf_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)
    body = _simple_pdf(pages=4)

    result = DocumentPayloadExtractor().extract(
        body=body,
        suffix=".pdf",
        mime_type="application/pdf",
    )

    assert result is not None
    assert result.page_count == 4


def test_result_has_no_ocr_or_preview_fields() -> None:
    body = _simple_pdf(pages=1)
    result = DocumentPayloadExtractor().extract(body=body, suffix=".pdf")
    assert result is not None
    field_names = set(result.__dataclass_fields__)
    forbidden = {
        "text_preview",
        "extracted_text",
        "ocr_result",
        "quality_score",
    }
    assert field_names.isdisjoint(forbidden)
