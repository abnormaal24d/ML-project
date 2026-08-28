"""Optional PDF text reader that reports extraction failures explicitly."""

from __future__ import annotations

from io import BytesIO

from preprocessing.privacy.inspection.content_readers.document_content import (
    DocumentPage,
)


def read_pdf_pages(payload: bytes) -> tuple[DocumentPage, ...]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF inspection") from exc
    reader = PdfReader(BytesIO(payload), strict=True)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        pages.append(DocumentPage(number, page.extract_text() or ""))
    return tuple(pages)
