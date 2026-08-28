"""Objective document payload metadata from already-fetched bytes/files.

No OCR, quality scoring, or trainability judgment. Text preview extraction
and OCR remain enrichment concerns outside this layer.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crawler.classification.mime_type_resolver import (
    normalize_mime_type,
)

_PDF_MAGIC = b"%PDF-"
# Count leaf page objects; avoid matching /Type /Pages.
_PDF_PAGE_OBJECT_RE = re.compile(rb"/Type\s*/Page(?=[^s/]|$)")


@dataclass(frozen=True, slots=True)
class DocumentPayloadExtractionResult:
    """Deterministic properties of one document payload."""

    format: str | None
    page_count: int | None
    is_pdf: bool
    file_suffix: str | None
    byte_size: int
    sha256: str

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "page_count": self.page_count,
            "is_pdf": self.is_pdf,
            "suffix": self.file_suffix,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


class DocumentPayloadExtractor:
    """Extract objective document payload metadata from raw bytes or a path."""

    def extract(
        self,
        *,
        body: bytes,
        suffix: str | None = None,
        mime_type: str | None = None,
    ) -> DocumentPayloadExtractionResult | None:
        """Return payload metadata, or ``None`` for empty bodies."""

        if not body:
            return None

        byte_size = len(body)
        sha256 = hashlib.sha256(body).hexdigest()
        normalized_suffix = _normalize_suffix(suffix)
        is_pdf = _body_is_pdf(body=body)
        format_name = _resolve_format(
            is_pdf=is_pdf,
            suffix=normalized_suffix,
            mime_type=mime_type,
            body=body,
        )
        page_count = _count_pdf_pages(body=body) if is_pdf else None

        return DocumentPayloadExtractionResult(
            format=format_name,
            page_count=page_count,
            is_pdf=is_pdf,
            file_suffix=normalized_suffix,
            byte_size=byte_size,
            sha256=sha256,
        )

    def extract_from_path(
        self,
        *,
        path: Path,
        mime_type: str | None = None,
    ) -> DocumentPayloadExtractionResult | None:
        """Extract objective metadata from a local document path."""

        try:
            body = path.read_bytes()
        except OSError:
            return None
        return self.extract(
            body=body,
            suffix=path.suffix,
            mime_type=mime_type,
        )


def _normalize_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if not text.startswith("."):
        text = f".{text}"
    return text


def _body_is_pdf(*, body: bytes) -> bool:
    """Return whether the payload has the strict PDF file signature."""

    return body.startswith(_PDF_MAGIC)


def _resolve_format(
    *,
    is_pdf: bool,
    suffix: str | None,
    mime_type: str | None,
    body: bytes,
) -> str | None:
    if is_pdf:
        return "pdf"
    if mime_type:
        lowered = normalize_mime_type(mime_type)
        if lowered and lowered.startswith("text/"):
            return "text"
        if lowered in {
            "application/msword",
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document",
        }:
            return "word"
        if lowered:
            return lowered
    if suffix in {".txt", ".md", ".csv", ".tsv", ".log"}:
        return "text"
    if suffix in {".doc", ".docx"}:
        return "word"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix:
        return suffix.lstrip(".")
    # Sniff plain text when most bytes are printable.
    if body and _looks_like_text(body=body):
        return "text"
    return None


def _looks_like_text(*, body: bytes) -> bool:
    sample = body[:4096]
    if not sample:
        return False
    if b"\x00" in sample:
        return False
    printable = sum(
        1 for byte in sample if 32 <= byte <= 126 or byte in {9, 10, 13}
    )
    return (printable / len(sample)) >= 0.85


def _count_pdf_pages(*, body: bytes) -> int | None:
    counted = _count_pdf_pages_with_pypdf(body=body)
    if counted is not None:
        return counted
    return _count_pdf_pages_heuristic(body=body)


def _count_pdf_pages_with_pypdf(*, body: bytes) -> int | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as handle:
            handle.write(body)
            temp_path = Path(handle.name)
    except OSError:
        return None

    try:
        try:
            reader = PdfReader(str(temp_path))
            pages = reader.pages
            count = len(pages)
        except Exception:
            return None
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        return count
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _count_pdf_pages_heuristic(*, body: bytes) -> int | None:
    matches = _PDF_PAGE_OBJECT_RE.findall(body)
    if not matches:
        return None
    return len(matches)
