"""Native text readers for document payloads already classified by crawling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_SUPPORTED_TEXT_FORMATS = frozenset(
    {
        "text",
        "html",
        "application/json",
        "application/xml",
        "application/xhtml+xml",
    }
)


class PdfTextReader:
    """Read bounded native text from a PDF selected by the payload extractor."""

    def __init__(self, *, max_pages: int) -> None:
        self._max_pages = max(1, int(max_pages))

    def read_text(self, *, path: Path) -> str | None:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = reader.pages
        except Exception:
            # pypdf exposes parser-specific errors that vary by version.
            return None

        parts: list[str] = []
        for page_index in _sample_page_indexes(
            pages=pages,
            max_pages=self._max_pages,
        ):
            try:
                page = pages[page_index]
                text = page.extract_text() or ""
                stripped = text.strip()
            except Exception:
                # A malformed page must not fail the complete document path.
                continue
            if stripped:
                parts.append(stripped)

        joined = "\n".join(parts).strip()
        return joined or None


class DocumentTextReader:
    """Read native document text using the payload extractor's classification."""

    def __init__(
        self,
        *,
        pdf_text_reader: PdfTextReader,
        max_characters: int,
    ) -> None:
        self._pdf_text_reader = pdf_text_reader
        self._max_characters = max(0, int(max_characters))

    def read_text(
        self,
        *,
        path: Path,
        is_pdf: bool,
        format_name: str | None,
        encoding: str | None,
    ) -> str | None:
        if is_pdf:
            text = self._pdf_text_reader.read_text(path=path)
        elif _is_supported_text_format(format_name=format_name) and encoding:
            text = _read_text(path=path, encoding=encoding)
        else:
            return None

        if not text:
            return None
        return text[: self._max_characters].strip() or None


def _is_supported_text_format(*, format_name: str | None) -> bool:
    normalized = str(format_name or "").strip().lower()
    return normalized in _SUPPORTED_TEXT_FORMATS


def _read_text(*, path: Path, encoding: str) -> str | None:
    try:
        with path.open("r", encoding=encoding, errors="strict") as handle:
            return handle.read().strip() or None
    except (LookupError, OSError, UnicodeError):
        return None


def _sample_page_indexes(*, pages: Any, max_pages: int) -> tuple[int, ...]:
    limit = max(0, int(max_pages))
    if limit == 0:
        return ()
    try:
        page_count = len(pages)
    except (TypeError, ValueError):
        return tuple(range(limit))
    if page_count <= 0:
        return ()
    if page_count <= limit:
        return tuple(range(page_count))
    if limit == 1:
        return (0,)
    if limit == 2:
        return (0, page_count - 1)

    indexes = [0, page_count // 2, page_count - 1]
    for index in range(1, page_count):
        if len(indexes) >= limit:
            break
        if index not in indexes:
            indexes.append(index)
    return tuple(sorted(indexes))
