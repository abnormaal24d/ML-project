"""Regression tests for native document text extraction."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from preprocessing.media.adapters.document_text_reader import (
    DocumentTextReader,
    PdfTextReader,
)


class _StubPdfTextReader(PdfTextReader):
    def __init__(self, text: str | None) -> None:
        super().__init__(max_pages=1)
        self.text = text
        self.paths: list[Path] = []

    def read_text(self, *, path: Path) -> str | None:
        self.paths.append(path)
        return self.text


def test_text_reader_uses_supplied_cp1252_encoding(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_bytes("café".encode("cp1252"))
    reader = DocumentTextReader(
        pdf_text_reader=_StubPdfTextReader(None),
        max_characters=100,
    )

    text = reader.read_text(
        path=path,
        is_pdf=False,
        format_name="text",
        encoding="cp1252",
    )

    assert text == "café"


def test_text_reader_rejects_unknown_encoding(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text("plain text", encoding="utf-8")
    reader = DocumentTextReader(
        pdf_text_reader=_StubPdfTextReader(None),
        max_characters=100,
    )

    assert (
        reader.read_text(
            path=path,
            is_pdf=False,
            format_name="text",
            encoding="not-a-real-encoding",
        )
        is None
    )


@pytest.mark.parametrize(
    ("format_name", "suffix"),
    (
        ("zip", ".zip"),
        ("word", ".docx"),
        ("application/epub+zip", ".epub"),
    ),
)
def test_text_reader_rejects_binary_document_formats(
    tmp_path: Path,
    format_name: str,
    suffix: str,
) -> None:
    path = tmp_path / f"document{suffix}"
    path.write_bytes(b"PK\x03\x04binary archive")
    reader = DocumentTextReader(
        pdf_text_reader=_StubPdfTextReader(None),
        max_characters=100,
    )

    assert (
        reader.read_text(
            path=path,
            is_pdf=False,
            format_name=format_name,
            encoding="cp1252",
        )
        is None
    )


def test_text_reader_uses_payload_pdf_decision_and_caps_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "document.payload"
    path.write_bytes(b"not used by the stub")
    pdf_reader = _StubPdfTextReader("native text")
    reader = DocumentTextReader(
        pdf_text_reader=pdf_reader,
        max_characters=6,
    )

    text = reader.read_text(
        path=path,
        is_pdf=True,
        format_name="zip",
        encoding=None,
    )

    assert text == "native"
    assert pdf_reader.paths == [path]


def test_pdf_text_reader_samples_configured_number_of_pages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    pages = [_Page("first"), _Page("second"), _Page("third"), _Page("last")]
    fake_pypdf = SimpleNamespace(
        PdfReader=lambda _path: SimpleNamespace(pages=pages),
    )
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-1.7")

    assert PdfTextReader(max_pages=2).read_text(path=path) == "first\nlast"
