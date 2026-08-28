"""Unit tests for PageTextContentExtractor (no re-parse)."""

from __future__ import annotations

from types import SimpleNamespace

from config.collection.discovery import HtmlParserSettings
from config.collection.processors import PageTextExtractionSettings
from crawler.extraction.html.html_parser import HtmlParser
from crawler.extraction.modalities.page_text_content_extractor import (
    PageTextContentExtractor,
)


def _logger() -> SimpleNamespace:
    return SimpleNamespace(
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )


def _parse(html: bytes):
    parser = HtmlParser(
        settings=HtmlParserSettings(
            parser="html.parser",
            parser_candidates=("html.parser",),
            prefer_beautiful_soup=True,
        ),
        logger=_logger(),
    )
    return parser.parse(body=html, encoding="utf-8")


def test_extracts_main_text_headings_and_markdown() -> None:
    document = _parse(
        b"""
        <html><body>
          <nav>Skip nav</nav>
          <main>
            <h1>Title One</h1>
            <p>Hello world content.</p>
            <h2>Section</h2>
            <pre>print(1)</pre>
          </main>
          <footer>cookie privacy</footer>
        </body></html>
        """
    )
    content = PageTextContentExtractor().extract(document=document)
    assert "Title One" in content.text
    assert "Hello world content." in content.text
    assert "Title One" in content.headings
    assert "Section" in content.headings
    assert content.code_block_count == 1
    assert "# Title One" in content.markdown
    assert content.char_count == len(content.text)
    assert content.text_preview
    assert "Skip nav" not in content.text


def test_does_not_accept_index_kwarg() -> None:
    document = _parse(b"<html><body><p>x</p></body></html>")
    extractor = PageTextContentExtractor()
    content = extractor.extract(document=document)
    assert "x" in content.text


def test_respects_max_text_chars() -> None:
    body = (
        b"<html><body><main><p>"
        + (b"word " * 100)
        + b"</p></main></body></html>"
    )
    document = _parse(body)
    content = PageTextContentExtractor(
        settings=PageTextExtractionSettings(max_text_chars=40)
    ).extract(document=document)
    assert content.char_count <= 40
    assert "text_truncated" in content.extraction_warnings
