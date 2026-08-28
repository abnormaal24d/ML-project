"""Unit tests for PageMetadataExtractor (index-only, no find_all)."""

from __future__ import annotations

from types import SimpleNamespace

from config.collection.discovery import HtmlParserSettings
from crawler.extraction.html.html_parser import HtmlParser
from crawler.extraction.modalities.page_element_index import (
    PageElementIndexBuilder,
)
from crawler.extraction.modalities.page_metadata_extractor import (
    PageMetadataExtractor,
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
    document = parser.parse(body=html, encoding="utf-8")
    return PageElementIndexBuilder().build(document=document)


def test_extracts_title_from_title_element() -> None:
    index = _parse(b"<html><head><title> Hello </title></head></html>")
    meta = PageMetadataExtractor().extract(index=index)
    assert meta.title == "Hello"


def test_title_falls_back_to_og_and_twitter() -> None:
    index = _parse(
        b"""
        <html><head>
          <meta property="og:title" content="OG Title"/>
        </head></html>
        """
    )
    assert PageMetadataExtractor().extract(index=index).title == "OG Title"

    index = _parse(
        b"""
        <html><head>
          <meta name="twitter:title" content="Tw Title"/>
        </head></html>
        """
    )
    assert PageMetadataExtractor().extract(index=index).title == "Tw Title"


def test_title_element_wins_over_og() -> None:
    index = _parse(
        b"""
        <html><head>
          <title>Document Title</title>
          <meta property="og:title" content="OG"/>
        </head></html>
        """
    )
    assert (
        PageMetadataExtractor().extract(index=index).title == "Document Title"
    )


def test_extracts_canonical_from_link() -> None:
    index = _parse(
        b"""
        <html><head>
          <link rel="canonical" href="https://example.test/can"/>
          <link rel="stylesheet" href="/style.css"/>
        </head></html>
        """
    )
    meta = PageMetadataExtractor().extract(index=index)
    assert meta.canonical_url == "https://example.test/can"


def test_extracts_robots_directives_deduped_in_order() -> None:
    index = _parse(
        b"""
        <html><head>
          <meta name="robots" content="noindex, nofollow"/>
          <meta name="googlebot" content="nofollow, noarchive"/>
        </head></html>
        """
    )
    meta = PageMetadataExtractor().extract(index=index)
    assert meta.meta_robots == ("noindex", "nofollow", "noarchive")


def test_extracts_meta_refresh_url() -> None:
    index = _parse(
        b"""
        <html><head>
          <meta http-equiv="refresh" content="0;url=/next"/>
        </head></html>
        """
    )
    meta = PageMetadataExtractor().extract(index=index)
    assert meta.meta_refresh_url == "/next"


def test_empty_document_yields_empty_metadata() -> None:
    index = _parse(b"<html><body></body></html>")
    meta = PageMetadataExtractor().extract(index=index)
    assert meta.title is None
    assert meta.canonical_url is None
    assert meta.meta_robots == ()
    assert meta.meta_refresh_url is None
