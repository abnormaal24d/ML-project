"""LinkExtractor index-based extraction tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from config.collection.discovery import HtmlParserSettings
from config.collection.extraction import LinkExtractorSettings
from crawler.extraction.html.html_parser import HtmlParser
from crawler.extraction.links.extractor import LinkExtractor
from crawler.extraction.modalities.page_element_index import (
    PageElementIndexBuilder,
)
from tests.support.url_components import make_url_candidate_resolution


def _logger() -> SimpleNamespace:
    return SimpleNamespace(
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )


def _extractor(
    *,
    settings: LinkExtractorSettings | None = None,
) -> LinkExtractor:
    resolution = make_url_candidate_resolution()
    return LinkExtractor(
        settings=settings or LinkExtractorSettings(),
        candidate_resolution=resolution,
        logger=_logger(),
    )


def _index(html: bytes):
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


def test_extract_candidates_from_index_only() -> None:
    index = _index(
        b"""
        <html>
          <head>
            <link rel="canonical" href="/can"/>
            <link rel="stylesheet" href="/style.css"/>
            <link rel="alternate" type="application/rss+xml" href="/feed.xml"/>
          </head>
          <body>
            <a href="/a">A</a>
            <a href="/b" rel="nofollow">B</a>
            <area href="/c"/>
          </body>
        </html>
        """
    )
    links = _extractor().extract_candidates(
        index=index,
        base_url="https://example.test/page",
    )
    urls = {item.url for item in links}
    assert "https://example.test/a" in urls
    assert "https://example.test/c" in urls
    assert "https://example.test/can" in urls
    assert "https://example.test/feed.xml" in urls
    # Default settings exclude nofollow and stylesheets.
    assert "https://example.test/b" not in urls
    assert "https://example.test/style.css" not in urls


def test_extract_candidates_does_not_call_find_all() -> None:
    index = _index(b"<html><body><a href='/x'>x</a></body></html>")
    # Replace buckets with plain tuples so any find_all would fail.
    spy_index = MagicMock()
    spy_index.link_elements = index.link_elements
    spy_index.resource_link_elements = index.resource_link_elements
    # Ensure extractor never walks a document attribute.
    del spy_index.find_all

    links = _extractor().extract_candidates(
        index=spy_index,
        base_url="https://example.test/",
    )
    assert any(item.url.endswith("/x") for item in links)


def test_iter_extract_candidates_matches_index_path() -> None:
    html = b"""
    <html><body>
      <a href="/one">One</a>
      <a href="/two">Two</a>
    </body></html>
    """
    parser = HtmlParser(
        settings=HtmlParserSettings(
            parser="html.parser",
            parser_candidates=("html.parser",),
            prefer_beautiful_soup=True,
        ),
        logger=_logger(),
    )
    document = parser.parse(body=html, encoding="utf-8")
    index = PageElementIndexBuilder().build(document=document)
    extractor = _extractor()
    from_index = {
        item.url
        for item in extractor.extract_candidates(
            index=index, base_url="https://example.test/"
        )
    }
    from_document = {
        item.url
        for item in extractor.iter_extract_candidates(
            base_url="https://example.test/",
            document=document,
        )
    }
    assert from_index == from_document
