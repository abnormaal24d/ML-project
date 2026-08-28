"""Tests for ImageReferenceExtractor draft-only image discovery."""

from __future__ import annotations

from types import SimpleNamespace

from config.collection.discovery import HtmlParserSettings
from crawler.extraction.html.html_parser import HtmlParser
from crawler.extraction.modalities.image_extractor import (
    ImageReferenceExtractor,
    PageExtractionContext,
)
from crawler.extraction.modalities.page_element_index import (
    PageElementIndexBuilder,
)


def _parser() -> HtmlParser:
    return HtmlParser(
        settings=HtmlParserSettings(
            parser="html.parser",
            parser_candidates=("html.parser",),
            prefer_beautiful_soup=True,
        ),
        logger=SimpleNamespace(
            debug=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
    )


def _extract_image_candidates(html: bytes) -> tuple[str, ...]:
    document = _parser().parse(body=html, encoding="utf-8")
    index = PageElementIndexBuilder().build(document=document)
    drafts = ImageReferenceExtractor().extract_references(
        index=index,
        context=PageExtractionContext(page_url="https://example.test/page"),
    )
    assert all(draft.kind == "image" for draft in drafts)
    return tuple(draft.candidate for draft in drafts)


def test_image_reference_extractor_covers_primary_sources() -> None:
    html = b"""
    <html>
      <head>
        <meta property="og:image" content="https://cdn.example.test/og.jpg"/>
        <meta name="twitter:image" content="https://cdn.example.test/tw.jpg"/>
        <link rel="preload" as="image" href="https://cdn.example.test/preload.jpg"/>
        <script type="application/ld+json">
          {"@type":"ImageObject","contentUrl":"https://cdn.example.test/ld.jpg"}
        </script>
      </head>
      <body>
        <img src="/a.jpg" data-src="/lazy.jpg" srcset="/s1.jpg 1x, /s2.jpg 2x"/>
        <picture>
          <source srcset="/p.jpg"/>
          <img src="/fallback.jpg"/>
        </picture>
        <video poster="/poster.jpg" src="/v.mp4"></video>
        <div style="background-image:url(/bg.png)">x</div>
      </body>
    </html>
    """
    candidates = _extract_image_candidates(html)

    assert "https://cdn.example.test/og.jpg" in candidates
    assert "https://cdn.example.test/tw.jpg" in candidates
    assert "https://cdn.example.test/preload.jpg" in candidates
    assert "https://cdn.example.test/ld.jpg" in candidates
    assert "/a.jpg" in candidates
    assert "/lazy.jpg" in candidates
    assert "/s1.jpg" in candidates
    assert "/s2.jpg" in candidates
    assert "/p.jpg" in candidates
    assert "/fallback.jpg" in candidates
    assert "/poster.jpg" in candidates
    assert "/bg.png" in candidates


def test_image_reference_extractor_does_not_emit_non_image_meta() -> None:
    html = b"""
    <html><head>
      <meta property="og:video" content="https://cdn.example.test/v.mp4"/>
      <meta name="description" content="not an image"/>
    </head></html>
    """
    candidates = _extract_image_candidates(html)
    assert "https://cdn.example.test/v.mp4" not in candidates
    assert "not an image" not in candidates


def test_image_reference_extractor_is_collector_free() -> None:
    """Draft extraction must not require collector/budget state."""

    extractor = ImageReferenceExtractor()
    assert not hasattr(extractor, "_collector")
    assert not hasattr(extractor, "collector")


def test_image_candidate_order_preserves_structural_phases() -> None:
    candidates = _extract_image_candidates(
        b"""
        <html><body>
          <img src="/ordinary.jpg"/>
          <picture>
            <source src="/picture-source.jpg" type="image/jpeg"/>
            <img src="/picture-image.jpg"/>
          </picture>
          <source src="/standalone.png" type="image/png"/>
          <video poster="/poster.jpg"></video>
        </body></html>
        """
    )

    assert candidates == (
        "/ordinary.jpg",
        "/picture-source.jpg",
        "/picture-image.jpg",
        "/standalone.png",
        "/poster.jpg",
    )
