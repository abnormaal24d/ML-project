"""Tests for AudioReferenceExtractor draft-only audio discovery."""

from __future__ import annotations

from types import SimpleNamespace

from config.collection.discovery import HtmlParserSettings
from crawler.extraction.assets.candidate.asset_candidate_collector import (
    AssetCandidateCollector,
)
from crawler.extraction.assets.candidate.asset_candidate_deduper import (
    AssetCandidateDeduper,
)
from crawler.extraction.assets.candidate.asset_candidate_resolution import (
    AssetCandidateResolution,
)
from crawler.extraction.assets.embedded_media.video_embed_detector import (
    VideoEmbedDetector,
)
from crawler.extraction.html.html_parser import HtmlParser
from crawler.extraction.modalities.audio_extractor import (
    AudioReferenceExtractor,
)
from crawler.extraction.modalities.image_extractor import (
    PageExtractionContext,
)
from crawler.extraction.modalities.page_element_index import (
    PageElementIndexBuilder,
)
from tests.support.url_components import make_url_candidate_resolution

_PAGE_URL = "https://example.test/page"


def _logger() -> SimpleNamespace:
    return SimpleNamespace(
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )


def _html_parser() -> HtmlParser:
    return HtmlParser(
        settings=HtmlParserSettings(
            parser="html.parser",
            parser_candidates=("html.parser",),
            prefer_beautiful_soup=True,
        ),
        logger=_logger(),
    )


def _collector() -> AssetCandidateCollector:
    return AssetCandidateCollector(
        candidate_resolution=AssetCandidateResolution(
            url_candidate_resolution=make_url_candidate_resolution(),
            deduper=AssetCandidateDeduper(),
        ),
        embed_detector=VideoEmbedDetector(),
    )


def _extract_audio_candidates(html: bytes) -> tuple[str, ...]:
    document = _html_parser().parse(body=html, encoding="utf-8")
    index = PageElementIndexBuilder().build(document=document)
    drafts = AudioReferenceExtractor().extract_references(
        index=index,
        context=PageExtractionContext(page_url=_PAGE_URL),
    )
    return tuple(d.candidate for d in drafts)


def test_audio_tag_and_nested_source() -> None:
    html = b"""
    <html><body>
      <audio src="/a.mp3" data-src="/lazy.m4a">
        <source src="/nested.ogg" type="audio/ogg"/>
      </audio>
    </body></html>
    """
    candidates = _extract_audio_candidates(html)
    assert "/a.mp3" in candidates
    assert "/lazy.m4a" in candidates
    assert "/nested.ogg" in candidates
    assert candidates.count("/nested.ogg") == 1


def test_meta_link_jsonld_and_inline_audio() -> None:
    html = b"""
    <html>
      <head>
        <meta property="og:audio" content="/og.mp3"/>
        <link rel="preload" as="audio" href="/preload.mp3"/>
        <script type="application/ld+json">
          {"@type":"AudioObject","contentUrl":"/ld.mp3"}
        </script>
        <script>
          var cfg = {"url": "https://cdn.example.test/inline.mp3"};
        </script>
      </head>
      <body></body>
    </html>
    """
    candidates = _extract_audio_candidates(html)
    assert "/og.mp3" in candidates
    assert "/preload.mp3" in candidates
    assert "/ld.mp3" in candidates
    assert "https://cdn.example.test/inline.mp3" in candidates


def test_collect_drafts_dedupes_audio() -> None:
    html = b"""
    <html>
      <head><meta property="og:audio" content="/same.mp3"/></head>
      <body><audio src="/same.mp3"></audio></body>
    </html>
    """
    document = _html_parser().parse(body=html, encoding="utf-8")
    index = PageElementIndexBuilder().build(document=document)
    drafts = AudioReferenceExtractor().extract_references(
        index=index,
        context=PageExtractionContext(page_url=_PAGE_URL),
    )
    discovery = _collector().collect_drafts(
        drafts=drafts,
        page_url=_PAGE_URL,
    )
    urls = [item.url for item in discovery.audio]
    assert urls.count("https://example.test/same.mp3") == 1


def test_audio_candidate_order_is_container_then_owned_sources() -> None:
    candidates = _extract_audio_candidates(
        b"""
        <html><body>
          <audio src="/first.mp3"><source src="/first.ogg"/></audio>
          <audio src="/second.mp3"><source src="/second.ogg"/></audio>
          <source src="/standalone.ogg" type="audio/ogg"/>
        </body></html>
        """
    )

    assert candidates == (
        "/first.mp3",
        "/first.ogg",
        "/second.mp3",
        "/second.ogg",
        "/standalone.ogg",
    )
