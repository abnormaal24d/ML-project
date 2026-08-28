"""Tests for VideoReferenceExtractor draft-only video discovery."""

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
from crawler.extraction.modalities.image_extractor import (
    PageExtractionContext,
)
from crawler.extraction.modalities.page_element_index import (
    PageElementIndexBuilder,
)
from crawler.extraction.modalities.video_extractor import (
    VideoReferenceExtractor,
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


def _extract_video_candidates(html: bytes) -> tuple[str, ...]:
    document = _html_parser().parse(body=html, encoding="utf-8")
    index = PageElementIndexBuilder().build(document=document)
    drafts = VideoReferenceExtractor(
        embed_detector=VideoEmbedDetector(),
    ).extract_references(
        index=index,
        context=PageExtractionContext(page_url=_PAGE_URL),
    )
    return tuple(d.candidate for d in drafts)


def test_video_tag_nested_source_and_poster_metadata() -> None:
    html = b"""
    <html><body>
      <video src="/v.mp4" poster="/poster.jpg" data-src="/lazy.webm">
        <source src="/nested2.mp4" type="video/mp4"/>
      </video>
    </body></html>
    """
    document = _html_parser().parse(body=html, encoding="utf-8")
    index = PageElementIndexBuilder().build(document=document)
    drafts = VideoReferenceExtractor(
        embed_detector=VideoEmbedDetector(),
    ).extract_references(
        index=index,
        context=PageExtractionContext(page_url=_PAGE_URL),
    )
    candidates = tuple(d.candidate for d in drafts)
    assert "/v.mp4" in candidates
    assert "/lazy.webm" in candidates
    assert "/nested2.mp4" in candidates
    assert "/poster.jpg" not in candidates
    video_src = [d for d in drafts if d.candidate == "/v.mp4"]
    assert video_src[0].metadata.get("poster_url") == "/poster.jpg"


def test_meta_link_jsonld_iframe_inline_video() -> None:
    html = b"""
    <html>
      <head>
        <meta property="og:video" content="/og.mp4"/>
        <link rel="preload" as="video" href="/preload.mp4"/>
        <script type="application/ld+json">
          {"@type":"VideoObject","contentUrl":"/ld.mp4"}
        </script>
        <script>
          var cfg = {"src": "https://cdn.example.test/inline.mp4"};
        </script>
      </head>
      <body>
        <iframe src="https://www.youtube.com/embed/abc123"></iframe>
      </body>
    </html>
    """
    candidates = _extract_video_candidates(html)
    assert "/og.mp4" in candidates
    assert "/preload.mp4" in candidates
    assert "/ld.mp4" in candidates
    assert "https://cdn.example.test/inline.mp4" in candidates
    assert any("youtube.com/embed" in c for c in candidates)


def test_collect_drafts_dedupes_video() -> None:
    html = b"""
    <html>
      <head><meta property="og:video" content="/same.mp4"/></head>
      <body><video src="/same.mp4"></video></body>
    </html>
    """
    document = _html_parser().parse(body=html, encoding="utf-8")
    index = PageElementIndexBuilder().build(document=document)
    drafts = VideoReferenceExtractor(
        embed_detector=VideoEmbedDetector(),
    ).extract_references(
        index=index,
        context=PageExtractionContext(page_url=_PAGE_URL),
    )
    discovery = _collector().collect_drafts(drafts=drafts, page_url=_PAGE_URL)
    urls = [item.url for item in discovery.video]
    assert urls.count("https://example.test/same.mp4") == 1


def test_video_candidate_order_is_container_then_owned_sources() -> None:
    candidates = _extract_video_candidates(
        b"""
        <html><body>
          <video src="/first.mp4"><source src="/first.webm"/></video>
          <video src="/second.mp4"><source src="/second.webm"/></video>
          <source src="/standalone.mp4" type="video/mp4"/>
        </body></html>
        """
    )

    assert candidates == (
        "/first.mp4",
        "/first.webm",
        "/second.mp4",
        "/second.webm",
        "/standalone.mp4",
    )
