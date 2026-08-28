"""Image discovery via ImageReferenceExtractor + collector."""

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
    ImageReferenceExtractor,
    PageExtractionContext,
)
from crawler.extraction.modalities.page_element_index import (
    PageElementIndexBuilder,
)
from tests.support.url_components import make_url_candidate_resolution

_PAGE_URL = "https://example.test/page"

_EXPECTED_IMAGE_URLS = (
    "https://cdn.example.test/ld.jpg",
    "https://cdn.example.test/og.jpg",
    "https://cdn.example.test/preload.jpg",
    "https://cdn.example.test/tw.jpg",
    "https://example.test/a.jpg",
    "https://example.test/bg.png",
    "https://example.test/fallback.jpg",
    "https://example.test/lazy.jpg",
    "https://example.test/p.jpg",
    "https://example.test/poster.jpg",
    "https://example.test/s1.jpg",
    "https://example.test/s2.jpg",
)


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


def _fixture_html() -> bytes:
    return b"""
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
        <img src="/a.jpg" data-src="/lazy.jpg"
             srcset="/s1.jpg 1x, /s2.jpg 2x"/>
        <picture>
          <source srcset="/p.jpg"/>
          <img src="/fallback.jpg"/>
        </picture>
        <video poster="/poster.jpg" src="/v.mp4"></video>
        <div style="background-image:url(/bg.png)">x</div>
      </body>
    </html>
    """


def _discover_images(*, html: bytes, max_results: int | None = None):
    document = _html_parser().parse(body=html, encoding="utf-8")
    index = PageElementIndexBuilder().build(document=document)
    drafts = ImageReferenceExtractor(
        include_icon_link_assets=True,
    ).extract_references(
        index=index,
        context=PageExtractionContext(page_url=_PAGE_URL),
    )
    return _collector().collect_drafts(
        drafts=drafts,
        page_url=_PAGE_URL,
        max_results=max_results,
    )


def test_image_reference_resolves_and_dedupes() -> None:
    discovery = _discover_images(html=_fixture_html())
    urls = tuple(sorted({item.url for item in discovery.images}))
    assert urls == _EXPECTED_IMAGE_URLS


def test_img_and_og_image_same_path_dedupe() -> None:
    html = b"""
    <html>
      <head><meta property="og:image" content="/image.jpg"/></head>
      <body><img src="/image.jpg"/></body>
    </html>
    """
    discovery = _discover_images(html=html)
    urls = [item.url for item in discovery.images]
    assert urls == ["https://example.test/image.jpg"]


def test_collect_drafts_respects_max_results() -> None:
    discovery = _discover_images(html=_fixture_html(), max_results=2)
    assert len(discovery.images) <= 2
