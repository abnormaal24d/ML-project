"""Characterization: PageMetadataExtractor structural fields on fixtures."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from config.collection.discovery import HtmlParserSettings
from config.collection.extraction import (
    LinkExtractorSettings,
    UrlNormalizerSettings,
)
from config.collection.processors import PageTextExtractionSettings
from crawler.classification.media_kind import MediaKind
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
from crawler.extraction.candidates.url_candidate_resolution import (
    UrlCandidateResolution,
)
from crawler.extraction.html.html_parser import HtmlParser
from crawler.extraction.links.extractor import LinkExtractor
from crawler.extraction.modalities.image_extractor import (
    ImageReferenceExtractor,
)
from crawler.extraction.modalities.page_content_extractor import (
    PageContentExtractor,
    PageExtractionResult,
)
from crawler.extraction.modalities.page_element_index import (
    PageElementIndexBuilder,
)
from crawler.extraction.modalities.page_metadata_extractor import (
    PageMetadataExtractor,
)
from crawler.extraction.modalities.page_text_content_extractor import (
    PageTextContentExtractor,
)
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.fetching.results.payload import FetchedPayload
from crawler.fetching.results.result import FetchResult
from crawler.governance.domains.host_normalizer import HostNormalizer
from tests.support.logging import TEST_LOGGER

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


def project_page_metadata(result: PageExtractionResult) -> dict[str, Any]:
    return {
        "title": result.metadata.title,
        "canonical_url": result.metadata.canonical_url,
        "meta_robots": result.metadata.meta_robots,
        "meta_refresh_url": result.metadata.meta_refresh_url,
    }


def _make_fetch_result(*, tmp_path: Path, html: bytes) -> FetchResult:
    path = tmp_path / "body.html"
    path.write_bytes(html)
    payload = FetchedPayload(
        temp_path=path,
        byte_size=len(html),
        sha256_hex="0" * 64,
        sniff_bytes=html[:64],
        chunk_count=1,
    )
    return FetchResult(
        url=_PAGE_URL,
        final_url=_PAGE_URL,
        status_code=200,
        headers={"content-type": "text/html"},
        fetched_at="2024-01-01T00:00:00Z",
        content_type="text/html",
        mime_type="text/html",
        encoding="utf-8",
        language=None,
        kind=MediaKind.PAGE,
        payload=payload,
        body_sha256="0" * 64,
    )


def _page_content_extractor() -> PageContentExtractor:
    normalizer = UrlNormalizer(
        settings=UrlNormalizerSettings(),
        logger=_logger(),
        host_normalizer=HostNormalizer(),
    )
    resolution = UrlCandidateResolution(
        url_normalizer=normalizer, logger=TEST_LOGGER
    )
    collector = AssetCandidateCollector(
        candidate_resolution=AssetCandidateResolution(
            url_candidate_resolution=resolution,
            deduper=AssetCandidateDeduper(),
        ),
        embed_detector=VideoEmbedDetector(),
    )
    return PageContentExtractor(
        html_parser=_html_parser(),
        element_index_builder=PageElementIndexBuilder(),
        metadata_extractor=PageMetadataExtractor(),
        text_content_extractor=PageTextContentExtractor(
            settings=PageTextExtractionSettings()
        ),
        link_extractor=LinkExtractor(
            settings=LinkExtractorSettings(),
            candidate_resolution=resolution,
            logger=_logger(),
        ),
        reference_extractors=(
            ImageReferenceExtractor(include_icon_link_assets=True),
        ),
        collector=collector,
        logger=_logger(),
    )


_FIXTURES: tuple[tuple[bytes, dict[str, Any]], ...] = (
    (
        b"""
        <html>
          <head>
            <title>Primary Title</title>
            <link rel="canonical" href="https://example.test/canonical"/>
            <meta name="robots" content="index, follow"/>
            <meta http-equiv="refresh" content="5; url=https://example.test/next"/>
          </head>
          <body><p>Hello</p></body>
        </html>
        """,
        {
            "title": "Primary Title",
            "canonical_url": "https://example.test/canonical",
            "meta_robots": ("index", "follow"),
            "meta_refresh_url": "https://example.test/next",
        },
    ),
    (
        b"""
        <html>
          <head>
            <meta property="og:title" content="OG Only"/>
            <meta name="googlebot" content="noindex, noarchive"/>
            <meta name="robots" content="nofollow"/>
          </head>
          <body></body>
        </html>
        """,
        {
            "title": "OG Only",
            "canonical_url": None,
            "meta_robots": ("noindex", "noarchive", "nofollow"),
            "meta_refresh_url": None,
        },
    ),
    (
        b"<html><body><p>no metadata</p></body></html>",
        {
            "title": None,
            "canonical_url": None,
            "meta_robots": (),
            "meta_refresh_url": None,
        },
    ),
)


@pytest.mark.parametrize(
    ("html", "expected"),
    _FIXTURES,
    ids=["full", "og_robots", "empty"],
)
def test_metadata_fixture_projection(
    tmp_path: Path,
    html: bytes,
    expected: dict[str, Any],
) -> None:
    result = _page_content_extractor().extract(
        fetch_result=_make_fetch_result(tmp_path=tmp_path, html=html)
    )
    assert project_page_metadata(result) == expected
    assert not hasattr(result, "parsed_document")
