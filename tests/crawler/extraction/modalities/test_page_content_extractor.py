"""Vertical-slice tests for PageContentExtractor (DOM-free page result)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from config.collection.discovery import HtmlParserSettings
from config.collection.extraction import LinkExtractorSettings
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
from crawler.extraction.modalities.audio_extractor import (
    AudioReferenceExtractor,
)
from crawler.extraction.modalities.document_extractor import (
    DocumentReferenceExtractor,
)
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
from crawler.extraction.modalities.video_extractor import (
    VideoReferenceExtractor,
)
from crawler.fetching.results.payload import FetchedPayload
from crawler.fetching.results.result import FetchResult
from tests.support.url_components import make_url_candidate_resolution

_PAGE_URL = "https://example.test/article"


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


def _url_resolution() -> UrlCandidateResolution:
    return make_url_candidate_resolution()


def _collector() -> AssetCandidateCollector:
    return AssetCandidateCollector(
        candidate_resolution=AssetCandidateResolution(
            url_candidate_resolution=_url_resolution(),
            deduper=AssetCandidateDeduper(),
        ),
        embed_detector=VideoEmbedDetector(),
    )


def _link_extractor(*, enabled: bool = True) -> LinkExtractor:
    return LinkExtractor(
        settings=LinkExtractorSettings(enabled=enabled),
        candidate_resolution=_url_resolution(),
        logger=_logger(),
    )


def _make_fetch_result(
    *,
    tmp_path: Path,
    html: bytes,
    url: str = _PAGE_URL,
    final_url: str | None = None,
    encoding: str | None = "utf-8",
) -> FetchResult:
    payload_path = tmp_path / "body.html"
    payload_path.write_bytes(html)
    payload = FetchedPayload(
        temp_path=payload_path,
        byte_size=len(html),
        sha256_hex="0" * 64,
        sniff_bytes=html[:64],
        chunk_count=1,
    )
    return FetchResult(
        url=url,
        final_url=final_url or url,
        status_code=200,
        headers={"content-type": "text/html"},
        fetched_at="2024-01-01T00:00:00Z",
        content_type="text/html",
        mime_type="text/html",
        encoding=encoding,
        language=None,
        kind=MediaKind.PAGE,
        payload=payload,
        body_sha256="0" * 64,
    )


def _build_extractor(
    *,
    html_parser: Any | None = None,
    index_builder: Any | None = None,
    metadata_extractor: Any | None = None,
    text_extractor: Any | None = None,
    link_extractor: Any | None = None,
    reference_extractors: tuple[Any, ...] | None = None,
    collector: Any | None = None,
) -> PageContentExtractor:
    if reference_extractors is None:
        reference_extractors = (
            ImageReferenceExtractor(include_icon_link_assets=True),
            AudioReferenceExtractor(include_icon_link_assets=True),
            VideoReferenceExtractor(
                include_icon_link_assets=True,
                embed_detector=VideoEmbedDetector(),
            ),
            DocumentReferenceExtractor(),
        )
    return PageContentExtractor(
        html_parser=html_parser or _html_parser(),
        element_index_builder=index_builder or PageElementIndexBuilder(),
        metadata_extractor=metadata_extractor or PageMetadataExtractor(),
        text_content_extractor=text_extractor
        or PageTextContentExtractor(settings=PageTextExtractionSettings()),
        link_extractor=link_extractor or _link_extractor(),
        reference_extractors=reference_extractors,
        collector=collector or _collector(),
        logger=_logger(),
    )


def test_extract_includes_all_modality_candidates(tmp_path: Path) -> None:
    html = b"""
    <html>
      <head>
        <title>Guide</title>
        <meta property="og:audio" content="/meta.mp3"/>
        <meta property="og:video" content="/meta.mp4"/>
        <link rel="canonical" href="https://example.test/guide"/>
      </head>
      <body>
        <main>
          <h1>Main Heading</h1>
          <p>Body paragraph about crawlers.</p>
          <a href="/next">Continue</a>
        </main>
        <audio src="/track.mp3"></audio>
        <video src="/clip.mp4" poster="/poster.jpg"></video>
        <img src="/cover.jpg"/>
        <object data="/spec.pdf"></object>
      </body>
    </html>
    """
    result = _build_extractor().extract(
        fetch_result=_make_fetch_result(tmp_path=tmp_path, html=html)
    )
    assert isinstance(result, PageExtractionResult)
    assert result.metadata.title == "Guide"
    assert "Main Heading" in result.text_content.text
    assert result.text_content.char_count > 0
    assert {item.url for item in result.links} >= {
        "https://example.test/next",
        "https://example.test/guide",
    }
    assert "https://example.test/cover.jpg" in {
        item.url for item in result.asset_discovery.images
    }
    assert "https://example.test/track.mp3" in {
        item.url for item in result.asset_discovery.audio
    }
    assert "https://example.test/clip.mp4" in {
        item.url for item in result.asset_discovery.video
    }
    assert "https://example.test/spec.pdf" in {
        item.url for item in result.asset_discovery.documents
    }


def test_reference_extractors_disabled_emit_no_assets(tmp_path: Path) -> None:
    html = b"""
    <html><body>
      <img src="/a.jpg"/><audio src="/a.mp3"/><video src="/v.mp4"/>
      <object data="/d.pdf"/>
    </body></html>
    """
    result = _build_extractor(reference_extractors=()).extract(
        fetch_result=_make_fetch_result(tmp_path=tmp_path, html=html)
    )
    assert result.asset_discovery.total_assets == 0


def test_extract_parses_html_exactly_once(tmp_path: Path) -> None:
    html = b"<html><body><img src='/a.jpg'/></body></html>"
    fetch_result = _make_fetch_result(tmp_path=tmp_path, html=html)
    real_parser = _html_parser()
    parse_calls: list[bytes] = []

    def _tracking_parse(*, body: bytes, encoding: str | None) -> Any:
        parse_calls.append(body)
        return real_parser.parse(body=body, encoding=encoding)

    html_parser = MagicMock()
    html_parser.parse.side_effect = _tracking_parse
    _build_extractor(html_parser=html_parser).extract(
        fetch_result=fetch_result
    )
    assert html_parser.parse.call_count == 1
    assert len(parse_calls) == 1


def test_extract_builds_index_exactly_once(tmp_path: Path) -> None:
    html = b"<html><body><img src='/a.jpg'/></body></html>"
    fetch_result = _make_fetch_result(tmp_path=tmp_path, html=html)
    real_builder = PageElementIndexBuilder()
    build_calls: list[Any] = []

    def _tracking_build(*, document: Any) -> Any:
        build_calls.append(document)
        return real_builder.build(document=document)

    index_builder = MagicMock()
    index_builder.build.side_effect = _tracking_build
    _build_extractor(index_builder=index_builder).extract(
        fetch_result=fetch_result
    )
    assert index_builder.build.call_count == 1


def test_text_extractor_does_not_reparse(tmp_path: Path) -> None:
    html = b"<html><body><main><p>Hello</p></main></body></html>"
    fetch_result = _make_fetch_result(tmp_path=tmp_path, html=html)
    real_parser = _html_parser()
    html_parser = MagicMock()
    html_parser.parse.side_effect = lambda *, body, encoding: (
        real_parser.parse(body=body, encoding=encoding)
    )
    text_extractor = MagicMock(
        wraps=PageTextContentExtractor(settings=PageTextExtractionSettings())
    )
    result = _build_extractor(
        html_parser=html_parser,
        text_extractor=text_extractor,
    ).extract(fetch_result=fetch_result)
    assert html_parser.parse.call_count == 1
    text_extractor.extract.assert_called_once()
    assert "document" in text_extractor.extract.call_args.kwargs
    assert "index" not in text_extractor.extract.call_args.kwargs
    assert "Hello" in result.text_content.text


def test_result_contains_no_dom_objects(tmp_path: Path) -> None:
    html = b"""
    <html>
      <head><title>Photo page</title></head>
      <body><main><p>Caption</p><a href="/more">More</a></main>
      <img src="/photo.jpg"/></body>
    </html>
    """
    result = _build_extractor().extract(
        fetch_result=_make_fetch_result(tmp_path=tmp_path, html=html)
    )
    assert not hasattr(result, "parsed_document")
    assert result.metadata.title == "Photo page"
    for value in (
        result.requested_url,
        result.final_url,
        result.encoding,
        result.metadata,
        result.text_content,
        result.links,
        result.asset_discovery,
    ):
        _assert_no_bs4_tag(value)


def test_extract_respects_max_assets(tmp_path: Path) -> None:
    html = b"""
    <html><body>
      <img src="/1.jpg"/><img src="/2.jpg"/><img src="/3.jpg"/>
      <audio src="/a.mp3"></audio>
    </body></html>
    """
    result = _build_extractor().extract(
        fetch_result=_make_fetch_result(tmp_path=tmp_path, html=html),
        max_assets=2,
    )
    assert result.asset_discovery.total_assets <= 2


def test_links_disabled_emits_no_links(tmp_path: Path) -> None:
    html = b"<html><body><a href='/x'>x</a></body></html>"
    result = _build_extractor(
        link_extractor=_link_extractor(enabled=False)
    ).extract(fetch_result=_make_fetch_result(tmp_path=tmp_path, html=html))
    assert result.links == ()


def _assert_no_bs4_tag(value: object) -> None:
    type_name = type(value).__name__
    module = type(value).__module__ or ""
    if type_name == "Tag" and "bs4" in module:
        raise AssertionError(f"DOM Tag leaked into result: {value!r}")
    if type_name in {"BeautifulSoup", "NavigableString"} and "bs4" in module:
        raise AssertionError(f"DOM object leaked into result: {value!r}")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_bs4_tag(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            _assert_no_bs4_tag(nested)
        return
    if hasattr(value, "__dataclass_fields__"):
        for field_name in value.__dataclass_fields__:  # type: ignore[attr-defined]
            _assert_no_bs4_tag(getattr(value, field_name))
