"""Tests for DocumentReferenceExtractor draft-only document discovery."""

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
from crawler.extraction.modalities.document_extractor import (
    DocumentReferenceExtractor,
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


def _extract_document_candidates(html: bytes) -> tuple[str, ...]:
    document = _html_parser().parse(body=html, encoding="utf-8")
    index = PageElementIndexBuilder().build(document=document)
    drafts = DocumentReferenceExtractor().extract_references(
        index=index,
        context=PageExtractionContext(page_url=_PAGE_URL),
    )
    return tuple(d.candidate for d in drafts)


def test_object_track_and_pdf_embed() -> None:
    html = b"""
    <html><body>
      <object data="/report.pdf"></object>
      <track src="/captions.vtt" kind="captions"/>
      <embed src="/sheet.xlsx"/>
      <iframe src="/doc.pdf"></iframe>
      <iframe src="https://www.youtube.com/embed/abc"></iframe>
    </body></html>
    """
    candidates = _extract_document_candidates(html)
    assert "/report.pdf" in candidates
    assert "/captions.vtt" in candidates
    assert "/sheet.xlsx" in candidates
    assert "/doc.pdf" in candidates
    assert not any("youtube.com" in c for c in candidates)


def test_inline_and_jsonld_transcript_document() -> None:
    html = b"""
    <html>
      <head>
        <script type="application/ld+json">
          {"@type":"VideoObject","transcript":"/transcript.txt"}
        </script>
        <script>
          var cfg = {"download": "https://cdn.example.test/whitepaper.pdf"};
        </script>
      </head>
      <body></body>
    </html>
    """
    candidates = _extract_document_candidates(html)
    assert "/transcript.txt" in candidates
    assert "https://cdn.example.test/whitepaper.pdf" in candidates


def test_collect_drafts_dedupes_documents() -> None:
    html = b"""
    <html><body>
      <object data="/same.pdf"></object>
      <embed src="/same.pdf"/>
    </body></html>
    """
    document = _html_parser().parse(body=html, encoding="utf-8")
    index = PageElementIndexBuilder().build(document=document)
    drafts = DocumentReferenceExtractor().extract_references(
        index=index,
        context=PageExtractionContext(page_url=_PAGE_URL),
    )
    discovery = _collector().collect_drafts(drafts=drafts, page_url=_PAGE_URL)
    urls = [item.url for item in discovery.documents]
    assert urls.count("https://example.test/same.pdf") == 1
