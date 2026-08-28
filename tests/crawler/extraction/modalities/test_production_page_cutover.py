"""Cutover tests: PageContentExtractor is the production page entrypoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config.collection.discovery import HtmlParserSettings
from config.collection.extraction import (
    AssetExtractorSettings,
    LinkExtractorSettings,
    UrlExtractorSettings,
    UrlNormalizerSettings,
)
from config.collection.processors import (
    PageProcessorSettings,
    PageTextExtractionSettings,
)
from crawler.classification.media_kind import MediaKind
from crawler.discovery.discovery_task_builder import DiscoveryTaskBuilder
from crawler.extraction.assets.candidate.asset_candidate_collector import (
    AssetCandidateCollector,
)
from crawler.extraction.assets.candidate.asset_candidate_deduper import (
    AssetCandidateDeduper,
)
from crawler.extraction.assets.candidate.asset_candidate_resolution import (
    AssetCandidateResolution,
)
from crawler.extraction.assets.candidate.asset_extraction_records import (
    AssetCandidate,
    AssetDiscoveryResult,
)
from crawler.extraction.assets.embedded_media.video_embed_detector import (
    VideoEmbedDetector,
)
from crawler.extraction.candidates.url_candidate_resolution import (
    ExtractionCandidate,
    UrlCandidateResolution,
)
from crawler.extraction.extensions_detector import ExtensionDetector
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
    PageMetadata,
    PageMetadataExtractor,
)
from crawler.extraction.modalities.page_text_content_extractor import (
    PageTextContent,
    PageTextContentExtractor,
)
from crawler.extraction.modalities.video_extractor import (
    VideoReferenceExtractor,
)
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.fetching.results.payload import FetchedPayload
from crawler.fetching.results.result import FetchResult
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.processing.handlers.page_handler import PageHandler
from crawler.processing.processors.processor_failure_handler import (
    ProcessorFailureHandler,
)
from tests.support.logging import TEST_LOGGER

_PAGE_URL = "https://example.test/page"


def _logger() -> SimpleNamespace:
    return SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
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


def _resolution() -> UrlCandidateResolution:
    return UrlCandidateResolution(
        url_normalizer=UrlNormalizer(
            settings=UrlNormalizerSettings(),
            logger=_logger(),
            host_normalizer=HostNormalizer(),
        ),
        logger=TEST_LOGGER,
    )


def _collector() -> AssetCandidateCollector:
    return AssetCandidateCollector(
        candidate_resolution=AssetCandidateResolution(
            url_candidate_resolution=_resolution(),
            deduper=AssetCandidateDeduper(),
        ),
        embed_detector=VideoEmbedDetector(),
    )


def _reference_extractors(
    *,
    asset_settings: AssetExtractorSettings | None = None,
):
    settings = asset_settings or AssetExtractorSettings()
    extractors = []
    if settings.enabled and settings.extract_images:
        extractors.append(ImageReferenceExtractor())
    if settings.enabled and settings.extract_audio:
        extractors.append(AudioReferenceExtractor())
    if settings.enabled and settings.extract_video:
        extractors.append(
            VideoReferenceExtractor(embed_detector=VideoEmbedDetector())
        )
    if settings.enabled and settings.extract_documents:
        extractors.append(DocumentReferenceExtractor())
    return tuple(extractors)


def _page_content_extractor(
    *,
    asset_settings: AssetExtractorSettings | None = None,
    link_enabled: bool = True,
) -> PageContentExtractor:
    return PageContentExtractor(
        html_parser=_html_parser(),
        element_index_builder=PageElementIndexBuilder(),
        metadata_extractor=PageMetadataExtractor(),
        text_content_extractor=PageTextContentExtractor(
            settings=PageTextExtractionSettings()
        ),
        link_extractor=LinkExtractor(
            settings=LinkExtractorSettings(enabled=link_enabled),
            candidate_resolution=_resolution(),
            logger=_logger(),
        ),
        reference_extractors=_reference_extractors(
            asset_settings=asset_settings
        ),
        collector=_collector(),
        logger=_logger(),
    )


def _fetch_result(*, tmp_path: Path, html: bytes) -> FetchResult:
    path = tmp_path / "body.html"
    path.write_bytes(html)
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
        payload=FetchedPayload(
            temp_path=path,
            byte_size=len(html),
            sha256_hex="0" * 64,
            sniff_bytes=html[:64],
            chunk_count=1,
        ),
        body_sha256="0" * 64,
    )


def test_settings_disable_images_only(tmp_path: Path) -> None:
    html = b"""
    <html><body>
      <img src="/a.jpg"/><audio src="/a.mp3"/><video src="/v.mp4"/>
      <object data="/d.pdf"/>
    </body></html>
    """
    settings = AssetExtractorSettings(extract_images=False)
    result = _page_content_extractor(asset_settings=settings).extract(
        fetch_result=_fetch_result(tmp_path=tmp_path, html=html)
    )
    assert result.asset_discovery.images == ()
    assert result.asset_discovery.audio
    assert result.asset_discovery.video
    assert result.asset_discovery.documents


def test_settings_disable_all_assets(tmp_path: Path) -> None:
    html = b"""
    <html><body>
      <img src="/a.jpg"/><audio src="/a.mp3"/>
      <a href="/next">n</a>
    </body></html>
    """
    settings = AssetExtractorSettings(enabled=False)
    result = _page_content_extractor(asset_settings=settings).extract(
        fetch_result=_fetch_result(tmp_path=tmp_path, html=html)
    )
    assert result.asset_discovery.total_assets == 0
    assert result.links


def test_discovery_task_builder_without_dom() -> None:
    from config.collection.discovery import ExtensionDetectorSettings

    builder = DiscoveryTaskBuilder(
        settings=UrlExtractorSettings(),
        extension_detector=ExtensionDetector(
            settings=ExtensionDetectorSettings(),
            logger=_logger(),
        ),
        id_generator=SimpleNamespace(generate=lambda: "task-001"),
        logger=_logger(),
    )
    result = PageExtractionResult(
        requested_url=_PAGE_URL,
        final_url=_PAGE_URL,
        encoding="utf-8",
        metadata=PageMetadata(title="T"),
        text_content=PageTextContent(
            text="hello",
            text_preview="hello",
            markdown="hello",
            headings=(),
            char_count=5,
            code_block_count=0,
            boilerplate_ratio=0.0,
            extraction_warnings=(),
        ),
        links=(
            ExtractionCandidate(
                url="https://example.test/next",
                source_type="discovered_link",
            ),
        ),
        asset_discovery=AssetDiscoveryResult(parent_url=_PAGE_URL),
    )
    tasks = builder.build_page_tasks(
        source_name="src",
        parent_url=_PAGE_URL,
        parent_depth=1,
        links=result.links,
        assets=result.asset_discovery,
        max_tasks=10,
        base_url=_PAGE_URL,
    )
    assert any(task.url.endswith("/next") for task in tasks)
    assert all(task.task_id == "task-001" for task in tasks)


def test_build_page_tasks_resolves_depth_per_kind() -> None:
    from config.collection.discovery import ExtensionDetectorSettings

    builder = DiscoveryTaskBuilder(
        settings=UrlExtractorSettings(),
        extension_detector=ExtensionDetector(
            settings=ExtensionDetectorSettings(),
            logger=_logger(),
        ),
        id_generator=SimpleNamespace(generate=lambda: "task-001"),
        logger=_logger(),
    )

    def asset(url: str, kind: str) -> AssetCandidate:
        return AssetCandidate(
            url=url,
            kind=kind,  # type: ignore[arg-type]
            parent_url=_PAGE_URL,
            source_attribute="src",
            source_tag="media",
        )

    result = PageExtractionResult(
        requested_url=_PAGE_URL,
        final_url=_PAGE_URL,
        encoding="utf-8",
        metadata=PageMetadata(title="T"),
        text_content=PageTextContent(
            text="hello",
            text_preview="hello",
            markdown="hello",
            headings=(),
            char_count=5,
            code_block_count=0,
            boilerplate_ratio=0.0,
            extraction_warnings=(),
        ),
        links=(
            ExtractionCandidate(
                url="https://example.test/next",
                source_type="discovered_link",
            ),
            ExtractionCandidate(
                url="https://example.test/feed",
                kind="feed",
                source_type="discovered_link",
            ),
        ),
        asset_discovery=AssetDiscoveryResult(
            parent_url=_PAGE_URL,
            images=(asset("https://example.test/a.jpg", "image"),),
            audio=(asset("https://example.test/a.mp3", "audio"),),
            video=(asset("https://example.test/v.mp4", "video"),),
            documents=(asset("https://example.test/d.pdf", "document"),),
        ),
    )
    tasks = builder.build_page_tasks(
        source_name="src",
        parent_url=_PAGE_URL,
        parent_depth=4,
        links=result.links,
        assets=result.asset_discovery,
        max_tasks=20,
        base_url=_PAGE_URL,
    )
    by_url = {task.url: task for task in tasks}

    assert by_url["https://example.test/next"].depth == 5
    assert by_url["https://example.test/feed"].depth == 5
    assert by_url["https://example.test/a.jpg"].depth == 4
    assert by_url["https://example.test/a.mp3"].depth == 4
    assert by_url["https://example.test/v.mp4"].depth == 4
    assert by_url["https://example.test/d.pdf"].depth == 4
    assert by_url["https://example.test/a.jpg"].context.source_page_depth == 4
    assert by_url["https://example.test/a.jpg"].context.source_page_url == (
        _PAGE_URL
    )
    assert by_url["https://example.test/next"].context is None


@pytest.mark.asyncio
async def test_page_handler_uses_page_content_extractor(
    tmp_path: Path,
) -> None:
    extractor = MagicMock()
    extraction = PageExtractionResult(
        requested_url=_PAGE_URL,
        final_url=_PAGE_URL,
        encoding="utf-8",
        metadata=PageMetadata(
            title="Hi",
            meta_robots=(),
            canonical_url=None,
        ),
        text_content=PageTextContent(
            text="x" * 100,
            text_preview="x" * 100,
            markdown="x" * 100,
            headings=(),
            char_count=100,
            code_block_count=0,
            boilerplate_ratio=0.0,
            extraction_warnings=(),
        ),
        links=(),
        asset_discovery=AssetDiscoveryResult(parent_url=_PAGE_URL),
    )
    extractor.extract.return_value = extraction
    handler = PageHandler(
        settings=PageProcessorSettings(min_html_chars=10),
        dataset_writer=MagicMock(),
        logger=_logger(),
        failure_handler=ProcessorFailureHandler(
            default_retry_wait_seconds=5.0
        ),
        page_content_extractor=extractor,
        discovery_task_builder=MagicMock(),
        url_filter=MagicMock(),
        url_normalizer=MagicMock(),
        scheduler=MagicMock(),
        cap_resolver=MagicMock(),
        coverage_tracker=MagicMock(),
        focus_asset_boost=1.0,
        host_normalizer=MagicMock(),
        id_generator=SimpleNamespace(generate=lambda: "task-001"),
    )
    fetch = _fetch_result(
        tmp_path=tmp_path,
        html=b"<html><body><p>hi</p></body></html>",
    )
    task = SimpleNamespace(source_name="s", depth=0)
    result = await handler.prepare_analysis(task=task, result=fetch)
    assert result is extraction
    extractor.extract.assert_called_once()
    ok, reason, payload = await handler.validate_result(
        task=task, result=fetch, analysis=result
    )
    assert ok is True
    assert reason is None
    assert payload["page_char_count"] == 100
    enrichment = await handler.build_enrichment(
        task=task, result=fetch, analysis=result
    )
    assert enrichment["page_text_preview"]
    assert "page_extraction_artifact" in enrichment
    assert "page_markdown" not in enrichment
    assert isinstance(enrichment["page_extraction_artifact"], dict)
    assert "text" in enrichment["page_extraction_artifact"]


def test_architecture_imports_forbid_removed_modules() -> None:
    import importlib

    for name in (
        "crawler.analysis.enrichment.pages.page_analyzer",
        "crawler.extraction.assets.asset_extractor",
        "crawler.extraction.urls.extractor",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)

    import inspect

    from orchestration.composition.runtime.handler_composition.page import (
        build_page_handler,
    )

    source = inspect.getsource(build_page_handler)
    assert "preprocessing" not in source
    assert "PageAnalyzer" not in source
