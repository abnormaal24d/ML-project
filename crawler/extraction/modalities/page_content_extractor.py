"""Thin page-content orchestrator.

Owns the parse-once / index-once flow and dispatches to specialized
extractors. Does not implement modality rules, URL normalization, or
persistence. Never returns DOM objects.

Efficiency invariants:
* one HTML parse
* one structural index traversal
* one text-content traversal
* no per-modality DOM scan (reference extractors consume the index only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from crawler.extraction.assets.candidate.asset_extraction_records import (
    AssetCandidateDraft,
    AssetDiscoveryResult,
)
from crawler.extraction.candidates.url_candidate_resolution import (
    ExtractionCandidate,
)
from crawler.extraction.modalities.image_extractor import (
    PageExtractionContext,
)
from crawler.extraction.modalities.page_element_index import (
    PageElementIndex,
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

if TYPE_CHECKING:
    from crawler.extraction.assets.candidate.asset_candidate_collector import (
        AssetCandidateCollector,
    )
    from crawler.extraction.html.html_parser import HtmlParser
    from crawler.extraction.links.extractor import LinkExtractor
    from crawler.fetching.results.result import FetchResult
    from logger.project_logger import ProjectLogger


@runtime_checkable
class ReferenceExtractorProtocol(Protocol):
    """Read one page index and return unresolved modality references."""

    def extract_references(
        self,
        *,
        index: PageElementIndex,
        context: PageExtractionContext,
    ) -> tuple[AssetCandidateDraft, ...]: ...


@dataclass(frozen=True, slots=True)
class PageExtractionResult:
    """Immutable page extraction output without DOM leakage.

    Nested collaborators own their fields. Flat dual property accessors are
    intentionally not provided.
    """

    requested_url: str
    final_url: str
    encoding: str | None
    metadata: PageMetadata
    text_content: PageTextContent
    links: tuple[ExtractionCandidate, ...]
    asset_discovery: AssetDiscoveryResult


class PageContentExtractor:
    """Orchestrate body read → single parse → index → page extraction.

    Content rules stay in specialized collaborators. This class only sequences
    them and materializes a DOM-free result. Enabled modality extractors are
    injected as a configured tuple of reference extractors.
    """

    def __init__(
        self,
        *,
        html_parser: HtmlParser,
        element_index_builder: PageElementIndexBuilder,
        metadata_extractor: PageMetadataExtractor,
        text_content_extractor: PageTextContentExtractor,
        link_extractor: LinkExtractor,
        reference_extractors: tuple[ReferenceExtractorProtocol, ...],
        collector: AssetCandidateCollector,
        logger: ProjectLogger,
    ) -> None:
        if html_parser is None:
            raise ValueError("html_parser is required")
        if element_index_builder is None:
            raise ValueError("element_index_builder is required")
        if metadata_extractor is None:
            raise ValueError("metadata_extractor is required")
        if text_content_extractor is None:
            raise ValueError("text_content_extractor is required")
        if link_extractor is None:
            raise ValueError("link_extractor is required")
        if collector is None:
            raise ValueError("collector is required")

        self._html_parser = html_parser
        self._element_index_builder = element_index_builder
        self._metadata_extractor = metadata_extractor
        self._text_content_extractor = text_content_extractor
        self._link_extractor = link_extractor
        self._reference_extractors = tuple(reference_extractors)
        self._collector = collector
        self._logger = logger
        self._logger.debug("page_content_extractor_initialized")

    def extract(
        self,
        *,
        fetch_result: FetchResult,
        parent_text: str | None = None,
        parent_title: str | None = None,
        max_assets: int | None = None,
    ) -> PageExtractionResult:
        """Extract metadata, text, links, and configured media candidates.

        Guarantees:
        * body is read once;
        * HTML is parsed exactly once;
        * the element index is built with one structural traversal;
        * reference extractors consume the index only;
        * the result never embeds ``parsed_document`` or other DOM objects.
        """

        body = fetch_result.read_body_required()
        document = self._html_parser.parse(
            body=body,
            encoding=fetch_result.encoding,
        )
        index = self._element_index_builder.build(document=document)

        page_url = fetch_result.final_url or fetch_result.url
        metadata = self._metadata_extractor.extract(index=index)
        text_content = self._text_content_extractor.extract(document=document)
        links = self._link_extractor.extract_candidates(
            index=index,
            base_url=page_url,
        )
        context = PageExtractionContext(
            page_url=page_url,
            parent_text=parent_text,
            parent_title=parent_title,
        )
        asset_drafts: tuple[AssetCandidateDraft, ...] = tuple(
            draft
            for extractor in self._reference_extractors
            for draft in extractor.extract_references(
                index=index,
                context=context,
            )
        )
        asset_discovery = self._collector.collect_drafts(
            drafts=asset_drafts,
            page_url=page_url,
            max_results=max_assets,
        )

        self._logger.debug(
            "page_content_extracted",
            page_url=page_url,
            title=metadata.title,
            char_count=text_content.char_count,
            link_count=len(links),
            image_count=len(asset_discovery.images),
            audio_count=len(asset_discovery.audio),
            video_count=len(asset_discovery.video),
            document_count=len(asset_discovery.documents),
            rejected=len(asset_discovery.rejected),
        )

        return PageExtractionResult(
            requested_url=fetch_result.url,
            final_url=page_url,
            encoding=fetch_result.encoding,
            metadata=metadata,
            text_content=text_content,
            links=links,
            asset_discovery=asset_discovery,
        )
