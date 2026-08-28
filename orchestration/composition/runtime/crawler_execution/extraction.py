"""Extraction runtime composition.

Builds page content extraction and discovery task construction.
"""

from __future__ import annotations

from config.settings.root import Settings
from crawler.discovery.discovery_task_builder import DiscoveryTaskBuilder
from crawler.extraction.modalities.page_content_extractor import (
    PageContentExtractor,
)
from crawler.extraction.urls.normalizer import UrlNormalizer
from logger.factory import ProjectLoggerFactory
from shared.runtime_primitives import IdGenerator


def build_extraction_runtime(
    *,
    settings: Settings,
    url_normalizer: UrlNormalizer,
    logger_factory: ProjectLoggerFactory,
    id_generator: IdGenerator,
) -> tuple[PageContentExtractor, DiscoveryTaskBuilder]:
    """Build the page extraction and discovery graph."""
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
    from crawler.extraction.extensions_detector import ExtensionDetector
    from crawler.extraction.html.html_parser import HtmlParser
    from crawler.extraction.links.extractor import LinkExtractor
    from crawler.extraction.modalities.page_content_extractor import (
        PageContentExtractor,
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

    collection = settings.collection
    asset_settings = collection.asset_extractor

    extension_detector = ExtensionDetector(
        settings=collection.extension_detector,
        logger=logger_factory.get_logger_for(ExtensionDetector),
    )
    html_parser = HtmlParser(
        settings=collection.html_parser,
        logger=logger_factory.get_logger_for(HtmlParser),
    )
    link_extractor = LinkExtractor(
        settings=collection.link_extractor,
        candidate_resolution=UrlCandidateResolution(
            url_normalizer=url_normalizer,
            logger=logger_factory.get_logger_for(UrlCandidateResolution),
            include_data_urls=False,
        ),
        logger=logger_factory.get_logger_for(LinkExtractor),
    )
    embed_detector = VideoEmbedDetector()
    collector = AssetCandidateCollector(
        candidate_resolution=AssetCandidateResolution(
            url_candidate_resolution=UrlCandidateResolution(
                url_normalizer=url_normalizer,
                logger=logger_factory.get_logger_for(UrlCandidateResolution),
                include_data_urls=asset_settings.include_data_urls,
            ),
            deduper=AssetCandidateDeduper(),
        ),
        embed_detector=embed_detector,
    )
    page_content_extractor = PageContentExtractor(
        html_parser=html_parser,
        element_index_builder=PageElementIndexBuilder(),
        metadata_extractor=PageMetadataExtractor(),
        text_content_extractor=PageTextContentExtractor(
            settings=collection.processors.page.text_extraction,
        ),
        link_extractor=link_extractor,
        reference_extractors=_build_reference_extractors(
            settings=settings,
            embed_detector=embed_detector,
        ),
        collector=collector,
        logger=logger_factory.get_logger_for(PageContentExtractor),
    )
    discovery_task_builder = _build_discovery_task_builder(
        settings=settings,
        extension_detector=extension_detector,
        id_generator=id_generator,
        logger_factory=logger_factory,
    )
    return page_content_extractor, discovery_task_builder


def _build_reference_extractors(
    *,
    settings: Settings,
    embed_detector: VideoEmbedDetector,
) -> tuple:
    """Build enabled modality-specific reference extractors."""
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
        ReferenceExtractorProtocol,
    )
    from crawler.extraction.modalities.video_extractor import (
        VideoReferenceExtractor,
    )

    asset_settings = settings.collection.asset_extractor
    if not asset_settings.enabled:
        return ()

    extractors: list[ReferenceExtractorProtocol] = []

    if asset_settings.extract_images:
        extractors.append(
            ImageReferenceExtractor(
                include_icon_link_assets=asset_settings.include_icon_link_assets,
            ),
        )

    if asset_settings.extract_audio:
        extractors.append(
            AudioReferenceExtractor(
                include_icon_link_assets=asset_settings.include_icon_link_assets,
            ),
        )

    if asset_settings.extract_video:
        extractors.append(
            VideoReferenceExtractor(
                include_icon_link_assets=asset_settings.include_icon_link_assets,
                embed_detector=embed_detector,
            ),
        )

    if asset_settings.extract_documents:
        extractors.append(
            DocumentReferenceExtractor(
                include_stylesheets_as_documents=asset_settings.include_stylesheets_as_documents,
                include_script_assets=asset_settings.include_script_assets,
                include_font_assets=asset_settings.include_font_assets,
                include_icon_link_assets=asset_settings.include_icon_link_assets,
            ),
        )

    return tuple(extractors)


def _build_discovery_task_builder(
    *,
    settings: Settings,
    extension_detector: ExtensionDetector,
    id_generator: IdGenerator,
    logger_factory: ProjectLoggerFactory,
) -> DiscoveryTaskBuilder:
    """Build the discovery-task builder for newly extracted URLs."""
    from crawler.discovery.discovery_task_builder import DiscoveryTaskBuilder

    return DiscoveryTaskBuilder(
        settings=settings.collection.url_extractor,
        extension_detector=extension_detector,
        id_generator=id_generator,
        logger=logger_factory.get_logger_for(DiscoveryTaskBuilder),
    )
