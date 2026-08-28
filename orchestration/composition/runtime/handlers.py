"""Compose the central crawl task processor and handler routing.

This module is the public facade for the handler composition. It assembles
the handler registry, analysis router, and task processor from the
modular subgraphs in handler_composition/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from config.collection.processors import PageProcessorSettings
from config.coverage.settings import CoverageSettings
from config.settings.root import Settings
from crawler.classification.media_kind import MediaKind
from crawler.coverage.fetch_admission import CoverageFetchGate
from crawler.coverage.state import CoverageState
from crawler.discovery.discovery_task_builder import DiscoveryTaskBuilder
from crawler.extraction.modalities.page_content_extractor import (
    PageContentExtractor,
)
from crawler.extraction.urls.normalizer import UrlNormalizer
from crawler.fetching.fetcher import FetchOrchestrator
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.network_access.network_address_guard import (
    NetworkAddressGuard,
)
from crawler.governance.redirect.redirect_rules_validator import (
    RedirectRulesValidator,
)
from crawler.governance.url_filter.url_admission_filter import (
    UrlAdmissionFilter,
)
from crawler.processing.processors.processor_failure_handler import (
    ProcessorFailureHandler,
)
from crawler.processing.routing.crawl_result_router import CrawlResultRouter
from crawler.processing.task_processor import CrawlTaskProcessor
from crawler.scheduling.url_scheduler import UrlScheduler
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from logger.factory import ProjectLoggerFactory
from orchestration.composition.runtime.handler_composition import (
    build_analysis_router,
    build_handler_registry,
)
from shared.runtime_primitives import IdGenerator

if TYPE_CHECKING:
    from config.collection.http_rules import TimeoutRulesSettings
    from config.collection.identity import IdentitySettings
    from config.collection.modality_acceptance import (
        ModalityAcceptanceSettings,
    )


def build_task_processor(
    *,
    settings: Settings,
    page_settings: PageProcessorSettings,
    logger_factory: ProjectLoggerFactory,
    id_generator: IdGenerator,
    fetcher: FetchOrchestrator,
    coverage_tracker: CoverageState,
    scheduler: UrlScheduler,
    dataset_writer: DatasetWriter,
    url_filter: UrlAdmissionFilter,
    url_normalizer: UrlNormalizer,
    host_normalizer: HostNormalizer,
    page_content_extractor: PageContentExtractor,
    discovery_task_builder: DiscoveryTaskBuilder,
    rejected_discovery_reporter: Any | None = None,
    network_access_guard: NetworkAddressGuard,
    redirector: RedirectRulesValidator,
) -> CrawlTaskProcessor:
    """Build routing, analysis lanes, and the central task processor."""

    task_settings = settings.collection.processors.task
    failure_handler = ProcessorFailureHandler(
        default_retry_wait_seconds=settings.collection.scheduling.default_retry_wait_seconds,
    )

    handlers = build_handler_registry(
        # Page
        page_settings=page_settings,
        coverage_settings=settings.coverage,
        page_content_extractor=page_content_extractor,
        discovery_task_builder=discovery_task_builder,
        url_filter=url_filter,
        url_normalizer=url_normalizer,
        scheduler=scheduler,
        writer=dataset_writer,
        coverage=coverage_tracker,
        host_normalizer=host_normalizer,
        id_generator=id_generator,
        rejected_reporter=None,
        logs=logger_factory,
        failure_handler=failure_handler,
        # Feed
        feed_settings=settings.collection.processors.feed,
        # Image
        image_settings=settings.collection.processors.image,
        ocr_settings=settings.preprocessing.ocr,
        modality_acceptance=settings.collection.modality_acceptance,
        # Audio
        audio_settings=settings.collection.processors.audio,
        transcription_settings=settings.preprocessing.transcription,
        diarization_settings=settings.preprocessing.diarization,
        # Video
        video_settings=settings.collection.processors.video,
        video_transcription_settings=settings.preprocessing.transcription,
        video_diarization_settings=settings.preprocessing.diarization,
        video_ocr_settings=settings.preprocessing.ocr,
        video_acceptance_settings=settings.collection.modality_acceptance.video,
        timeout_settings=settings.collection.http_rules.timeouts,
        identity_settings=settings.collection.identity,
        video_metadata_probe_bytes=settings.collection.fetcher.video_metadata_probe_bytes,
        network_access_guard=network_access_guard,
        redirector=redirector,
        # Document
        document_settings=settings.collection.processors.document,
        document_ocr_settings=settings.preprocessing.ocr,
    )

    analysis_router = build_analysis_router(
        processor_settings=settings.collection.processors,
        logger_factory=logger_factory,
    )

    result_router = CrawlResultRouter(
        handlers_by_result_kind=handlers,
        analysis_router=analysis_router,
        failure_handler=failure_handler,
        drop_unknown_tasks=task_settings.drop_unknown_tasks,
        logger=logger_factory.get_logger_for(CrawlResultRouter),
    )

    return CrawlTaskProcessor(
        fetch_service=fetcher,
        coverage_gate=CoverageFetchGate(
            settings=settings.coverage,
            coverage_tracker=coverage_tracker,
        ),
        failure_handler=failure_handler,
        result_router=result_router,
        logger=logger_factory.get_logger_for(CrawlTaskProcessor),
    )


__all__ = ["build_task_processor"]
