"""Handler registry composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.collection.http_rules import TimeoutRulesSettings
from config.collection.identity import IdentitySettings
from config.collection.modality_acceptance import ModalityAcceptanceSettings
from config.collection.processors import (
    AudioProcessorSettings,
    DocumentProcessorSettings,
    FeedProcessorSettings,
    ImageProcessorSettings,
    PageProcessorSettings,
    ProcessorSettings,
    VideoProcessorSettings,
)
from config.coverage.settings import CoverageSettings
from config.preprocessing.media_settings import (
    DiarizationSettings,
    OcrBackendSettings,
    TranscriptionSettings,
)
from config.settings.root import Settings
from crawler.analysis.enrichment.lanes.analysis_result_writer import (
    AnalysisResultWriter,
)
from crawler.analysis.enrichment.lanes.analysis_router import AnalysisRouter
from crawler.analysis.enrichment.lanes.analysis_worker_lane import (
    AnalysisWorkerLane,
)
from crawler.classification.media_kind import MediaKind
from crawler.coverage.state import CoverageState
from crawler.discovery.discovery_task_builder import DiscoveryTaskBuilder
from crawler.extraction.modalities.page_content_extractor import (
    PageContentExtractor,
)
from crawler.extraction.urls.normalizer import UrlNormalizer
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
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from crawler.processing.processors.processor_failure_handler import (
    ProcessorFailureHandler,
)
from crawler.scheduling.url_scheduler import UrlScheduler
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from logger.factory import ProjectLoggerFactory
from orchestration.composition.runtime.handler_composition.audio import (
    build_audio_handler,
)
from orchestration.composition.runtime.handler_composition.document import (
    build_document_handler,
)
from orchestration.composition.runtime.handler_composition.feed import (
    build_feed_handler,
)
from orchestration.composition.runtime.handler_composition.image import (
    build_image_handler,
)
from orchestration.composition.runtime.handler_composition.page import (
    build_page_handler,
)
from orchestration.composition.runtime.handler_composition.video import (
    build_video_handler,
)
from shared.runtime_primitives import IdGenerator

if TYPE_CHECKING:
    pass


def build_handler_registry(
    *,
    # Page
    page_settings: PageProcessorSettings,
    coverage_settings: CoverageSettings,
    page_content_extractor: PageContentExtractor,
    discovery_task_builder: DiscoveryTaskBuilder,
    url_filter: UrlAdmissionFilter,
    url_normalizer: UrlNormalizer,
    scheduler: UrlScheduler,
    writer: DatasetWriter,
    coverage: CoverageState,
    host_normalizer: HostNormalizer,
    id_generator: IdGenerator,
    rejected_reporter: object | None,
    logs: ProjectLoggerFactory,
    failure_handler: ProcessorFailureHandler,
    # Feed
    feed_settings: FeedProcessorSettings,
    # Image
    image_settings: ImageProcessorSettings,
    ocr_settings: OcrBackendSettings,
    modality_acceptance: ModalityAcceptanceSettingsCatalog,
    # Audio
    audio_settings: AudioProcessorSettings,
    transcription_settings: TranscriptionSettings,
    diarization_settings: DiarizationSettings,
    # Video
    video_settings: VideoProcessorSettings,
    video_transcription_settings: TranscriptionSettings,
    video_diarization_settings: DiarizationSettings,
    video_ocr_settings: OcrBackendSettings,
    video_acceptance_settings: ModalityAcceptanceSettings,
    timeout_settings: TimeoutRulesSettings,
    identity_settings: IdentitySettings,
    video_metadata_probe_bytes: int | None,
    network_access_guard: NetworkAddressGuard,
    redirector: RedirectRulesValidator,
    # Document
    document_settings: DocumentProcessorSettings,
    document_ocr_settings: OcrBackendSettings,
) -> dict[MediaKind, PersistingProcessor[object, object]]:
    """Build the complete handler registry for all media kinds."""

    return {
        MediaKind.PAGE: build_page_handler(
            page_settings=page_settings,
            coverage_settings=coverage_settings,
            page_content_extractor=page_content_extractor,
            discovery_task_builder=discovery_task_builder,
            url_filter=url_filter,
            url_normalizer=url_normalizer,
            scheduler=scheduler,
            writer=writer,
            coverage=coverage,
            host_normalizer=host_normalizer,
            id_generator=id_generator,
            rejected_reporter=None,
            logs=logs,
            failure_handler=failure_handler,
        ),
        MediaKind.FEED: build_feed_handler(
            feed_settings=feed_settings,
            url_filter=url_filter,
            url_normalizer=url_normalizer,
            scheduler=scheduler,
            writer=writer,
            id_generator=id_generator,
            host_normalizer=host_normalizer,
            logs=logs,
            failure_handler=failure_handler,
        ),
        MediaKind.IMAGE: build_image_handler(
            image_settings=image_settings,
            image_acceptance=modality_acceptance.image,
            ocr_settings=ocr_settings,
            writer=writer,
            logs=logs,
            failure_handler=failure_handler,
        ),
        MediaKind.AUDIO: build_audio_handler(
            audio_settings=audio_settings,
            transcription_settings=transcription_settings,
            diarization_settings=diarization_settings,
            writer=writer,
            logs=logs,
            failure_handler=failure_handler,
        ),
        MediaKind.VIDEO: build_video_handler(
            video_settings=video_settings,
            transcription_settings=video_transcription_settings,
            diarization_settings=video_diarization_settings,
            ocr_settings=video_ocr_settings,
            video_acceptance_settings=video_acceptance_settings,
            timeout_settings=timeout_settings,
            identity_settings=identity_settings,
            video_metadata_probe_bytes=video_metadata_probe_bytes,
            writer=writer,
            network_guard=network_access_guard,
            redirector=redirector,
            id_generator=id_generator,
            logs=logs,
            failure_handler=failure_handler,
        ),
        MediaKind.DOCUMENT: build_document_handler(
            document_settings=document_settings,
            ocr_settings=document_ocr_settings,
            writer=writer,
            logs=logs,
            failure_handler=failure_handler,
        ),
    }


__all__ = ["build_handler_registry", "build_analysis_router"]


def build_analysis_router(
    *,
    processor_settings: ProcessorSettings,
    logger_factory: ProjectLoggerFactory,
) -> AnalysisRouter:
    """Build the analysis router with configured worker lanes."""

    analysis_logger = logger_factory.get_logger_for(AnalysisRouter)
    result_writer = AnalysisResultWriter(logger=analysis_logger)
    lanes = {
        MediaKind.IMAGE: _build_lane(
            name="image",
            settings=processor_settings.image,
            result_writer=result_writer,
            logger=analysis_logger,
        ),
        MediaKind.AUDIO: _build_lane(
            name="audio",
            settings=processor_settings.audio,
            result_writer=result_writer,
            logger=analysis_logger,
        ),
        MediaKind.VIDEO: _build_lane(
            name="video",
            settings=processor_settings.video,
            result_writer=result_writer,
            logger=analysis_logger,
        ),
        MediaKind.DOCUMENT: _build_lane(
            name="document",
            settings=processor_settings.document,
            result_writer=result_writer,
            logger=analysis_logger,
        ),
    }
    return AnalysisRouter(
        lanes=lanes,
        record_writer=result_writer,
        logger=analysis_logger,
    )


def _build_lane(
    *,
    name: str,
    settings: object,
    result_writer: AnalysisResultWriter,
    logger: object,
) -> AnalysisWorkerLane:
    """Build a single analysis worker lane."""

    return AnalysisWorkerLane(
        name=name,
        worker_count=settings.analysis_workers,
        queue_size=settings.analysis_queue_size,
        timeout_seconds=settings.analysis_timeout_seconds,
        result_sink=result_writer.write_result,
        logger=logger,
    )


__all__ = ["build_handler_registry", "build_analysis_router"]
