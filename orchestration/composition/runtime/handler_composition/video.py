"""Video handler composition."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from config.collection.processors import VideoProcessorSettings
from config.preprocessing.media_settings import (
    DiarizationSettings,
    OcrBackendSettings,
    TranscriptionSettings,
)
from crawler.analysis.enrichment.media_files.media_payload_path_resolver import (
    MediaPayloadPathResolver,
)
from crawler.analysis.enrichment.media_files.media_temp_file_writer import (
    MediaTempFileWriter,
)
from crawler.analysis.enrichment.video.video_analyzer import VideoAnalyzer
from crawler.analysis.enrichment.video.video_frame_analysis_service import (
    VideoFrameAnalysisService,
)
from crawler.analysis.enrichment.video.video_frame_sampler import (
    VideoFrameSampler,
)
from crawler.analysis.enrichment.video.video_optional_analysis_service import (
    VideoOptionalAnalysisService,
)
from crawler.analysis.enrichment.video.video_probe_download import (
    UrlopenTransport,
    VideoFullProbeDownloader,
    VideoTailProbeDownloader,
)
from crawler.analysis.enrichment.video.video_probe_resolver import (
    VideoProbeResolver,
)
from crawler.extraction.payloads.video_payload_extractor import (
    VideoPayloadExtractor,
)
from crawler.processing.handlers.video_handler import VideoHandler
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from logger.factory import ProjectLoggerFactory
from orchestration.composition.preprocessing_dependencies import (
    build_ocr_engine,
)
from preprocessing.media.adapters.opencv_video import (
    OpenCvFrameProcessor,
    OpenCvVideoClipWriter,
    OpenCvVideoReader,
)
from preprocessing.media.adapters.pyav_media import (
    CompositeVideoNormalizerBackend,
    PyAvAudioTrackExtractor,
    PyAvContainerProbe,
)
from preprocessing.media.adapters.whisper_model_loader import (
    WhisperModelLoader,
)
from preprocessing.media.speech.speaker_diarizer import get_diarization_service
from preprocessing.media.speech.speech_transcriber import SpeechTranscriber
from preprocessing.media.video.mp4_tail_metadata import Mp4TailMetadataReader
from preprocessing.media.video.video_frame_ocr import (
    VideoFrameTextExtractionService,
)
from preprocessing.media.video.video_keyframe_selector import (
    select_keyframe_metadata,
)

if TYPE_CHECKING:
    from config.collection.http_rules import TimeoutRulesSettings
    from config.collection.identity import IdentitySettings
    from config.collection.modality_acceptance import (
        ModalityAcceptanceSettings,
    )
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from crawler.governance.network_access.network_address_guard import (
        NetworkAddressGuard,
    )
    from crawler.governance.redirect.redirect_rules_validator import (
        RedirectRulesValidator,
    )
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )
    from shared.runtime_primitives import IdGenerator


_DEFAULT_METADATA_BYTES: int = 2_097_152


def _build_probe_transport(
    *,
    network_guard: NetworkAddressGuard,
    redirector: RedirectRulesValidator,
) -> UrlopenTransport:
    """Build a probe transport governed by network and redirect rules."""

    return UrlopenTransport(
        network_access_guard=network_guard,
        redirector=redirector,
        max_redirects=redirector.max_redirects,
    )


def _build_full_downloader(
    *,
    video_acceptance_settings: ModalityAcceptanceSettings,
    timeout_settings: TimeoutRulesSettings,
    identity_settings: IdentitySettings,
    network_guard: NetworkAddressGuard,
    redirector: RedirectRulesValidator,
) -> VideoFullProbeDownloader:
    """Build the bounded full-video fallback downloader."""

    return VideoFullProbeDownloader(
        max_bytes=video_acceptance_settings.fetch_max_bytes,
        timeout_seconds=timeout_settings.large_media_request_timeout_seconds,
        transport=_build_probe_transport(
            network_guard=network_guard,
            redirector=redirector,
        ),
        user_agent=identity_settings.user_agent,
    )


def _build_tail_downloader(
    *,
    video_metadata_probe_bytes: int | None,
    timeout_settings: TimeoutRulesSettings,
    identity_settings: IdentitySettings,
    network_guard: NetworkAddressGuard,
    redirector: RedirectRulesValidator,
) -> VideoTailProbeDownloader:
    """Build the bounded MP4-tail metadata downloader."""

    configured_tail_bytes = video_metadata_probe_bytes
    tail_bytes = (
        _DEFAULT_METADATA_BYTES
        if configured_tail_bytes is None
        else int(configured_tail_bytes)
    )

    return VideoTailProbeDownloader(
        tail_bytes=tail_bytes,
        timeout_seconds=timeout_settings.request_timeout_seconds,
        transport=_build_probe_transport(
            network_guard=network_guard,
            redirector=redirector,
        ),
        user_agent=identity_settings.user_agent,
    )


def _build_video_probe_resolver(
    *,
    video_acceptance_settings: ModalityAcceptanceSettings,
    timeout_settings: TimeoutRulesSettings,
    identity_settings: IdentitySettings,
    video_metadata_probe_bytes: int | None,
    network_guard: NetworkAddressGuard,
    redirector: RedirectRulesValidator,
    logs: ProjectLoggerFactory,
    container_probe: PyAvContainerProbe,
    payload_extractor: VideoPayloadExtractor,
) -> VideoProbeResolver:
    """Build the video probe resolver with its download dependencies."""

    return VideoProbeResolver(
        payload_extractor=payload_extractor,
        tail_probe_downloader=_build_tail_downloader(
            video_metadata_probe_bytes=video_metadata_probe_bytes,
            timeout_settings=timeout_settings,
            identity_settings=identity_settings,
            network_guard=network_guard,
            redirector=redirector,
        ),
        tail_metadata_reader=Mp4TailMetadataReader(),
        full_probe_downloader=_build_full_downloader(
            video_acceptance_settings=video_acceptance_settings,
            timeout_settings=timeout_settings,
            identity_settings=identity_settings,
            network_guard=network_guard,
            redirector=redirector,
        ),
        logger=logs.get_logger_for(VideoProbeResolver),
        container_probe=container_probe,
    )


def _build_media_path_resolver(
    *,
    logs: ProjectLoggerFactory,
) -> MediaPayloadPathResolver:
    """Build the temporary media payload path resolver."""

    return MediaPayloadPathResolver(
        writer=MediaTempFileWriter(
            logger=logs.get_logger_for(MediaTempFileWriter),
        ),
        logger=logs.get_logger_for(MediaPayloadPathResolver),
    )


def _build_transcriber(
    *,
    transcription_settings: TranscriptionSettings,
    logs: ProjectLoggerFactory,
    audio_stream_status: Callable[[Path], str] | None = None,
) -> SpeechTranscriber:
    """Build the optional Whisper-backed transcription executor."""

    model_repository = (
        WhisperModelLoader(settings=transcription_settings)
        if transcription_settings.enabled
        else None
    )

    return SpeechTranscriber(
        settings=transcription_settings,
        model_repository=model_repository,
        logger=logs.get_logger_for(SpeechTranscriber),
        audio_stream_status=audio_stream_status,
    )


def build_video_handler(
    *,
    video_settings: VideoProcessorSettings,
    transcription_settings: TranscriptionSettings,
    diarization_settings: DiarizationSettings,
    ocr_settings: OcrBackendSettings,
    video_acceptance_settings: ModalityAcceptanceSettings,
    timeout_settings: TimeoutRulesSettings,
    identity_settings: IdentitySettings,
    video_metadata_probe_bytes: int | None,
    writer: DatasetWriter,
    network_guard: NetworkAddressGuard,
    redirector: RedirectRulesValidator,
    id_generator: IdGenerator,
    logs: ProjectLoggerFactory,
    failure_handler: ProcessorFailureHandler,
) -> VideoHandler:
    """Build the video handler with explicit runtime dependencies."""

    analyzer = _build_video_analyzer(
        video_settings=video_settings,
        transcription_settings=transcription_settings,
        diarization_settings=diarization_settings,
        ocr_settings=ocr_settings,
        video_acceptance_settings=video_acceptance_settings,
        timeout_settings=timeout_settings,
        identity_settings=identity_settings,
        video_metadata_probe_bytes=video_metadata_probe_bytes,
        network_guard=network_guard,
        redirector=redirector,
        id_generator=id_generator,
        logs=logs,
    )

    return VideoHandler(
        settings=video_settings,
        dataset_writer=writer,
        logger=logs.get_logger_for(VideoHandler),
        failure_handler=failure_handler,
        analyzer=analyzer,
    )


def _build_video_analyzer(
    *,
    video_settings: VideoProcessorSettings,
    transcription_settings: TranscriptionSettings,
    diarization_settings: DiarizationSettings,
    ocr_settings: OcrBackendSettings,
    video_acceptance_settings: ModalityAcceptanceSettings,
    timeout_settings: TimeoutRulesSettings,
    identity_settings: IdentitySettings,
    video_metadata_probe_bytes: int | None,
    network_guard: NetworkAddressGuard,
    redirector: RedirectRulesValidator,
    id_generator: IdGenerator,
    logs: ProjectLoggerFactory,
) -> VideoAnalyzer:
    """Build the video-analysis pipeline and its reusable media adapters."""

    video_reader = OpenCvVideoReader()
    frame_processor = OpenCvFrameProcessor()
    container_probe = PyAvContainerProbe()
    audio_track_extractor = PyAvAudioTrackExtractor()

    probe_resolver = _build_video_probe_resolver(
        video_acceptance_settings=video_acceptance_settings,
        timeout_settings=timeout_settings,
        identity_settings=identity_settings,
        video_metadata_probe_bytes=video_metadata_probe_bytes,
        network_guard=network_guard,
        redirector=redirector,
        logs=logs,
        container_probe=container_probe,
        payload_extractor=VideoPayloadExtractor(),
    )

    frame_analysis = VideoFrameAnalysisService(
        frame_sampler=VideoFrameSampler(
            id_generator=id_generator,
            video_reader=video_reader,
            frame_processor=frame_processor,
        ),
        frame_text_extraction_service=VideoFrameTextExtractionService(
            ocr_engine=build_ocr_engine(settings=ocr_settings),
            frame_processor=frame_processor,
        ),
        transcription_executor=_build_transcriber(
            transcription_settings=transcription_settings,
            logs=logs,
            audio_stream_status=container_probe.audio_stream_status,
        ),
        keyframe_selector=select_keyframe_metadata,
        video_reader=video_reader,
        frame_processor=frame_processor,
        logger=logs.get_logger_for(VideoFrameAnalysisService),
    )

    optional_analysis = VideoOptionalAnalysisService(
        settings=video_settings,
        diarization_service=get_diarization_service(diarization_settings),
        audio_track_extractor=audio_track_extractor.extract_to_wav,
        video_normalizer=CompositeVideoNormalizerBackend(
            clip_writer=OpenCvVideoClipWriter(),
        ).normalize,
        logger=logs.get_logger_for(VideoOptionalAnalysisService),
    )

    return VideoAnalyzer(
        settings=video_settings,
        media_file_resolver=_build_media_path_resolver(logs=logs),
        probe_resolver=probe_resolver,
        frame_analysis=frame_analysis,
        optional_analysis=optional_analysis,
        logger=logs.get_logger_for(VideoAnalyzer),
    )


__all__ = ["build_video_handler"]
