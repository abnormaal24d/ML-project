"""Audio handler composition."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from config.collection.processors import AudioProcessorSettings
from config.preprocessing.media_settings import DiarizationSettings, TranscriptionSettings
from crawler.analysis.enrichment.audio.audio_analyzer import AudioAnalyzer
from crawler.analysis.enrichment.media_files.media_payload_path_resolver import (
    MediaPayloadPathResolver,
)
from crawler.analysis.enrichment.media_files.media_temp_file_writer import (
    MediaTempFileWriter,
)
from crawler.extraction.payloads.audio_payload_extractor import AudioPayloadExtractor
from crawler.processing.handlers.audio_handler import AudioHandler
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from logger.factory import ProjectLoggerFactory
from preprocessing.media.adapters.whisper_model_loader import WhisperModelLoader
from preprocessing.media.audio.audio_emotion_analyzer import AudioEmotionAnalyzer
from preprocessing.media.audio.audio_event_analyzer import AudioEventAnalyzer
from preprocessing.media.speech.prosody_extractor import ProsodyExtractor
from preprocessing.media.speech.speaker_diarizer import get_diarization_service
from preprocessing.media.speech.speech_transcriber import SpeechTranscriber

if TYPE_CHECKING:
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
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


def build_audio_handler(
    *,
    audio_settings: AudioProcessorSettings,
    transcription_settings: TranscriptionSettings,
    diarization_settings: DiarizationSettings,
    writer: DatasetWriter,
    logs: ProjectLoggerFactory,
    failure_handler: ProcessorFailureHandler,
) -> AudioHandler:
    """Build the audio handler with explicit runtime dependencies."""

    return AudioHandler(
        settings=audio_settings,
        dataset_writer=writer,
        logger=logs.get_logger_for(AudioHandler),
        failure_handler=failure_handler,
        analyzer=_build_audio_analyzer(
            audio_settings=audio_settings,
            transcription_settings=transcription_settings,
            diarization_settings=diarization_settings,
            logs=logs,
        ),
    )


def _build_audio_analyzer(
    *,
    audio_settings: AudioProcessorSettings,
    transcription_settings: TranscriptionSettings,
    diarization_settings: DiarizationSettings,
    logs: ProjectLoggerFactory,
) -> AudioAnalyzer:
    """Build the audio analyzer and all enrichment dependencies."""

    return AudioAnalyzer(
        settings=audio_settings,
        media_file_resolver=_build_media_path_resolver(logs=logs),
        payload_extractor=AudioPayloadExtractor(),
        diarization_service=get_diarization_service(diarization_settings),
        transcription_executor=_build_transcriber(
            transcription_settings=transcription_settings,
            logs=logs,
        ),
        event_analyzer=AudioEventAnalyzer(),
        emotion_analyzer=AudioEmotionAnalyzer(
            prosody_extractor=ProsodyExtractor(),
        ),
        logger=logs.get_logger_for(AudioAnalyzer),
    )


__all__ = ["build_audio_handler"]