"""Optional audio extraction, diarization, and normalization for video."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.collection.processors import VideoProcessorSettings
from logger.project_logger import ProjectLogger

_ANALYSIS_SOFT_ERRORS = (OSError, RuntimeError, ValueError)

if TYPE_CHECKING:
    from preprocessing.media.ports import (
        VideoAudioTrackResult,
        VideoNormalizationResult,
    )
    from preprocessing.media.speech.speaker_diarizer import (
        SpeakerDiarizer,
    )
    from preprocessing.media.speech.transcription_result import (
        TranscriptionResult,
    )


class VideoOptionalAnalysisService:
    def __init__(
        self,
        *,
        settings: VideoProcessorSettings,
        diarization_service: SpeakerDiarizer,
        audio_track_extractor: Callable[..., VideoAudioTrackResult],
        video_normalizer: Callable[..., VideoNormalizationResult],
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._diarization_service = diarization_service
        self._audio_track_extractor = audio_track_extractor
        self._video_normalizer = video_normalizer
        self._logger = logger

    async def run(
        self,
        *,
        analysis_path: Path,
        metadata: dict[str, Any],
        run_transcription: bool,
        transcription: TranscriptionResult | None,
    ) -> tuple[
        Any | None, str | None, Any | None, str | None, Any | None, str | None
    ]:
        audio_track_result: Any | None = None
        audio_track_error_type: str | None = None
        if self._settings.extract_audio_track:
            (
                audio_track_result,
                audio_track_error_type,
            ) = await asyncio.to_thread(
                self._extract_audio_track,
                analysis_path=analysis_path,
            )

        speaker_diarization: Any | None = None
        speaker_diarization_error_type: str | None = None
        if run_transcription and self._settings.extract_audio_track:
            scratch_audio_path = (
                audio_track_result.audio_path if audio_track_result else None
            )
            try:
                (
                    speaker_diarization,
                    speaker_diarization_error_type,
                ) = await asyncio.to_thread(
                    self._run_speaker_diarization,
                    analysis_path=analysis_path,
                    transcription=transcription,
                    audio_track_result=audio_track_result,
                )
            finally:
                # The extracted WAV is scratch for diarization only; it must
                # disappear immediately after consumption.
                self._discard_scratch_audio(scratch_audio_path)

        normalization_result: Any | None = None
        normalization_error_type: str | None = None
        if self._settings.normalize_video:
            (
                normalization_result,
                normalization_error_type,
            ) = await asyncio.to_thread(
                self._normalize_video,
                analysis_path=analysis_path,
                metadata=metadata,
            )
        return (
            audio_track_result,
            audio_track_error_type,
            speaker_diarization,
            speaker_diarization_error_type,
            normalization_result,
            normalization_error_type,
        )

    def _extract_audio_track(
        self, *, analysis_path: Path
    ) -> tuple[Any | None, str | None]:
        try:
            return (
                self._audio_track_extractor(video_path=str(analysis_path)),
                None,
            )
        except _ANALYSIS_SOFT_ERRORS as exc:
            return None, type(exc).__name__

    def _run_speaker_diarization(
        self,
        *,
        analysis_path: Path,
        transcription: TranscriptionResult | None,
        audio_track_result: Any | None,
    ) -> tuple[Any | None, str | None]:
        audio_path = (
            audio_track_result.audio_path if audio_track_result else None
        )
        sample_rate = (
            audio_track_result.sample_rate
            if audio_track_result is not None
            else None
        )
        try:
            transcript_segments = (
                None
                if transcription is None
                else [asdict(segment) for segment in transcription.segments]
            )
            return (
                self._diarization_service.diarize(
                    audio_bytes=b"",
                    audio_path=audio_path or str(analysis_path),
                    sample_rate=sample_rate,
                    transcript_segments=transcript_segments,
                ),
                None,
            )
        except _ANALYSIS_SOFT_ERRORS as exc:
            return None, type(exc).__name__

    def _discard_scratch_audio(self, audio_path: object | None) -> None:
        """Best-effort removal of the temporary extracted audio track."""
        if not isinstance(audio_path, str) or not audio_path.strip():
            return
        path = Path(audio_path)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            self._logger.warning(
                "video_scratch_audio_cleanup_failed",
                path=str(path),
                error_type=type(exc).__name__,
            )

    def _normalize_video(
        self,
        *,
        analysis_path: Path,
        metadata: dict[str, Any],
    ) -> tuple[Any | None, str | None]:
        try:
            return (
                self._video_normalizer(
                    input_path=str(analysis_path),
                    target_fps=float(
                        metadata.get("fps") or self._settings.fallback_fps
                    ),
                ),
                None,
            )
        except _ANALYSIS_SOFT_ERRORS as exc:
            return None, type(exc).__name__
