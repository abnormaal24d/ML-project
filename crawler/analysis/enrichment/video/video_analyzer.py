"""Video analysis orchestration for fetched crawler media."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from config.collection.processors import VideoProcessorSettings
from crawler.analysis.enrichment.video.video_analysis_result import (
    VideoAnalysisResult,
)
from crawler.analysis.enrichment.video.video_embed_metadata import (
    embed_metadata,
)
from crawler.analysis.enrichment.video.video_frame_analysis_service import (
    VideoFrameAnalysisService,
)
from crawler.analysis.enrichment.video.video_metadata_detection import (
    is_embed_video_metadata,
    is_head_only_video_probe,
)
from crawler.analysis.enrichment.video.video_optional_analysis_service import (
    VideoOptionalAnalysisService,
)
from crawler.analysis.enrichment.video.video_probe_resolver import (
    VideoProbeResolver,
)
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.analysis.enrichment.media_files.media_payload_path_resolver import (
        MediaPayloadPathResolver,
    )
    from crawler.fetching.results.result import FetchResult


class VideoAnalyzer:
    """Coordinate probe, frame, optional analysis, timeout, and cleanup."""

    def __init__(
        self,
        *,
        settings: VideoProcessorSettings,
        media_file_resolver: MediaPayloadPathResolver,
        probe_resolver: VideoProbeResolver,
        frame_analysis: VideoFrameAnalysisService,
        optional_analysis: VideoOptionalAnalysisService,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._media_file_resolver = media_file_resolver
        self._probe_resolver = probe_resolver
        self._frame_analysis = frame_analysis
        self._optional_analysis = optional_analysis
        self._logger = logger

    async def analyze(self, *, result: FetchResult) -> VideoAnalysisResult:
        if is_embed_video_metadata(result=result):
            return VideoAnalysisResult(
                metadata=embed_metadata(result=result),
                metadata_status="embed_metadata",
            )
        if is_head_only_video_probe(result=result):
            return VideoAnalysisResult(
                metadata={},
                metadata_status="head_only_oversized",
            )

        path = await self._media_file_resolver.resolve_path(
            result=result,
            suffix=".video",
        )
        try:
            return await self._analyze_resolved_video(
                result=result,
                path=path,
            )
        finally:
            self._media_file_resolver.cleanup_owned_path(path)

    async def _analyze_resolved_video(
        self,
        *,
        result: FetchResult,
        path: Path,
    ) -> VideoAnalysisResult:
        cleanup_path: Path | None = None
        max_frames, max_duration = self._frame_analysis.normalize_limits(
            max_sampled_frames=self._settings.keyframes.max_keyframes,
            max_duration_seconds=self._settings.max_duration_seconds,
        )
        try:
            probe_result = await self._probe_resolver.resolve_probe(
                result=result,
                path=path,
            )
            metadata = (
                probe_result.metadata
                if isinstance(probe_result.metadata, dict)
                else {}
            )
            analysis_path = probe_result.analysis_path
            cleanup_path = probe_result.cleanup_path
            self._logger.debug(
                "video_probe_metadata_normalized",
                status=probe_result.metadata_status,
            )
            (
                keyframes,
                ocr,
                transcription,
                frame_ocr_results,
                scene_graph,
                action_result,
            ) = await self._frame_analysis.analyze_frames(
                analysis_path=analysis_path,
                metadata=metadata,
                run_transcription=self._settings.run_transcription,
                run_ocr=self._settings.run_ocr,
                extract_keyframes=self._settings.keyframes.enabled,
                max_sampled_frames=max_frames,
                max_duration_seconds=max_duration,
            )
            (
                audio_track_result,
                audio_track_error_type,
                speaker_diarization,
                speaker_diarization_error_type,
                normalization_result,
                normalization_error_type,
            ) = await self._optional_analysis.run(
                analysis_path=analysis_path,
                metadata=metadata,
                run_transcription=self._settings.run_transcription,
                transcription=transcription,
            )
            return VideoAnalysisResult(
                payload_path=str(path),
                metadata=metadata,
                metadata_status=probe_result.metadata_status,
                transcription=transcription,
                keyframes=keyframes,
                frame_ocr=ocr,
                frame_ocr_results=frame_ocr_results,
                scene_graph=scene_graph,
                action_result=action_result,
                probe_result=probe_result,
                audio_track_result=audio_track_result,
                audio_track_error_type=audio_track_error_type,
                speaker_diarization=speaker_diarization,
                speaker_diarization_error_type=speaker_diarization_error_type,
                normalization_result=normalization_result,
                normalization_error_type=normalization_error_type,
            )
        finally:
            if cleanup_path is not None:
                try:
                    cleanup_path.unlink(missing_ok=True)
                except OSError as exc:
                    self._logger.warning(
                        "video_cleanup_failed",
                        path=str(cleanup_path),
                        error_type=type(exc).__name__,
                    )
