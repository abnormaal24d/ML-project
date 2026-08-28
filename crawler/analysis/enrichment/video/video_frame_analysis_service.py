"""Frame sampling, OCR, transcription, and semantic video analysis."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from logger.project_logger import ProjectLogger
from preprocessing.media.ports import FrameProcessor, VideoReader
from preprocessing.media.video.video_semantic_outputs import (
    build_video_semantic_outputs,
)

_ANALYSIS_SOFT_ERRORS = (OSError, RuntimeError, ValueError)
_ResultT = TypeVar("_ResultT")

if TYPE_CHECKING:
    from crawler.analysis.enrichment.video.video_frame_sampler import (
        VideoFrameSampler,
    )
    from preprocessing.media.ocr.ocr_result import (
        OpticalCharacterRecognitionResult,
    )
    from preprocessing.media.speech.speech_transcriber import (
        SpeechTranscriber,
    )
    from preprocessing.media.speech.transcription_result import (
        TranscriptionResult,
    )
    from preprocessing.media.video.video_frame_ocr import (
        VideoFrameTextExtractionService,
    )


class VideoFrameAnalysisService:
    """Own frame-oriented video analysis independently from probe orchestration."""

    def __init__(
        self,
        *,
        frame_sampler: VideoFrameSampler,
        frame_text_extraction_service: VideoFrameTextExtractionService,
        transcription_executor: SpeechTranscriber,
        keyframe_selector: Callable[..., tuple[dict[str, Any], ...]],
        video_reader: VideoReader,
        frame_processor: FrameProcessor,
        logger: ProjectLogger,
    ) -> None:
        self._frame_sampler = frame_sampler
        self._frame_text_extraction_service = frame_text_extraction_service
        self._transcription_executor = transcription_executor
        self._keyframe_selector = keyframe_selector
        self._video_reader = video_reader
        self._frame_processor = frame_processor
        self._logger = logger

    def normalize_limits(
        self,
        *,
        max_sampled_frames: int,
        max_duration_seconds: float,
    ) -> tuple[int, float]:
        return (
            max(0, max_sampled_frames),
            max(0.0, max_duration_seconds),
        )

    async def analyze_frames(
        self,
        *,
        analysis_path: Path,
        metadata: dict[str, Any],
        run_transcription: bool,
        run_ocr: bool,
        extract_keyframes: bool,
        max_sampled_frames: int,
        max_duration_seconds: float,
    ) -> tuple[
        tuple[dict[str, Any], ...],
        OpticalCharacterRecognitionResult | None,
        TranscriptionResult | None,
        tuple[dict[str, Any], ...],
        dict[str, Any],
        dict[str, Any],
    ]:
        sampled_frames: list[dict[str, Any]] = []
        keyframes: tuple[dict[str, Any], ...] = ()
        if (extract_keyframes or run_ocr) and max_sampled_frames > 0:
            self._logger.debug(
                "video_frame_sampling_start",
                path=str(analysis_path),
                max=max_sampled_frames,
            )
            sampled_frames = await asyncio.to_thread(
                self._frame_sampler.sample,
                path=analysis_path,
                max_frames=max_sampled_frames,
            )
            self._logger.debug(
                "video_frame_sampling_done",
                num_frames=len(sampled_frames),
            )
            if extract_keyframes:
                keyframes = await asyncio.to_thread(
                    self._keyframe_selector,
                    sampled_frames=sampled_frames,
                )

        ocr_task = (
            asyncio.to_thread(
                self._frame_text_extraction_service.extract_if_allowed,
                sampled_frames=sampled_frames,
                run_ocr=True,
                duration_seconds=metadata.get("duration_seconds"),
                max_duration_seconds=max_duration_seconds,
            )
            if run_ocr
            else None
        )
        transcription_task = (
            self._transcription_executor.transcribe_if_allowed(
                media_path=analysis_path,
                run_transcription=True,
                duration_seconds=metadata.get("duration_seconds"),
                max_duration_seconds=max_duration_seconds,
            )
            if run_transcription
            else None
        )
        ocr_result: (
            OpticalCharacterRecognitionResult | BaseException | None
        ) = None
        transcription_result: TranscriptionResult | BaseException | None = None
        if ocr_task is not None and transcription_task is not None:
            ocr_result, transcription_result = await asyncio.gather(
                ocr_task,
                transcription_task,
                return_exceptions=True,
            )
        elif ocr_task is not None:
            ocr_result = (
                await asyncio.gather(ocr_task, return_exceptions=True)
            )[0]
        elif transcription_task is not None:
            transcription_result = (
                await asyncio.gather(
                    transcription_task, return_exceptions=True
                )
            )[0]

        ocr = self._optional_result(label="frame_ocr", result=ocr_result)
        transcription = self._optional_result(
            label="transcription",
            result=transcription_result,
        )
        frame_ocr_results, scene_graph, action_result = (
            build_video_semantic_outputs(
                analysis_path=analysis_path,
                metadata=metadata,
                keyframes=keyframes,
                ocr=ocr,
                transcription=transcription,
                video_reader=self._video_reader,
                frame_processor=self._frame_processor,
            )
        )
        self._discard_unselected_sampled_frames(
            sampled_frames=sampled_frames,
            keyframes=keyframes,
        )
        return (
            keyframes,
            ocr,
            transcription,
            frame_ocr_results,
            scene_graph,
            action_result,
        )

    def _discard_unselected_sampled_frames(
        self,
        *,
        sampled_frames: list[dict[str, Any]],
        keyframes: tuple[dict[str, Any], ...],
    ) -> None:
        """Remove sampled frame scratch not selected as a persistent keyframe.

        Only the selected keyframes are relocated into the dataset
        transaction; every other sampled frame is scratch and must disappear
        right after frame analysis consumed it.
        """

        selected_paths = {
            Path(str(frame.get("frame_path"))).resolve()
            for frame in keyframes
            if isinstance(frame, dict)
            and isinstance(frame.get("frame_path"), str)
            and str(frame.get("frame_path")).strip()
        }
        for frame in sampled_frames:
            raw_path = frame.get("frame_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            path = Path(raw_path)
            if path.resolve() in selected_paths:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                self._logger.warning(
                    "video_sampled_frame_scratch_cleanup_failed",
                    path=str(path),
                    error_type=type(exc).__name__,
                )

    def _optional_result(
        self,
        *,
        label: str,
        result: _ResultT | BaseException | None,
    ) -> _ResultT | None:
        if result is None:
            return None
        if isinstance(result, _ANALYSIS_SOFT_ERRORS):
            self._logger.warning(
                f"{label}_analysis_failed",
                error_type=type(result).__name__,
            )
            return None
        if isinstance(result, BaseException):
            raise result
        return result
