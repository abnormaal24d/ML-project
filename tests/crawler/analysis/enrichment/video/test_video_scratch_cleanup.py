"""Video analysis scratch is removed immediately after consumption.

Unselected sampled frames disappear after frame analysis; the temporary
extracted audio track disappears after speaker diarization consumed it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from crawler.analysis.enrichment.video.video_frame_analysis_service import (
    VideoFrameAnalysisService,
)
from crawler.analysis.enrichment.video.video_optional_analysis_service import (
    VideoOptionalAnalysisService,
)
from preprocessing.media.ports import VideoAudioTrackResult


class _Logger:
    def debug(self, *args: object, **kwargs: object) -> None:
        return None

    def info(self, *args: object, **kwargs: object) -> None:
        return None

    def warning(self, *args: object, **kwargs: object) -> None:
        return None

    def error(self, *args: object, **kwargs: object) -> None:
        return None


def _write_frames(tmp_path: Path, *names: str) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for index, name in enumerate(names):
        path = tmp_path / name
        path.write_bytes(b"jpeg")
        frames.append(
            {
                "frame_index": index,
                "timestamp_seconds": float(index),
                "frame_path": str(path),
                "scene_change_score": 0.1,
                "motion_score": 0.1,
                "blur_score": 0.9,
                "ocr_density": 0.0,
            }
        )
    return frames


def test_frame_analysis_keeps_selected_and_discards_unselected_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selected_path = tmp_path / "frame_selected.jpg"
    selected_path.write_bytes(b"jpeg")
    sampled = _write_frames(tmp_path, "frame_1.jpg", "frame_2.jpg")
    sampled.append(
        {
            "frame_index": 2,
            "timestamp_seconds": 2.0,
            "frame_path": str(selected_path),
            "scene_change_score": 0.9,
            "motion_score": 0.0,
            "blur_score": 0.8,
            "ocr_density": 0.0,
        }
    )

    def keyframe_selector(
        sampled_frames: list[dict[str, object]],
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            dict(frame)
            for frame in sampled_frames
            if frame.get("scene_change_score", 0.0) > 0.3
        )

    service = VideoFrameAnalysisService(
        frame_sampler=SimpleNamespace(
            sample=lambda **kwargs: [dict(item) for item in sampled]
        ),
        frame_text_extraction_service=SimpleNamespace(
            extract_if_allowed=lambda **kwargs: None
        ),
        transcription_executor=SimpleNamespace(
            transcribe_if_allowed=lambda **kwargs: None
        ),
        keyframe_selector=keyframe_selector,
        video_reader=SimpleNamespace(),
        frame_processor=SimpleNamespace(),
        logger=_Logger(),
    )

    async def run() -> None:
        monkeypatch.setattr(
            "crawler.analysis.enrichment.video.video_frame_analysis_service"
            ".build_video_semantic_outputs",
            lambda **kwargs: ((), {}, {}),
        )
        await service.analyze_frames(
            analysis_path=tmp_path / "clip.mp4",
            metadata={},
            run_transcription=False,
            run_ocr=False,
            extract_keyframes=True,
            max_sampled_frames=8,
            max_duration_seconds=0.0,
        )

    asyncio.run(run())

    assert selected_path.exists()
    assert not (tmp_path / "frame_1.jpg").exists()
    assert not (tmp_path / "frame_2.jpg").exists()


def test_optional_analysis_discards_scratch_audio_after_diarization(
    tmp_path: Path,
) -> None:
    scratch_wav = tmp_path / "clip_audio_16000.wav"
    scratch_wav.write_bytes(b"wav")
    diarized_audio_paths: list[str] = []

    settings = SimpleNamespace(
        extract_audio_track=True,
        normalize_video=False,
        fallback_fps=25.0,
    )
    diarization_service = SimpleNamespace()
    diarization_service.diarize = lambda **kwargs: (
        diarized_audio_paths.append(str(kwargs.get("audio_path")))
        or SimpleNamespace(segments=(), speaker_count=0)
    )
    service = VideoOptionalAnalysisService(
        settings=settings,  # type: ignore[arg-type]
        diarization_service=diarization_service,  # type: ignore[arg-type]
        audio_track_extractor=lambda video_path: VideoAudioTrackResult(
            extracted=True,
            audio_path=str(scratch_wav),
            sample_rate=16000,
            channels=1,
            duration_seconds=2.5,
        ),
        video_normalizer=None,
        logger=_Logger(),
    )

    async def run() -> None:
        await service.run(
            analysis_path=tmp_path / "clip.mp4",
            metadata={},
            run_transcription=True,
            transcription=None,
        )

    asyncio.run(run())

    assert diarized_audio_paths == [str(scratch_wav)]
    assert not scratch_wav.exists()


def test_optional_analysis_discards_scratch_audio_on_diarization_error(
    tmp_path: Path,
) -> None:
    scratch_wav = tmp_path / "clip_audio_16000.wav"
    scratch_wav.write_bytes(b"wav")

    settings = SimpleNamespace(
        extract_audio_track=True,
        normalize_video=False,
        fallback_fps=25.0,
    )
    diarization_service = SimpleNamespace()
    diarization_service.diarize = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("diarization backend failed")
    )
    service = VideoOptionalAnalysisService(
        settings=settings,  # type: ignore[arg-type]
        diarization_service=diarization_service,  # type: ignore[arg-type]
        audio_track_extractor=lambda video_path: VideoAudioTrackResult(
            extracted=True,
            audio_path=str(scratch_wav),
            sample_rate=16000,
            channels=1,
            duration_seconds=2.5,
        ),
        video_normalizer=None,
        logger=_Logger(),
    )

    async def run() -> None:
        await service.run(
            analysis_path=tmp_path / "clip.mp4",
            metadata={},
            run_transcription=True,
            transcription=None,
        )

    asyncio.run(run())

    assert not scratch_wav.exists()
