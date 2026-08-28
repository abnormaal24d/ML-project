"""Semantic output builders for video analysis.

Split across modules for maintainability (per guardrail remediation):
- video_ocr.py: OCR frame result normalization
- video_scene_analysis.py: scene graph / visual proxy construction
- video_action_recognition.py: motion/action proxy recognition
- This module now provides the thin composition entrypoint only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .video_action_recognition import recognize_actions
from .video_ocr import _json_safe_dict, build_frame_ocr_results
from .video_scene_analysis import analyze_video_scenes

if TYPE_CHECKING:
    from preprocessing.media.ports import FrameProcessor, VideoReader


def _dict_tuple(value: object) -> tuple[dict[str, Any], ...]:
    """Lightweight tuple normalizer kept here for composition glue."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        item_payload
        for item in value
        if (item_payload := _json_safe_dict(item))
    )


def build_video_semantic_outputs(
    *,
    analysis_path: Path,
    metadata: dict[str, Any],
    keyframes: tuple[dict[str, Any], ...],
    ocr: Any | None,
    transcription: Any | None,
    video_reader: VideoReader,
    frame_processor: FrameProcessor,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any]]:
    """Compose OCR + scene + action outputs (delegates to split modules)."""
    frame_ocr_results = build_frame_ocr_results(ocr=ocr)
    transcript_text = (
        getattr(transcription, "text", None) if transcription else None
    )
    transcript_segments = _dict_tuple(
        getattr(transcription, "segments", ()) if transcription else ()
    )
    scene_graph = analyze_video_scenes(
        video_path=str(analysis_path),
        keyframes=list(keyframes or []),
        transcript_segments=list(transcript_segments),
        frame_ocr_results=list(frame_ocr_results or []),
        probe_metadata=metadata,
        video_reader=video_reader,
        frame_processor=frame_processor,
    )
    action_result = recognize_actions(
        video_path=str(analysis_path),
        keyframes=list(keyframes or []),
        clip_duration=metadata.get("duration_seconds"),
        probe=metadata,
        transcript_text=transcript_text,
        frame_ocr_text=getattr(ocr, "text", None) if ocr else None,
        video_reader=video_reader,
        frame_processor=frame_processor,
    )
    return (
        frame_ocr_results,
        scene_graph,
        action_result,
    )
