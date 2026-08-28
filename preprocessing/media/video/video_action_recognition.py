"""Evidence-based motion analysis for video semantic outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .video_scene_analysis import (
    append_reason,
    frame_differences,
    sample_video_frames,
)

if TYPE_CHECKING:
    from preprocessing.media.ports import FrameProcessor, VideoReader


def _motion_label(score: float) -> str:
    if score >= 0.35:
        return "high_motion"
    if score >= 0.10:
        return "medium_motion"
    return "low_or_static"


def _motion_segments(
    differences: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    if not differences:
        return []

    segments: list[dict[str, Any]] = []
    for index, (timestamp_seconds, raw_score) in enumerate(differences):
        score = max(0.0, min(1.0, float(raw_score) / 255.0))
        next_timestamp = (
            differences[index + 1][0]
            if index + 1 < len(differences)
            else timestamp_seconds + 0.5
        )
        end_seconds = max(float(timestamp_seconds) + 0.001, next_timestamp)
        segments.append(
            {
                "label": _motion_label(score),
                "start_seconds": round(float(timestamp_seconds), 3),
                "end_seconds": round(float(end_seconds), 3),
                "confidence": round(min(1.0, 0.5 + score / 2.0), 3),
                "motion_score": round(score, 4),
                "evidence_source": "measured_frame_difference",
            }
        )
    return segments


def _text_action_hints(
    *,
    transcript_text: str | None,
    frame_ocr_text: str | None,
) -> list[dict[str, Any]]:
    """Return text observations without promoting them to visual evidence."""

    sources = (
        ("transcript", transcript_text or ""),
        ("ocr", frame_ocr_text or ""),
    )
    rules = (
        ("speech_or_presentation", ("speaking", "speech", "talk")),
        ("walking_or_running", ("walking", "running", "run ", "walk ")),
        ("vehicle_or_flight_motion", ("driving", "flying", "launch")),
        ("demonstration_or_instruction", ("showing", "tutorial")),
        ("sports_or_exercise", ("playing", "exercise", "workout")),
    )
    hints: list[dict[str, Any]] = []
    for source, text in sources:
        normalized = text.lower()
        if not normalized:
            continue
        for label, keywords in rules:
            if any(keyword in normalized for keyword in keywords):
                hints.append(
                    {
                        "label": label,
                        "source": source,
                        "confidence": 0.4,
                        "evidence_semantics": "text_observation",
                    }
                )
    return hints[:4]


def recognize_actions(
    *,
    video_path: str | None = None,
    keyframes: list[dict[str, Any]] | None = None,
    clip_duration: float | None = None,
    probe: dict[str, Any] | None = None,
    transcript_text: str | None = None,
    frame_ocr_text: str | None = None,
    video_reader: VideoReader,
    frame_processor: FrameProcessor,
) -> dict[str, Any]:
    """Measure motion and report action taxonomy as unavailable.

    Motion segments are emitted only when actual decoded frames can be
    compared. Text can provide contextual hints, but never fabricates visual
    segments or an action-classification result.
    """

    del clip_duration, probe
    analysis_reasons: list[str] = []
    frames = sample_video_frames(
        video_path=video_path,
        analysis_reasons=analysis_reasons,
        video_reader=video_reader,
    )
    if not frames and keyframes:
        frames = [
            frame for frame in keyframes if frame.get("frame") is not None
        ]
        if frames:
            append_reason(analysis_reasons, "using_decoded_keyframe_evidence")

    segments = _motion_segments(
        frame_differences(
            frames=frames,
            frame_processor=frame_processor,
        )
    )
    if not segments:
        append_reason(analysis_reasons, "motion_evidence_unavailable")
    append_reason(analysis_reasons, "action_taxonomy_model_unavailable")

    return {
        "action_label": None,
        "action_segments": [],
        "motion_segments": segments,
        "status": "motion_analysis_available" if segments else "unavailable",
        "backend": "frame_difference" if segments else None,
        "vision_backend": "opencv" if segments else None,
        "vision_backend_status": "motion_ok" if segments else "not_run",
        "alternative_vision_backend_available": False,
        "recognition_level": "motion_only" if segments else "none",
        "action_recognition_available": False,
        "action_recognition_status": "taxonomy_model_unavailable",
        "taxonomy_model_available": False,
        "action_taxonomy_status": "unavailable",
        "taxonomy_classification_status": "not_run",
        "analysis_status": "passed" if segments else "unavailable",
        "analysis_reasons": tuple(sorted(set(analysis_reasons))),
        "text_action_hints": _text_action_hints(
            transcript_text=transcript_text,
            frame_ocr_text=frame_ocr_text,
        ),
    }
