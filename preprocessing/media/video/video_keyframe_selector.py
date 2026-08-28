"""Video keyframe selection from sampled frame metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from preprocessing.media.media_input_validation import (
    as_optional_float,
    as_optional_int,
    as_optional_text,
)


@dataclass(frozen=True, slots=True)
class SelectedKeyframe:
    frame_index: int
    timestamp_seconds: float
    frame_path: str | None = None
    scene_change_score: float | None = None
    motion_score: float | None = None
    blur_score: float | None = None
    ocr_density: float | None = None
    dedupe_hash: str | None = None
    reason: str | None = None


def select_keyframes(
    *,
    frames: list[dict[str, Any]],
    max_keyframes: int = 8,
    min_scene_change: float = 0.3,
) -> list[SelectedKeyframe]:
    """Select diverse, informative keyframes from scored frame metadata."""

    if not frames:
        return []

    selected: list[SelectedKeyframe] = []
    prev_hash = None
    for frame in sorted(
        frames, key=lambda item: item.get("timestamp_seconds", 0)
    ):
        timestamp = frame.get("timestamp_seconds", 0.0)
        index = frame.get("frame_index", 0)
        frame_path = frame.get("frame_path")
        scene = frame.get("scene_change_score", 0.0)
        motion = frame.get("motion_score", 0.0)
        blur = frame.get("blur_score", 1.0)
        ocr_density = frame.get("ocr_density", 0.0)
        dedupe_hash = frame.get("dedupe_hash")

        reason = None
        if scene > min_scene_change:
            reason = "scene_change"
        elif motion > 0.5 and len(selected) < max_keyframes // 2:
            reason = "diverse_motion"
        elif ocr_density > 0.2:
            reason = "high_ocr_value"
        elif blur < 0.3 and len(selected) < max_keyframes:
            reason = "low_blur"

        if reason and (dedupe_hash != prev_hash or prev_hash is None):
            selected.append(
                SelectedKeyframe(
                    frame_index=index,
                    timestamp_seconds=timestamp,
                    frame_path=frame_path,
                    scene_change_score=scene,
                    motion_score=motion,
                    blur_score=blur,
                    ocr_density=ocr_density,
                    dedupe_hash=dedupe_hash,
                    reason=reason,
                )
            )
            prev_hash = dedupe_hash

        if len(selected) >= max_keyframes:
            break

    return selected


def select_keyframe_metadata(
    *,
    sampled_frames: list[dict[str, Any]],
    max_keyframes: int = 12,
) -> tuple[dict[str, Any], ...]:
    """Normalize sampled-frame dicts and select keyframe metadata records."""

    if not sampled_frames:
        return ()

    frames: list[dict[str, Any]] = []
    for index, item in enumerate(sampled_frames):
        # Coerce first, then fall back so invalid values never become None.
        frame_index = as_optional_int(item.get("frame_index"))
        if frame_index is None:
            frame_index = index
        timestamp = as_optional_float(item.get("timestamp_seconds"))
        if timestamp is None:
            timestamp = float(index)
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "frame_path": as_optional_text(item.get("frame_path")),
                "scene_change_score": as_optional_float(
                    item.get("scene_change_score")
                )
                or 0.0,
                "motion_score": as_optional_float(item.get("motion_score"))
                or 0.0,
                "blur_score": as_optional_float(item.get("blur_score")) or 0.0,
                "ocr_density": as_optional_float(item.get("ocr_density"))
                or 0.0,
                "dedupe_hash": as_optional_text(item.get("dedupe_hash")),
            }
        )

    selected = select_keyframes(
        frames=frames,
        max_keyframes=max_keyframes,
    )
    return tuple(
        {
            "frame_index": keyframe.frame_index,
            "timestamp_seconds": keyframe.timestamp_seconds,
            "frame_path": keyframe.frame_path,
            "source": "selected_keyframe",
            "selection_reason": keyframe.reason,
            "scene_change_score": keyframe.scene_change_score,
            "motion_score": keyframe.motion_score,
            "ocr_density": keyframe.ocr_density,
        }
        for keyframe in selected
    )
