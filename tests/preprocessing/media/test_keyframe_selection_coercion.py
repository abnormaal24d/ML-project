"""Keyframe metadata coercion must fall back after invalid values."""

from __future__ import annotations

from preprocessing.media.video.video_keyframe_selector import (
    select_keyframe_metadata,
)
from preprocessing.media.video.video_preprocessor import (
    _video_alignment_summary,
)


def test_invalid_string_frame_index_falls_back_to_enumeration() -> None:
    selected = select_keyframe_metadata(
        sampled_frames=[
            {
                "frame_index": "bad",
                "timestamp_seconds": 1.0,
                "scene_change_score": 0.9,
            }
        ],
        max_keyframes=4,
    )
    assert len(selected) == 1
    assert selected[0]["frame_index"] == 0
    assert selected[0]["timestamp_seconds"] == 1.0


def test_boolean_values_are_not_coerced_to_numeric() -> None:
    selected = select_keyframe_metadata(
        sampled_frames=[
            {
                "frame_index": True,
                "timestamp_seconds": False,
                "scene_change_score": 0.9,
            }
        ],
        max_keyframes=4,
    )
    assert len(selected) == 1
    assert selected[0]["frame_index"] == 0
    assert selected[0]["timestamp_seconds"] == 0.0


def test_mixed_valid_and_invalid_timestamps_sort_without_error() -> None:
    selected = select_keyframe_metadata(
        sampled_frames=[
            {
                "frame_index": 2,
                "timestamp_seconds": "bad",
                "scene_change_score": 0.9,
            },
            {
                "frame_index": 1,
                "timestamp_seconds": 0.5,
                "scene_change_score": 0.9,
            },
            {
                "frame_index": 0,
                "timestamp_seconds": None,
                "scene_change_score": 0.9,
            },
        ],
        max_keyframes=8,
    )
    assert len(selected) >= 1
    timestamps = [frame["timestamp_seconds"] for frame in selected]
    assert all(isinstance(value, float) for value in timestamps)
    assert timestamps == sorted(timestamps)


def test_zero_frame_index_and_timestamp_are_preserved() -> None:
    selected = select_keyframe_metadata(
        sampled_frames=[
            {
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "scene_change_score": 0.95,
            }
        ],
        max_keyframes=4,
    )
    assert len(selected) == 1
    assert selected[0]["frame_index"] == 0
    assert selected[0]["timestamp_seconds"] == 0.0


def test_video_alignment_summary_preserves_keyframe_scene_and_track_evidence() -> (
    None
):
    summary = _video_alignment_summary(
        payload={
            "keyframes": [
                {"frame_path": "frames/0.jpg", "timestamp_seconds": 1.0},
                {"frame_path": "frames/1.jpg", "timestamp_seconds": 9.0},
            ],
            "scene_boundaries": [2.0, 7.5],
            "subtitle_segments": [{"start_seconds": 1.0, "text": "hello"}],
            "object_tracks": [{"track_id": "person-1"}],
            "privacy_intervals": [{"start_seconds": 4.0, "end_seconds": 5.0}],
        },
        duration_seconds=10.0,
    )

    assert summary["keyframe_count"] == 2
    assert summary["timed_keyframe_count"] == 2
    assert summary["keyframe_timeline_ratio"] == 0.8
    assert summary["scene_boundary_count"] == 2
    assert summary["shot_boundaries_available"] is True
    assert summary["subtitle_alignment_available"] is True
    assert summary["object_tracks_available"] is True
    assert summary["privacy_intervals_available"] is True
