from __future__ import annotations

from types import SimpleNamespace

from datachecker.validation.preprocessing_artifact_validator import (
    PreprocessingArtifactValidator,
)


def test_video_readiness_uses_only_canonical_trainable_count() -> None:
    coverage = {
        "curated_accepted": 4,
        "accepted_with_keyframes": 4,
    }
    assert (
        PreprocessingArtifactValidator._video_readiness_ratio(
            coverage=coverage,
            denominator_key="curated_accepted",
        )
        == 0.0
    )
    coverage["trainable"] = 3
    assert (
        PreprocessingArtifactValidator._video_readiness_ratio(
            coverage=coverage,
            denominator_key="curated_accepted",
        )
        == 0.75
    )


def test_transcript_coverage_is_weighted_across_audio_and_video() -> None:
    manifest = SimpleNamespace(
        audio_coverage={
            "curated_accepted": 100,
            "accepted_with_transcript": 0,
        },
        video_coverage={
            "curated_accepted": 1,
            "accepted_with_transcript": 1,
            "trainable": 0,
        },
        image_coverage={},
    )

    result = (
        PreprocessingArtifactValidator._validate_multimodal_coverage_checks(
            manifest=manifest,
            minimum_transcript_coverage=0.5,
            minimum_ocr_coverage=0.0,
            minimum_keyframe_coverage=0.0,
        )
    )

    assert result is not None
    assert result.is_valid is False
    assert "transcript coverage" in result.details[0]


def test_transcript_coverage_rejects_impossible_counts() -> None:
    manifest = SimpleNamespace(
        audio_coverage={
            "curated_accepted": 1,
            "accepted_with_transcript": 2,
        },
        video_coverage={
            "curated_accepted": 0,
            "accepted_with_transcript": 0,
            "trainable": 0,
        },
        image_coverage={},
    )

    result = (
        PreprocessingArtifactValidator._validate_multimodal_coverage_checks(
            manifest=manifest,
            minimum_transcript_coverage=0.0,
            minimum_ocr_coverage=0.0,
            minimum_keyframe_coverage=0.0,
        )
    )

    assert result is not None
    assert result.is_valid is False
    assert "impossible accepted counts" in result.details[0]
