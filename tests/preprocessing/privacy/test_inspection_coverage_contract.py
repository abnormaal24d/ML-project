"""Strict range/page coverage contract for privacy inspection evidence.

Complete inspection requires exact temporal coverage: zero expected
durations, gaps, negative starts, and zero-length spans must never be
reported as complete.
"""

from __future__ import annotations

import pytest

from preprocessing.privacy.inspection.evidence_location import (
    BoundingBox,
    TextSpan,
    TimeRange,
)
from preprocessing.privacy.inspection.inspection_coverage import (
    InspectionCoverage,
    ranges_cover_duration,
)


def _coverage(
    *,
    audio: tuple[tuple[int, int], ...] = (),
    audio_duration: int | None = None,
    video: tuple[tuple[int, int], ...] = (),
    video_duration: int | None = None,
    checked_fields: frozenset[str] = frozenset({"a"}),
    required_fields: frozenset[str] = frozenset({"a"}),
    checked_pages: frozenset[int] = frozenset(),
    expected_page_count: int | None = None,
    visual_analysis_completed: bool | None = None,
    detector_failures: tuple[str, ...] = (),
) -> InspectionCoverage:
    return InspectionCoverage(
        checked_fields=checked_fields,
        required_fields=required_fields,
        checked_audio_ranges_ms=audio,
        expected_audio_duration_ms=audio_duration,
        checked_video_ranges_ms=video,
        expected_video_duration_ms=video_duration,
        checked_pages=checked_pages,
        expected_page_count=expected_page_count,
        visual_analysis_completed=visual_analysis_completed,
        detector_failures=detector_failures,
    )


def test_zero_audio_duration_is_not_complete() -> None:
    coverage = _coverage(audio=(), audio_duration=0)
    assert not coverage.complete


def test_zero_video_duration_is_not_complete() -> None:
    coverage = _coverage(video=((0, 100),), video_duration=0)
    assert not coverage.complete


def test_gap_between_ranges_is_incomplete() -> None:
    coverage = _coverage(
        audio=((0, 500), (1_500, 2_000)),
        audio_duration=2_000,
    )
    assert not coverage.complete


def test_gap_before_first_range_is_incomplete() -> None:
    coverage = _coverage(audio=((1_000, 2_000),), audio_duration=2_000)
    assert not coverage.complete


def test_partial_overlap_with_gap_is_incomplete() -> None:
    coverage = _coverage(
        audio=((0, 1_000), (1_200, 2_000)),
        audio_duration=2_000,
    )
    assert not coverage.complete


def test_negative_start_range_is_incomplete() -> None:
    coverage = _coverage(audio=((-10, 1_000),), audio_duration=1_000)
    assert not coverage.complete


def test_zero_length_range_is_incomplete() -> None:
    coverage = _coverage(
        audio=((0, 0), (0, 1_000)),
        audio_duration=1_000,
    )
    assert not coverage.complete


def test_trailing_invalid_range_invalidates_full_coverage() -> None:
    assert not ranges_cover_duration(
        ranges=((0, 1_000), (1_000, 1_000)),
        duration_ms=1_000,
    )


def test_contiguous_ordered_cover_is_complete() -> None:
    coverage = _coverage(
        audio=((0, 1_000), (1_000, 2_000)),
        audio_duration=2_000,
    )
    assert coverage.complete


def test_overlapping_ordered_cover_is_complete() -> None:
    coverage = _coverage(
        audio=((0, 1_500), (500, 2_000)),
        audio_duration=2_000,
    )
    assert coverage.complete


def test_unsorted_ranges_are_sorted_before_covering() -> None:
    coverage = _coverage(
        audio=((1_000, 2_000), (0, 1_000)),
        audio_duration=2_000,
    )
    assert coverage.complete


def test_range_beyond_duration_is_clamped() -> None:
    coverage = _coverage(audio=((0, 3_000),), audio_duration=2_000)
    assert coverage.complete


def test_missing_pages_leave_document_incomplete() -> None:
    coverage = _coverage(
        checked_pages=frozenset({1}),
        expected_page_count=2,
    )
    assert not coverage.complete


def test_all_pages_present_is_complete() -> None:
    coverage = _coverage(
        checked_pages=frozenset({1, 2}),
        expected_page_count=2,
    )
    assert coverage.complete


def test_unchecked_required_field_is_incomplete() -> None:
    coverage = InspectionCoverage(
        checked_fields=frozenset({"a"}),
        required_fields=frozenset({"a", "b"}),
    )
    assert not coverage.complete
    assert coverage.unchecked_fields == frozenset({"b"})


def test_detector_failure_is_incomplete() -> None:
    coverage = _coverage(detector_failures=("decode_error",))
    assert not coverage.complete


def test_visual_analysis_explicitly_incomplete() -> None:
    coverage = _coverage(visual_analysis_completed=False)
    assert not coverage.complete


def test_audio_and_video_durations_must_both_cover() -> None:
    coverage = _coverage(
        audio=((0, 1_000),),
        audio_duration=1_000,
        video=((0, 500),),
        video_duration=1_000,
    )
    assert not coverage.complete


def test_pages_and_durations_fully_complete() -> None:
    coverage = _coverage(
        audio=((0, 1_000),),
        audio_duration=1_000,
        checked_pages=frozenset({1}),
        expected_page_count=1,
    )
    assert coverage.complete


def test_time_range_negative_start_raises() -> None:
    with pytest.raises(ValueError, match="0 <= start < end"):
        TimeRange(start_ms=-1, end_ms=100)


def test_time_range_zero_duration_raises() -> None:
    with pytest.raises(ValueError, match="0 <= start < end"):
        TimeRange(start_ms=500, end_ms=500)


def test_time_range_reversed_raises() -> None:
    with pytest.raises(ValueError, match="0 <= start < end"):
        TimeRange(start_ms=100, end_ms=50)


def test_text_span_negative_start_raises() -> None:
    with pytest.raises(ValueError, match="0 <= start < end"):
        TextSpan(start=-1, end=5)


def test_bounding_box_zero_dimension_raises() -> None:
    with pytest.raises(ValueError, match="positive dimensions"):
        BoundingBox(x=0, y=0, width=10, height=0)


def test_zero_expected_page_count_is_incomplete() -> None:
    coverage = _coverage(
        checked_pages=frozenset(),
        expected_page_count=0,
    )
    assert not coverage.complete


def test_negative_or_out_of_range_pages_are_incomplete() -> None:
    assert not _coverage(
        checked_pages=frozenset({-1, 1}),
        expected_page_count=1,
    ).complete
    assert not _coverage(
        checked_pages=frozenset({1, 2}),
        expected_page_count=1,
    ).complete


def test_uncertainty_flags_make_coverage_incomplete() -> None:
    coverage = InspectionCoverage(
        checked_fields=frozenset({"a"}),
        required_fields=frozenset({"a"}),
        visual_analysis_completed=True,
        uncertainty_flags=("visual_blur_outside_validated_range",),
    )
    assert not coverage.complete


def test_multiple_uncertainty_flags_make_coverage_incomplete() -> None:
    coverage = InspectionCoverage(
        checked_fields=frozenset({"a"}),
        required_fields=frozenset({"a"}),
        visual_analysis_completed=True,
        uncertainty_flags=(
            "visual_blur_outside_validated_range",
            "ocr_pii_location_unavailable",
        ),
    )
    assert not coverage.complete


def test_empty_uncertainty_flags_does_not_affect_complete() -> None:
    coverage = InspectionCoverage(
        checked_fields=frozenset({"a"}),
        required_fields=frozenset({"a"}),
        visual_analysis_completed=True,
        uncertainty_flags=(),
    )
    assert coverage.complete


def test_uncertainty_flags_with_detector_failure() -> None:
    coverage = InspectionCoverage(
        checked_fields=frozenset({"a"}),
        required_fields=frozenset({"a"}),
        visual_analysis_completed=True,
        detector_failures=("face_detector:ValueError",),
        uncertainty_flags=("visual_blur_outside_validated_range",),
    )
    assert not coverage.complete
