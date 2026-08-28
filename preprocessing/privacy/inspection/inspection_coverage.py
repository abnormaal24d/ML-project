"""Coverage evidence distinguishing no findings from no inspection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InspectionCoverage:
    checked_fields: frozenset[str]
    required_fields: frozenset[str]
    checked_pages: frozenset[int] = frozenset()
    expected_page_count: int | None = None
    checked_audio_ranges_ms: tuple[tuple[int, int], ...] = ()
    expected_audio_duration_ms: int | None = None
    checked_video_ranges_ms: tuple[tuple[int, int], ...] = ()
    expected_video_duration_ms: int | None = None
    visual_analysis_completed: bool | None = None
    detector_failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    uncertainty_flags: tuple[str, ...] = ()

    @property
    def unchecked_fields(self) -> frozenset[str]:
        return self.required_fields - self.checked_fields

    @property
    def complete(self) -> bool:
        if self.detector_failures or self.unchecked_fields:
            return False

        if self.uncertainty_flags:
            return False

        if any(page_number <= 0 for page_number in self.checked_pages):
            return False
        if self.expected_page_count is not None:
            if self.expected_page_count <= 0:
                return False
            expected = frozenset(range(1, self.expected_page_count + 1))
            if self.checked_pages != expected:
                return False
        if self.expected_audio_duration_ms is not None:
            if not ranges_cover_duration(
                ranges=self.checked_audio_ranges_ms,
                duration_ms=self.expected_audio_duration_ms,
            ):
                return False
        if self.expected_video_duration_ms is not None:
            if not ranges_cover_duration(
                ranges=self.checked_video_ranges_ms,
                duration_ms=self.expected_video_duration_ms,
            ):
                return False
        return self.visual_analysis_completed is not False


def ranges_cover_duration(
    *,
    ranges: tuple[tuple[int, int], ...],
    duration_ms: int,
) -> bool:
    """Return whether valid ranges continuously cover the full duration."""

    if duration_ms <= 0 or not ranges:
        return False

    ordered = sorted(ranges)

    if any(start < 0 or end <= start for start, end in ordered):
        return False

    cursor = 0

    for start, end in ordered:
        if start > cursor:
            return False

        cursor = max(
            cursor,
            min(end, duration_ms),
        )

    return cursor >= duration_ms
