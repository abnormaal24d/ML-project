"""Precise locations for text, page, image, audio, and video findings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("text span must satisfy 0 <= start < end")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if min(self.x, self.y, self.width, self.height) < 0:
            raise ValueError("bounding-box values must be non-negative")
        if self.width == 0 or self.height == 0:
            raise ValueError("bounding box must have positive dimensions")


@dataclass(frozen=True, slots=True)
class TimeRange:
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("time range must satisfy 0 <= start < end")


@dataclass(frozen=True, slots=True)
class EvidenceLocation:
    field_name: str
    text_span: TextSpan | None = None
    page_number: int | None = None
    bounding_box: BoundingBox | None = None
    time_range: TimeRange | None = None
    frame_index: int | None = None

    def __post_init__(self) -> None:
        if not self.field_name.strip():
            raise ValueError("field_name must not be blank")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number is one-based")
        if self.frame_index is not None and self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
