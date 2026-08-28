"""Receipts for video annotation transformations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoAnnotationReceipt:
    """Counts describing annotation clipping and coordinate transformation."""

    transformed_boxes: int = 0
    dropped_boxes: int = 0
    transformed_intervals: int = 0
    dropped_intervals: int = 0
    transformed_points: int = 0
    dropped_points: int = 0

    def merge(self, other: VideoAnnotationReceipt) -> VideoAnnotationReceipt:
        return VideoAnnotationReceipt(
            transformed_boxes=self.transformed_boxes + other.transformed_boxes,
            dropped_boxes=self.dropped_boxes + other.dropped_boxes,
            transformed_intervals=(
                self.transformed_intervals + other.transformed_intervals
            ),
            dropped_intervals=self.dropped_intervals + other.dropped_intervals,
            transformed_points=self.transformed_points
            + other.transformed_points,
            dropped_points=self.dropped_points + other.dropped_points,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "transformed_boxes": self.transformed_boxes,
            "dropped_boxes": self.dropped_boxes,
            "transformed_intervals": self.transformed_intervals,
            "dropped_intervals": self.dropped_intervals,
            "transformed_points": self.transformed_points,
            "dropped_points": self.dropped_points,
        }
