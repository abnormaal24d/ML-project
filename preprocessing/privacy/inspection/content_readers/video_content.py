"""Complete video inspection input and local coverage evidence."""

from dataclasses import dataclass, field

from preprocessing.privacy.inspection.content_readers.audio_content import (
    TranscriptSegment,
)
from preprocessing.privacy.inspection.detector import VisualRegion


@dataclass(frozen=True, slots=True)
class FrameText:
    frame_index: int
    timestamp_ms: int
    text: str
    phash: str


@dataclass(frozen=True, slots=True)
class VideoContent:
    subject_bytes: bytes
    subject_sha256: str
    duration_ms: int
    decoded_frame_count: int
    inspected_frame_count: int
    scene_count: int
    transcript_segments: tuple[TranscriptSegment, ...]
    transcript_checked_ranges_ms: tuple[tuple[int, int], ...]
    frame_text: tuple[FrameText, ...]
    frame_ocr_checked_ranges_ms: tuple[tuple[int, int], ...]
    visual_regions: tuple[VisualRegion, ...]
    checked_video_ranges_ms: tuple[tuple[int, int], ...]
    uninspected_intervals_ms: tuple[tuple[int, int], ...]
    tracking_completed: bool
    audio_inspection_completed: bool
    metadata_inspection_completed: bool
    residual_scan_completed: bool
    detector_versions: dict[str, str]
    visual_analysis_completed: bool
    audio_fingerprint: str | None
    metadata: dict[str, str] = field(default_factory=dict)
    language: str | None = None
    country: str | None = None
    analysis_errors: tuple[str, ...] = ()
