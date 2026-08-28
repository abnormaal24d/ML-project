"""Image inspection input with explicit decode and detector evidence."""

from dataclasses import dataclass, field

from preprocessing.media.ocr.ocr_result import OcrSpan
from preprocessing.privacy.inspection.detector import VisualRegion


@dataclass(frozen=True, slots=True)
class ImageContent:
    subject_bytes: bytes
    ocr_text: str | None
    metadata: dict[str, str]
    visual_regions: tuple[VisualRegion, ...]
    media_decode_completed: bool
    ocr_analysis_completed: bool
    visual_analysis_completed: bool
    metadata_analysis_completed: bool
    language: str | None
    country: str | None
    detector_versions: dict[str, str] = field(default_factory=dict)
    analysis_errors: tuple[str, ...] = ()
    ocr_spans: tuple[OcrSpan, ...] = ()
    visual_uncertainty_flags: tuple[str, ...] = ()
