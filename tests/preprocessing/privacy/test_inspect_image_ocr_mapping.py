"""Tests for OCR finding to bounding box mapping and uncertainty handling."""

from __future__ import annotations

from dataclasses import replace

from preprocessing.media.ocr.ocr_result import OcrOrigin, OcrSpan
from preprocessing.privacy.inspection.content_readers.image_content import (
    ImageContent,
)
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.evidence_location import (
    EvidenceLocation,
    TextSpan,
)
from preprocessing.privacy.inspection.finding import PrivacyFinding
from preprocessing.privacy.inspection.finding_type import FindingType
from preprocessing.privacy.inspection.inspect_image import (
    _map_ocr_finding_to_bounding_box,
    inspect_image,
)


def _image_content_base() -> ImageContent:
    return ImageContent(
        subject_bytes=b"image",
        ocr_text=None,
        metadata={},
        visual_regions=(),
        media_decode_completed=True,
        ocr_analysis_completed=True,
        visual_analysis_completed=False,
        metadata_analysis_completed=True,
        language=None,
        country=None,
        detector_versions={},
        analysis_errors=(),
        ocr_spans=(),
        visual_uncertainty_flags=(),
    )


def _make_finding(
    field_name: str,
    text_span: TextSpan | None = None,
) -> PrivacyFinding:
    return PrivacyFinding(
        finding_id="test-finding",
        finding_type=FindingType.EMAIL_ADDRESS,
        confidence=0.95,
        location=EvidenceLocation(
            field_name=field_name,
            text_span=text_span,
        ),
        detector_name="test",
        detector_version="1",
    )


def _make_ocr_span(
    text: str,
    box: tuple[float, float, float, float] | None,
) -> OcrSpan:
    return OcrSpan(
        text=text,
        confidence=0.9,
        origin=OcrOrigin.TESSERACT,
        producer_revision="1",
        box=box,
    )


def test_map_ocr_finding_single_span_with_box() -> None:
    """Single OCR span with box maps correctly."""
    ocr_text = "john@example.com"
    ocr_spans = (
        _make_ocr_span("john@example.com", (10.0, 20.0, 100.0, 30.0)),
    )
    finding = _make_finding("ocr_text", TextSpan(start=0, end=17))

    box, mappable = _map_ocr_finding_to_bounding_box(
        finding, ocr_spans, ocr_text
    )

    assert mappable is True
    assert box is not None
    assert box.x == 10
    assert box.y == 20
    assert box.width == 100
    assert box.height == 30


def test_map_ocr_finding_multiple_spans_unions_boxes() -> None:
    """Multiple overlapping spans are unioned."""
    ocr_text = "john@example.com call 555-1234"
    ocr_spans = (
        _make_ocr_span("john@example.com", (10.0, 20.0, 100.0, 30.0)),
        _make_ocr_span(" call ", (110.0, 20.0, 40.0, 30.0)),
        _make_ocr_span("555-1234", (150.0, 20.0, 60.0, 30.0)),
    )
    # Finding spans the email part
    finding = _make_finding("ocr_text", TextSpan(start=0, end=17))

    box, mappable = _map_ocr_finding_to_bounding_box(
        finding, ocr_spans, ocr_text
    )

    assert mappable is True
    assert box is not None
    # Should cover just the email span
    assert box.x == 10
    assert box.y == 20
    assert box.width == 100
    assert box.height == 30


def test_map_ocr_finding_no_overlapping_boxes_returns_unmappable() -> None:
    """Finding with no overlapping spans that have boxes is unmappable."""
    ocr_text = "john@example.com"
    ocr_spans = (_make_ocr_span("john@example.com", None),)  # No box
    finding = _make_finding("ocr_text", TextSpan(start=0, end=17))

    box, mappable = _map_ocr_finding_to_bounding_box(
        finding, ocr_spans, ocr_text
    )

    assert mappable is False
    assert box is None


def test_map_ocr_finding_empty_spans_returns_unmappable() -> None:
    """Empty OCR spans list returns unmappable."""
    ocr_text = "john@example.com"
    ocr_spans = ()
    finding = _make_finding("ocr_text", TextSpan(start=0, end=17))

    box, mappable = _map_ocr_finding_to_bounding_box(
        finding, ocr_spans, ocr_text
    )

    assert mappable is False
    assert box is None


def test_map_ocr_finding_no_text_span_returns_unmappable() -> None:
    """Finding without text span returns unmappable."""
    ocr_text = "john@example.com"
    ocr_spans = (
        _make_ocr_span("john@example.com", (10.0, 20.0, 100.0, 30.0)),
    )
    finding = _make_finding("ocr_text", text_span=None)

    box, mappable = _map_ocr_finding_to_bounding_box(
        finding, ocr_spans, ocr_text
    )

    assert mappable is False
    assert box is None


def test_map_ocr_finding_finding_in_metadata_field_not_mapped() -> None:
    """Only ocr_text field findings are mapped."""
    ocr_text = "john@example.com"
    ocr_spans = (
        _make_ocr_span("john@example.com", (10.0, 20.0, 100.0, 30.0)),
    )
    finding = _make_finding("metadata:email", TextSpan(start=0, end=17))

    box, mappable = _map_ocr_finding_to_bounding_box(
        finding, ocr_spans, ocr_text
    )

    assert mappable is False
    assert box is None


def test_inspect_image_ocr_finding_with_box_gets_bounding_box() -> None:
    """OCR finding with mappable box gets bounding box attached."""
    ocr_span = _make_ocr_span("john@example.com", (10.0, 20.0, 100.0, 30.0))
    content = replace(
        _image_content_base(),
        ocr_text="john@example.com",
        ocr_spans=(ocr_span,),
    )

    class _TextDetector:
        name = "email"
        version = "1"

        def detect(self, _item):
            return (_make_finding("ocr_text", TextSpan(start=0, end=17)),)

    detector = _TextDetector()
    registry = DetectorRegistry(
        text_detectors=(detector,), visual_detectors=()
    )

    inspection = inspect_image(content, registry)

    assert len(inspection.findings) == 1
    finding = inspection.findings[0]
    assert finding.location.bounding_box is not None
    assert finding.location.bounding_box.x == 10
    assert finding.location.bounding_box.y == 20
    assert finding.location.bounding_box.width == 100
    assert finding.location.bounding_box.height == 30
    assert (
        "ocr_pii_location_unavailable"
        not in inspection.coverage.uncertainty_flags
    )


def test_inspect_image_ocr_finding_without_box_adds_uncertainty() -> None:
    """OCR finding without mappable box adds uncertainty flag."""
    ocr_span = _make_ocr_span("john@example.com", None)  # No box
    content = replace(
        _image_content_base(),
        ocr_text="john@example.com",
        ocr_spans=(ocr_span,),
    )

    class _TextDetector:
        name = "email"
        version = "1"

        def detect(self, _item):
            return (_make_finding("ocr_text", TextSpan(start=0, end=17)),)

    detector = _TextDetector()
    registry = DetectorRegistry(
        text_detectors=(detector,), visual_detectors=()
    )

    inspection = inspect_image(content, registry)

    assert len(inspection.findings) == 1
    finding = inspection.findings[0]
    # Finding should still exist but without bounding box
    assert finding.location.bounding_box is None
    assert (
        "ocr_pii_location_unavailable" in inspection.coverage.uncertainty_flags
    )
    assert inspection.coverage.complete is False


def _image_content_base() -> ImageContent:
    return ImageContent(
        subject_bytes=b"image",
        ocr_text=None,
        metadata={},
        visual_regions=(),
        media_decode_completed=True,
        ocr_analysis_completed=True,
        visual_analysis_completed=False,
        metadata_analysis_completed=True,
        language=None,
        country=None,
        detector_versions={},
        analysis_errors=(),
        ocr_spans=(),
        visual_uncertainty_flags=(),
    )
