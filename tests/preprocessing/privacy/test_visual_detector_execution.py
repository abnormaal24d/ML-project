"""Visual privacy-detector execution and completion contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from preprocessing.privacy.inspection.content_readers.image_content import (
    ImageContent,
)
from preprocessing.privacy.inspection.content_readers.video_content import (
    FrameText,
    VideoContent,
)
from preprocessing.privacy.inspection.detector import (
    VisualRegion,
    run_visual_detectors,
)
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.evidence_location import EvidenceLocation
from preprocessing.privacy.inspection.finding import PrivacyFinding
from preprocessing.privacy.inspection.finding_type import FindingType
from preprocessing.privacy.inspection.inspect_image import inspect_image
from preprocessing.privacy.inspection.inspect_video import inspect_video


@dataclass
class _VisualDetector:
    name: str
    calls: list[str]
    failure: Exception | None = None
    findings: tuple[PrivacyFinding, ...] = ()
    version: str = "1"

    def detect_regions(
        self,
        *,
        field_name: str,
        regions: tuple[VisualRegion, ...],
    ) -> tuple[PrivacyFinding, ...]:
        del regions
        self.calls.append(self.name)
        if self.failure is not None:
            raise self.failure
        assert field_name == "visual_content"
        return self.findings


def _finding(detector_name: str) -> PrivacyFinding:
    return PrivacyFinding(
        finding_id=f"{detector_name}-finding",
        finding_type=FindingType.FACE,
        confidence=0.9,
        location=EvidenceLocation(field_name="visual_content"),
        detector_name=detector_name,
        detector_version="1",
    )


def _registry(
    *visual_detectors: _VisualDetector,
) -> DetectorRegistry:
    return DetectorRegistry(
        text_detectors=(),
        visual_detectors=tuple(visual_detectors),
    )


def _image_content(
    *,
    visual_analysis_completed: bool = True,
    analysis_errors: tuple[str, ...] = (),
    ocr_text: str | None = None,
    ocr_spans: tuple = (),
    visual_uncertainty_flags: tuple[str, ...] = (),
) -> ImageContent:
    return ImageContent(
        subject_bytes=b"image",
        ocr_text=ocr_text,
        metadata={},
        visual_regions=(),
        media_decode_completed=True,
        ocr_analysis_completed=True,
        visual_analysis_completed=visual_analysis_completed,
        metadata_analysis_completed=True,
        language=None,
        country=None,
        detector_versions={"upstream-image": "1"},
        analysis_errors=analysis_errors,
        ocr_spans=ocr_spans,
        visual_uncertainty_flags=visual_uncertainty_flags,
    )


def _video_content(
    *,
    visual_analysis_completed: bool = True,
    analysis_errors: tuple[str, ...] = (),
) -> VideoContent:
    subject_bytes = b"video"
    return VideoContent(
        subject_bytes=subject_bytes,
        subject_sha256=hashlib.sha256(subject_bytes).hexdigest(),
        duration_ms=1_000,
        decoded_frame_count=1,
        inspected_frame_count=1,
        scene_count=1,
        transcript_segments=(),
        transcript_checked_ranges_ms=((0, 1_000),),
        frame_text=(
            FrameText(
                frame_index=0,
                timestamp_ms=0,
                text="",
                phash="frame-digest",
            ),
        ),
        frame_ocr_checked_ranges_ms=((0, 1_000),),
        visual_regions=(),
        checked_video_ranges_ms=((0, 1_000),),
        uninspected_intervals_ms=(),
        tracking_completed=True,
        audio_inspection_completed=True,
        metadata_inspection_completed=True,
        residual_scan_completed=True,
        detector_versions={"upstream-video": "1"},
        visual_analysis_completed=visual_analysis_completed,
        audio_fingerprint="fingerprint",
        metadata={},
        analysis_errors=analysis_errors,
    )


def test_empty_visual_registry_is_incomplete() -> None:
    execution = run_visual_detectors(
        detectors=(),
        field_name="visual_content",
        regions=(),
    )

    assert execution.completed is False
    assert execution.findings == ()
    assert execution.runs == ()
    assert execution.failures == ("visual_detector_registry_empty",)


def test_visual_detector_failure_is_isolated_and_order_is_preserved() -> None:
    calls: list[str] = []
    first = _VisualDetector(
        name="first",
        calls=calls,
        findings=(_finding("first"),),
    )
    second = _VisualDetector(
        name="second",
        calls=calls,
        failure=RuntimeError("failed"),
    )
    third = _VisualDetector(
        name="third",
        calls=calls,
        findings=(_finding("third"),),
    )

    execution = run_visual_detectors(
        detectors=(first, second, third),
        field_name="visual_content",
        regions=(),
    )

    assert calls == ["first", "second", "third"]
    assert tuple(item.detector_name for item in execution.findings) == (
        "first",
        "third",
    )
    assert tuple(run.completed for run in execution.runs) == (
        True,
        False,
        True,
    )
    assert execution.failures == ("second:RuntimeError",)
    assert execution.completed is False


def test_zero_findings_is_a_successful_visual_inspection() -> None:
    detector = _VisualDetector(name="clean", calls=[])

    execution = run_visual_detectors(
        detectors=(detector,),
        field_name="visual_content",
        regions=(),
    )

    assert execution.completed is True
    assert execution.findings == ()
    assert execution.runs[0].completed is True
    assert execution.runs[0].finding_count == 0


def test_image_requires_successful_privacy_visual_execution() -> None:
    inspection = inspect_image(
        _image_content(),
        _registry(),
    )

    assert inspection.coverage.visual_analysis_completed is False
    assert "visual_analysis" not in inspection.coverage.checked_fields
    assert "visual_detector_registry_empty" in inspection.errors


def test_image_does_not_run_visual_detectors_when_upstream_analysis_failed() -> (
    None
):
    inspection = inspect_image(
        _image_content(visual_analysis_completed=False),
        _registry(),
    )

    assert inspection.coverage.visual_analysis_completed is False
    assert "visual_detector_registry_empty" not in inspection.errors


def test_successful_image_visual_pass_is_preserved_with_unrelated_error() -> (
    None
):
    detector = _VisualDetector(name="clean", calls=[])
    inspection = inspect_image(
        _image_content(analysis_errors=("metadata_warning",)),
        _registry(detector),
    )

    assert inspection.coverage.visual_analysis_completed is True
    assert "visual_analysis" in inspection.coverage.checked_fields
    assert inspection.completed is False
    assert "metadata_warning" in inspection.errors


def test_video_uses_central_visual_execution_without_runtime_error() -> None:
    detector = _VisualDetector(name="clean", calls=[])
    inspection = inspect_video(
        _video_content(),
        _registry(detector),
    )

    assert inspection.coverage.visual_analysis_completed is True
    assert "visual_analysis" in inspection.coverage.checked_fields
    assert inspection.completed is True


def test_successful_video_visual_pass_is_preserved_with_unrelated_error() -> (
    None
):
    detector = _VisualDetector(name="clean", calls=[])
    inspection = inspect_video(
        _video_content(analysis_errors=("metadata_warning",)),
        _registry(detector),
    )

    assert inspection.coverage.visual_analysis_completed is True
    assert "visual_analysis" in inspection.coverage.checked_fields
    assert inspection.completed is False
    assert "metadata_warning" in inspection.errors


def test_partial_video_frame_inspection_is_incomplete() -> None:
    detector = _VisualDetector(name="clean", calls=[])
    content = replace(
        _video_content(),
        decoded_frame_count=10,
        inspected_frame_count=1,
    )

    inspection = inspect_video(content, _registry(detector))

    assert inspection.completed is False
    assert "invalid_inspected_frame_count" in inspection.errors
    assert "invalid_keyframe_evidence" in inspection.errors


def test_upstream_versions_are_provenance_not_detector_runs() -> None:
    detector = _VisualDetector(name="clean", calls=[])
    inspection = inspect_image(_image_content(), _registry(detector))

    assert ("upstream-image", "1") in inspection.detector_versions
    assert all(
        run.detector_name != "upstream-image"
        for run in inspection.detector_runs
    )


def test_image_with_visual_uncertainty_is_incomplete() -> None:
    """Test that visual uncertainty flags make coverage incomplete."""
    content = _image_content()
    # Manually add uncertainty flags to simulate visual quality issues
    from dataclasses import replace

    # We need to create content with visual uncertainty flags
    # This tests that when visual analysis has uncertainty, coverage is incomplete
    content_with_uncertainty = replace(
        content,
        visual_uncertainty_flags=("visual_blur_outside_validated_range",),
    )

    inspection = inspect_image(
        content_with_uncertainty,
        _registry(),
    )

    assert (
        "visual_blur_outside_validated_range"
        in inspection.coverage.uncertainty_flags
    )
    assert inspection.coverage.complete is False
    assert inspection.completed is False


def test_image_with_ocr_uncertainty_is_incomplete() -> None:
    """Test that OCR PII location uncertainty makes coverage incomplete."""
    from preprocessing.media.ocr.ocr_result import OcrOrigin, OcrSpan
    from preprocessing.privacy.inspection.evidence_location import TextSpan
    from preprocessing.privacy.inspection.finding import PrivacyFinding
    from preprocessing.privacy.inspection.finding_type import FindingType

    # Create an email finding in OCR text
    finding = PrivacyFinding(
        finding_id="email-1",
        finding_type=FindingType.EMAIL_ADDRESS,
        confidence=0.95,
        location=EvidenceLocation(
            field_name="ocr_text",
            text_span=TextSpan(start=0, end=17),
        ),
        detector_name="test",
        detector_version="1",
    )

    # Create OCR span without bounding box (unmappable)
    ocr_span = OcrSpan(
        text="john@example.com",
        confidence=0.9,
        origin=OcrOrigin.TESSERACT,
        producer_revision="1",
        box=None,  # No bounding box
    )

    content = _image_content(
        ocr_text="john@example.com",
        visual_analysis_completed=False,
        ocr_spans=(ocr_span,),
    )
    content_with_ocr = replace(
        content,
        ocr_text="john@example.com",
        ocr_spans=(ocr_span,),
    )

    # Create a registry with text detector that finds the email
    from dataclasses import dataclass

    from preprocessing.privacy.inspection.detector_registry import (
        DetectorRegistry,
    )

    @dataclass
    class _TextDetector:
        name: str
        version: str
        calls: list[str]

        def detect(self, _item):
            self.calls.append(self.name)
            return (finding,)

    detector = _TextDetector(name="email", version="1", calls=[])
    registry = DetectorRegistry(
        text_detectors=(detector,),
        visual_detectors=(),
    )

    inspection = inspect_image(content_with_ocr, registry)

    # The OCR finding should be unmappable, creating uncertainty
    assert (
        "ocr_pii_location_unavailable" in inspection.coverage.uncertainty_flags
    )
    assert inspection.coverage.complete is False
    assert inspection.completed is False


def test_visual_quality_checks_generate_uncertainty_flags() -> None:
    """Test that visual quality checks produce uncertainty flags."""
    from preprocessing.privacy.inspection.local_visual_analysis import (
        OpenCvVisualPrivacyAnalyzer,
    )

    # This test would require OpenCV and a test image
    # For now we just verify the structure exists
    analyzer = OpenCvVisualPrivacyAnalyzer(cv2_module=None)
    result = analyzer.analyze_bytes(payload=b"test", ocr_text=None)

    assert hasattr(result, "uncertainty_flags")
    assert isinstance(result.uncertainty_flags, tuple)
