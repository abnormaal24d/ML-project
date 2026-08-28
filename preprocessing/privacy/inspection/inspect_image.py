"""Inspect OCR, metadata, and trusted in-process visual detections."""

from __future__ import annotations

from preprocessing.media.ocr.ocr_result import OcrSpan
from preprocessing.privacy.inspection.content_readers.image_content import (
    ImageContent,
)
from preprocessing.privacy.inspection.detector import (
    run_visual_detectors,
)
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.evidence_location import (
    BoundingBox,
    EvidenceLocation,
)
from preprocessing.privacy.inspection.finding import PrivacyFinding
from preprocessing.privacy.inspection.inspect_text import inspect_text_fields
from preprocessing.privacy.inspection.inspection_coverage import (
    InspectionCoverage,
)
from preprocessing.privacy.inspection.inspection_result import InspectionResult


def _map_ocr_finding_to_bounding_box(
    finding: PrivacyFinding,
    ocr_spans: tuple[OcrSpan, ...],
    ocr_text: str,
) -> tuple[BoundingBox | None, bool]:
    """Map a text finding in ocr_text to a bounding box using OCR spans.

    Returns (bounding_box, mappable) where mappable is False if the finding
    cannot be reliably mapped to a bounding box.
    """
    # Only map findings from the ocr_text field
    if finding.location.field_name != "ocr_text":
        return None, False

    span = finding.location.text_span
    if span is None:
        return None, False

    finding_start = span.start
    finding_end = span.end
    finding_text = ocr_text[finding_start:finding_end]

    if not finding_text.strip():
        return None, False

    # Build a list of spans with their text positions
    located_spans: list[tuple[int, int, OcrSpan]] = []
    cursor = 0

    for ocr_span in ocr_spans:
        # Skip empty spans
        if not ocr_span.text.strip():
            continue

        start = cursor
        cursor += len(ocr_span.text)
        end = cursor

        located_spans.append((start, end, ocr_span))

        # Add space between spans if there's more text
        if cursor < len(ocr_text):
            cursor += 1

    # Find spans that overlap with the finding
    overlapping_spans: list[OcrSpan] = []
    for start, end, ocr_span in located_spans:
        if finding_start < end and finding_end > start:
            if ocr_span.box is not None:
                overlapping_spans.append(ocr_span)

    if not overlapping_spans:
        return None, False

    # Union the overlapping boxes
    boxes = [span.box for span in overlapping_spans if span.box is not None]
    if not boxes:
        return None, False

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)

    return BoundingBox(
        x=max(0, int(left)),
        y=max(0, int(top)),
        width=max(1, int(right - left)),
        height=max(1, int(bottom - top)),
    ), True


def inspect_image(
    content: ImageContent,
    registry: DetectorRegistry,
) -> InspectionResult:
    fields = {
        f"metadata:{key}": value for key, value in content.metadata.items()
    }
    if content.ocr_text is not None:
        fields["ocr_text"] = content.ocr_text
    required = {
        *fields,
        "media_decode",
        "ocr_analysis",
        "visual_analysis",
        "metadata_inspection",
    }
    text_result = inspect_text_fields(
        fields=fields,
        registry=registry,
        language=content.language,
        country=content.country,
        required_fields=frozenset(fields),
        subject_bytes=content.subject_bytes,
    )
    findings = list(text_result.findings)
    failures = [*text_result.errors, *content.analysis_errors]
    runs = list(text_result.detector_runs)

    # Map OCR text findings to bounding boxes
    ocr_uncertainty_flags: list[str] = []
    if content.ocr_text is not None and content.ocr_spans:
        mapped_findings: list[PrivacyFinding] = []
        for finding in findings:
            if finding.location.field_name == "ocr_text":
                box, mappable = _map_ocr_finding_to_bounding_box(
                    finding, content.ocr_spans, content.ocr_text
                )
                if mappable and box is not None:
                    # Create new finding with bounding box
                    new_location = EvidenceLocation(
                        field_name=finding.location.field_name,
                        text_span=finding.location.text_span,
                        page_number=finding.location.page_number,
                        bounding_box=box,
                        time_range=finding.location.time_range,
                        frame_index=finding.location.frame_index,
                    )
                    mapped_findings.append(
                        PrivacyFinding(
                            finding_id=finding.finding_id,
                            finding_type=finding.finding_type,
                            confidence=finding.confidence,
                            location=new_location,
                            detector_name=finding.detector_name,
                            detector_version=finding.detector_version,
                            normalized_value_digest=finding.normalized_value_digest,
                            country=finding.country,
                            language=finding.language,
                            attributes=finding.attributes,
                        )
                    )
                else:
                    # OCR PII found but cannot map to bounding box
                    mapped_findings.append(finding)
                    ocr_uncertainty_flags.append(
                        "ocr_pii_location_unavailable"
                    )
            else:
                mapped_findings.append(finding)
        findings = mapped_findings

    visual_inspection_completed = False
    visual_uncertainty_flags: list[str] = list(
        content.visual_uncertainty_flags
    )
    if content.visual_analysis_completed:
        execution = run_visual_detectors(
            detectors=registry.visual_detectors,
            field_name="visual_content",
            regions=content.visual_regions,
        )

        findings.extend(execution.findings)
        runs.extend(execution.runs)
        failures.extend(execution.failures)
        visual_inspection_completed = execution.completed

    checked = set(text_result.coverage.checked_fields)
    if content.media_decode_completed:
        checked.add("media_decode")
    if content.ocr_analysis_completed:
        checked.add("ocr_analysis")
    if visual_inspection_completed:
        checked.add("visual_analysis")
    if content.metadata_analysis_completed:
        checked.add("metadata_inspection")

    # Combine all uncertainty flags
    all_uncertainty = visual_uncertainty_flags + ocr_uncertainty_flags

    coverage = InspectionCoverage(
        checked_fields=frozenset(checked),
        required_fields=frozenset(required),
        visual_analysis_completed=visual_inspection_completed,
        detector_failures=tuple(failures),
        uncertainty_flags=tuple(dict.fromkeys(all_uncertainty)),
    )
    return InspectionResult(
        subject_digest=text_result.subject_digest,
        findings=tuple(findings),
        coverage=coverage,
        detector_runs=tuple(runs),
        completed=text_result.completed and coverage.complete and not failures,
        errors=tuple(failures),
        detector_versions=tuple(sorted(content.detector_versions.items())),
    )
