"""Fail-closed local video transcript, frame, tracking and metadata inspection."""

from __future__ import annotations

import hashlib

from preprocessing.privacy.inspection.content_readers.video_content import (
    VideoContent,
)
from preprocessing.privacy.inspection.detector import (
    run_visual_detectors,
)
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.inspect_text import inspect_text_fields
from preprocessing.privacy.inspection.inspection_coverage import (
    InspectionCoverage,
    ranges_cover_duration,
)
from preprocessing.privacy.inspection.inspection_result import InspectionResult


def inspect_video(
    content: VideoContent,
    registry: DetectorRegistry,
) -> InspectionResult:
    failures: list[str] = list(content.analysis_errors)
    actual_sha256 = hashlib.sha256(content.subject_bytes).hexdigest()
    if actual_sha256 != content.subject_sha256:
        failures.append("subject_sha256_mismatch")
    if content.duration_ms <= 0:
        failures.append("duration_missing")
    if content.decoded_frame_count <= 0:
        failures.append("no_decoded_frames")
    if (
        content.inspected_frame_count <= 0
        or content.inspected_frame_count != content.decoded_frame_count
    ):
        failures.append("invalid_inspected_frame_count")
    if (
        content.scene_count <= 0
        or content.scene_count > content.decoded_frame_count
    ):
        failures.append("invalid_scene_evidence")
    if content.uninspected_intervals_ms:
        failures.append("uninspected_video_intervals")

    frame_indexes = tuple(frame.frame_index for frame in content.frame_text)
    expected_frame_indexes = tuple(range(content.decoded_frame_count))
    if (
        len(content.frame_text) != content.inspected_frame_count
        or len(set(frame_indexes)) != len(frame_indexes)
        or tuple(sorted(frame_indexes)) != expected_frame_indexes
        or any(
            not frame.phash.strip()
            or frame.timestamp_ms < 0
            or (
                content.duration_ms > 0
                and frame.timestamp_ms >= content.duration_ms
            )
            for frame in content.frame_text
        )
    ):
        failures.append("invalid_keyframe_evidence")
    if any(
        (
            region.frame_index is not None
            and not 0 <= region.frame_index < content.decoded_frame_count
        )
        or (
            region.timestamp_ms is not None
            and (
                region.timestamp_ms < 0
                or (
                    content.duration_ms > 0
                    and region.timestamp_ms >= content.duration_ms
                )
            )
        )
        for region in content.visual_regions
    ):
        failures.append("invalid_visual_region_evidence")
    if not content.detector_versions or any(
        not str(name).strip() or not str(version).strip()
        for name, version in content.detector_versions.items()
    ):
        failures.append("invalid_detector_versions")

    fields = {
        f"metadata:{key}": value for key, value in content.metadata.items()
    }
    for index, segment in enumerate(content.transcript_segments):
        fields[f"transcript:{index}"] = segment.text
    for frame in content.frame_text:
        if frame.text:
            fields[f"frame_ocr:{frame.frame_index}:{frame.timestamp_ms}"] = (
                frame.text
            )

    evidence_required = {
        "media_decode",
        "transcript_coverage",
        "frame_ocr_coverage",
        "visual_analysis",
        "metadata_inspection",
        "tracking",
        "audio_inspection",
        "residual_scan",
        "scene_detection",
        "detector_versions",
    }
    required = set(fields) | evidence_required
    text_result = inspect_text_fields(
        fields=fields,
        registry=registry,
        language=content.language,
        country=content.country,
        required_fields=frozenset(fields),
        subject_bytes=content.subject_bytes,
    )
    findings = list(text_result.findings)
    failures.extend(text_result.errors)
    runs = list(text_result.detector_runs)

    checked = set(text_result.coverage.checked_fields)

    visual_inspection_completed = False
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

    if content.decoded_frame_count > 0:
        checked.add("media_decode")
    if ranges_cover_duration(
        ranges=content.transcript_checked_ranges_ms,
        duration_ms=content.duration_ms,
    ):
        checked.add("transcript_coverage")
    if ranges_cover_duration(
        ranges=content.frame_ocr_checked_ranges_ms,
        duration_ms=content.duration_ms,
    ):
        checked.add("frame_ocr_coverage")
    if visual_inspection_completed:
        checked.add("visual_analysis")
    if content.metadata_inspection_completed:
        checked.add("metadata_inspection")
    if content.tracking_completed:
        checked.add("tracking")
    if content.audio_inspection_completed:
        checked.add("audio_inspection")
    if content.residual_scan_completed:
        checked.add("residual_scan")
    if content.scene_count > 0:
        checked.add("scene_detection")
    if content.detector_versions:
        checked.add("detector_versions")

    coverage = InspectionCoverage(
        checked_fields=frozenset(checked),
        required_fields=frozenset(required),
        checked_video_ranges_ms=content.checked_video_ranges_ms,
        expected_video_duration_ms=content.duration_ms,
        visual_analysis_completed=visual_inspection_completed,
        detector_failures=tuple(dict.fromkeys(failures)),
    )
    return InspectionResult(
        subject_digest=text_result.subject_digest,
        findings=tuple(findings),
        coverage=coverage,
        detector_runs=tuple(runs),
        completed=not failures and coverage.complete,
        errors=tuple(dict.fromkeys(failures)),
        detector_versions=tuple(sorted(content.detector_versions.items())),
    )
