"""Inspect audio transcript, metadata, coverage, and voice evidence locally."""

from __future__ import annotations

from preprocessing.privacy.inspection.content_readers.audio_content import (
    AudioContent,
)
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.inspect_text import inspect_text_fields
from preprocessing.privacy.inspection.inspection_coverage import (
    InspectionCoverage,
    ranges_cover_duration,
)
from preprocessing.privacy.inspection.inspection_result import InspectionResult


def inspect_audio(
    content: AudioContent,
    registry: DetectorRegistry,
) -> InspectionResult:
    fields = {
        f"metadata:{key}": value for key, value in content.metadata.items()
    }
    for index, segment in enumerate(content.transcript_segments):
        fields[f"transcript:{index}"] = segment.text
    required = {
        *fields,
        "media_decode",
        "transcript_coverage",
        "speaker_analysis",
        "background_speech_analysis",
        "voice_analysis",
        "metadata_inspection",
        "audio_fingerprint",
    }
    result = inspect_text_fields(
        fields=fields,
        registry=registry,
        language=content.language,
        country=content.country,
        required_fields=frozenset(fields),
        subject_bytes=content.subject_bytes,
    )
    checked = set(result.coverage.checked_fields)
    if content.full_decode_completed:
        checked.add("media_decode")
    if content.speaker_analysis_completed:
        checked.add("speaker_analysis")
    if content.background_speech_analysis_completed:
        checked.add("background_speech_analysis")
    if content.voice_analysis_completed:
        checked.add("voice_analysis")
    if content.metadata_analysis_completed:
        checked.add("metadata_inspection")
    if content.audio_fingerprint and content.audio_fingerprint.strip():
        checked.add("audio_fingerprint")

    ranges = content.transcript_checked_ranges_ms or tuple(
        (segment.start_ms, segment.end_ms)
        for segment in content.transcript_segments
    )
    if content.transcript_analysis_completed and ranges_cover_duration(
        ranges=ranges,
        duration_ms=content.duration_ms,
    ):
        checked.add("transcript_coverage")

    failures = [*result.errors, *content.analysis_errors]
    runs = list(result.detector_runs)
    if (
        content.voice_identity_detected
        and not content.voice_identity_authorized
    ):
        failures.append("identifiable_voice_without_authorization")
    coverage = InspectionCoverage(
        checked_fields=frozenset(checked),
        required_fields=frozenset(required),
        checked_audio_ranges_ms=ranges,
        expected_audio_duration_ms=content.duration_ms,
        detector_failures=tuple(failures),
    )
    return InspectionResult(
        subject_digest=result.subject_digest,
        findings=result.findings,
        coverage=coverage,
        detector_runs=tuple(runs),
        completed=result.completed and coverage.complete and not failures,
        errors=tuple(failures),
        detector_versions=tuple(sorted(content.detector_versions.items())),
    )
