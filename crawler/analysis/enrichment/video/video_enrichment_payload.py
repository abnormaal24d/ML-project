"""Video enrichment payload assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.environment.default_values import (
    ENRICHMENT_PREVIEW_MAX_CHARACTERS,
)
from crawler.analysis.enrichment.video.video_analysis_result import (
    VideoAnalysisResult,
)
from crawler.analysis.enrichment.video.video_semantic_status import (
    resolve_video_semantic_status,
)

if TYPE_CHECKING:
    from config.collection.processors import VideoProcessorSettings
    from crawler.fetching.results.result import FetchResult


def build_video_enrichment_payload(
    *,
    analysis: VideoAnalysisResult,
    result: FetchResult,
    settings: VideoProcessorSettings,
    extract_metadata: bool,
    run_transcription: bool,
) -> dict[str, object]:
    """Build persisted video enrichment from a video analysis result."""

    payload = _semantic_status_payload(
        analysis=analysis,
        result=result,
        settings=settings,
    )
    if extract_metadata:
        payload.update(_metadata_payload(analysis=analysis))
    payload.update(_visual_evidence_payload(analysis=analysis))
    payload.update(_semantic_proxy_payload(analysis=analysis))
    if run_transcription:
        payload.update(_transcription_payload(analysis=analysis))
    # Speaker / normalization / probe are now pure from analysis DTO
    # (all work done by VideoAnalyzer before assembly)
    payload.update(_speaker_diarization_payload(analysis=analysis))
    if extract_metadata:
        payload.update(_normalization_payload(analysis=analysis))
    payload.update(_probe_payload(analysis=analysis))
    payload.update(_build_health_payload(payload=payload))
    return payload


def _semantic_status_payload(
    *,
    analysis: VideoAnalysisResult,
    result: FetchResult,
    settings: VideoProcessorSettings,
) -> dict[str, object]:
    semantic_status = resolve_video_semantic_status(
        result=result,
        analysis=analysis,
        settings=settings,
    )
    payload: dict[str, object] = {
        "semantic_video_analysis_status": semantic_status.status,
    }
    if semantic_status.reason:
        payload["semantic_video_analysis_reason"] = semantic_status.reason
    return payload


def _metadata_payload(*, analysis: VideoAnalysisResult) -> dict[str, object]:
    canonical_fields = {
        "duration_seconds": "video_duration_seconds",
        "width": "video_width",
        "height": "video_height",
        "fps": "video_fps",
        "frame_count": "video_frame_count",
    }
    payload: dict[str, object] = {
        key: value
        for key, value in analysis.metadata.items()
        if value is not None
        and key != "final_url"
        and key not in canonical_fields
    }
    payload.update(
        {
            canonical_name: analysis.metadata[raw_name]
            for raw_name, canonical_name in canonical_fields.items()
            if analysis.metadata.get(raw_name) is not None
        }
    )
    payload["video_metadata_status"] = analysis.metadata_status
    return payload


def _visual_evidence_payload(
    *, analysis: VideoAnalysisResult
) -> dict[str, object]:
    payload: dict[str, object] = {}
    payload["keyframes"] = list(analysis.keyframes)
    payload["keyframe_count"] = len(analysis.keyframes)
    payload["scene_boundaries_seconds"] = [
        frame.get("timestamp_seconds")
        for frame in analysis.keyframes
        if isinstance(frame, dict)
        and frame.get("timestamp_seconds") is not None
    ]
    if analysis.frame_ocr_text:
        payload["frame_ocr_text"] = analysis.frame_ocr_text
        payload["frame_ocr_preview"] = analysis.frame_ocr_text[
            :ENRICHMENT_PREVIEW_MAX_CHARACTERS
        ]
    if analysis.frame_ocr_results:
        payload["frame_ocr_results"] = list(analysis.frame_ocr_results)
    return payload


def _transcription_payload(
    *, analysis: VideoAnalysisResult
) -> dict[str, object]:
    transcript_text = analysis.transcript_text
    payload: dict[str, object] = {
        "transcription_status": analysis.transcription_status,
        "transcript_available": bool(
            transcript_text and transcript_text.strip()
        ),
    }
    if analysis.transcription_provenance is not None:
        payload["transcription_provenance"] = dict(
            analysis.transcription_provenance
        )
    if transcript_text is None or not transcript_text.strip():
        return payload
    payload.update(
        {
            "transcript_text": transcript_text,
            "transcript_preview": transcript_text[
                :ENRICHMENT_PREVIEW_MAX_CHARACTERS
            ],
            "transcript_confidence": analysis.transcript_confidence,
            "transcript_source": analysis.transcript_source,
            "transcript_language": analysis.transcript_language,
            "transcript_segments": list(analysis.transcript_segments),
            "transcript_quality_score": analysis.transcript_confidence,
        }
    )
    return payload


def _semantic_proxy_payload(
    *, analysis: VideoAnalysisResult
) -> dict[str, object]:
    payload: dict[str, object] = {}
    if analysis.scene_graph:
        payload["scene_graph"] = analysis.scene_graph
        payload["scene_graph_available"] = True
        payload["scene_graph_status"] = analysis.scene_graph.get("status")
        payload["scene_graph_semantics"] = analysis.scene_graph.get(
            "scene_graph_semantics",
            "visual_proxy",
        )
    if analysis.action_segments or analysis.motion_segments:
        payload["action_segments"] = list(analysis.action_segments)
        payload["motion_segments"] = list(analysis.motion_segments)
        if analysis.action_label:
            payload["action_label"] = analysis.action_label
            payload["activity_label"] = analysis.action_label
        if analysis.action_analysis_status:
            payload["action_analysis_status"] = analysis.action_analysis_status
        if analysis.action_analysis_reasons:
            payload["action_analysis_reasons"] = list(
                analysis.action_analysis_reasons
            )
        if analysis.action_proxy_backend:
            payload["action_proxy_backend"] = analysis.action_proxy_backend
        if analysis.action_recognition_status:
            payload["action_recognition_status"] = (
                analysis.action_recognition_status
            )
        payload["action_recognition_available"] = (
            analysis.action_recognition_available
        )
        payload["motion_proxy_available"] = True
        payload["motion_proxy_status"] = "passed"
    return payload


def _speaker_diarization_payload(
    *,
    analysis: VideoAnalysisResult,
) -> dict[str, object]:
    """Pure serialization — shaping done by video_analyzer + this payload builder."""
    if not analysis.speaker_diarization_status:
        return {}

    payload: dict[str, object] = {
        "speaker_segments": list(analysis.speaker_segments),
        "speaker_diarization_available": bool(analysis.speaker_segments),
        "speaker_diarization_status": analysis.speaker_diarization_status,
        "speaker_count": analysis.speaker_count,
    }
    if analysis.speaker_diarization_error_type:
        payload["speaker_diarization_error_type"] = (
            analysis.speaker_diarization_error_type
        )
    return payload


def _normalization_payload(
    *, analysis: VideoAnalysisResult
) -> dict[str, object]:
    """Pure serialization (shaping by assembler)."""
    payload: dict[str, object] = {}
    if analysis.normalized_video_path:
        payload["normalized_video_path"] = analysis.normalized_video_path
    if analysis.video_normalization_status:
        payload["video_normalization_status"] = (
            analysis.video_normalization_status
        )
    if analysis.normalization_error_type:
        payload["video_normalization_error_type"] = (
            analysis.normalization_error_type
        )
    return payload


def _probe_payload(*, analysis: VideoAnalysisResult) -> dict[str, object]:
    """Pure serialization of probe results (probe_result shaped in assembler)."""
    payload: dict[str, object] = {}
    if analysis.video_probe_metadata:
        payload["video_probe_metadata"] = analysis.video_probe_metadata
    if analysis.video_probe_status:
        payload["video_probe_status"] = analysis.video_probe_status
    if analysis.video_probe_error_type:
        payload["video_probe_error_type"] = analysis.video_probe_error_type
    return payload


def _build_health_payload(*, payload: dict[str, object]) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    for feature, status_key, error_key, severity in (
        (
            "speaker_diarization",
            "speaker_diarization_status",
            "speaker_diarization_error_type",
            "optional",
        ),
        (
            "video_normalization",
            "video_normalization_status",
            "video_normalization_error_type",
            "quality_signal",
        ),
        (
            "video_probe",
            "video_probe_status",
            "video_probe_error_type",
            "quality_signal",
        ),
    ):
        status = payload.get(status_key)
        if status is None or status == "passed":
            continue
        item = {
            "feature": feature,
            "status": status,
            "severity": severity,
        }
        error_type = payload.get(error_key)
        if error_type is not None:
            item["error_type"] = error_type
        if status == "failed":
            failures.append(item)
        else:
            warnings.append(item)

    quality_failures = [
        item for item in failures if item["severity"] == "quality_signal"
    ]
    quality_warnings = [
        item for item in warnings if item["severity"] == "quality_signal"
    ]
    optional_failures = [
        item for item in failures if item["severity"] == "optional"
    ]
    if quality_failures:
        status = "quality_signal_failed"
    elif optional_failures:
        status = "optional_failed"
    elif warnings:
        status = "partial_unavailable"
    else:
        status = "passed"

    return {
        "video_enrichment_status": status,
        "video_enrichment_complete": not failures and not warnings,
        "video_enrichment_failures": failures,
        "video_enrichment_warnings": warnings,
        "video_enrichment_quality_failures": quality_failures,
        "video_enrichment_quality_warnings": quality_warnings,
        "video_enrichment_optional_failures": optional_failures,
        "video_enrichment_quality_decision": _quality_decision(
            quality_failures=quality_failures,
            quality_warnings=quality_warnings,
        ),
    }


def _quality_decision(
    *,
    quality_failures: list[dict[str, object]],
    quality_warnings: list[dict[str, object]],
) -> str:
    if quality_failures:
        return "failed"
    if quality_warnings:
        return "review"
    return "passed"
