"""Video preprocessing orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from logger.project_logger import ProjectLogger
from preprocessing.media.base_media_preprocessor import BaseMediaPreprocessor
from preprocessing.media.media_input_validation import (
    MediaValidationResult,
    accepted_media_result,
    as_optional_float,
    as_optional_int,
    as_optional_text,
    has_video_training_metadata,
    is_metadata_fetch_mode,
    modality_preprocessing_limit,
    payload_field,
    rejected_media_result,
    resolve_media_path,
    resolve_path_object,
    validate_common_media_fields,
)
from preprocessing.media.privacy_inspection import (
    inspect_media_privacy,
)
from preprocessing.media.transcript_segment_normalizer import (
    normalize_segments,
    summarize_timeline,
)
from preprocessing.preprocessed_media import (
    PreprocessedVideo,
    canonical_privacy_intervals,
    canonical_transcript_segments,
    canonical_video_structure,
)
from preprocessing.preprocessing_input import PreprocessingInput
from preprocessing.preprocessing_result import PreprocessingQuarantineRecord
from preprocessing.privacy.field_inspection import text_payload_fields
from preprocessing.privacy.inspection.inspect_video import inspect_video
from preprocessing.privacy.inspection.local_content_factories import (
    VideoPrivacyContentFactory,
)
from preprocessing.privacy.text_privacy import PiiDetector

if TYPE_CHECKING:
    from config.collection.modality_acceptance import (
        ModalityAcceptanceSettings,
    )
    from config.preprocessing.media_settings import VideoValidationSettings
    from preprocessing.media.ports import EmbeddedMetadataAdapter, VideoReader

_ALLOWED_VIDEO_MIME_TYPES: tuple[str, ...] = (
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/ogg",
    "video/x-matroska",
    "video/x-msvideo",
)


class VideoPreprocessor(BaseMediaPreprocessor[PreprocessedVideo]):
    """Validate video transcript, OCR, keyframes, and fingerprints."""

    def __init__(
        self,
        *,
        logger: ProjectLogger,
        settings: VideoValidationSettings,
        modality_acceptance: ModalityAcceptanceSettings,
        max_duration_seconds: float | None = None,
        pii_detector: PiiDetector,
        privacy_content_factory: VideoPrivacyContentFactory,
        video_reader: VideoReader,
        embedded_metadata_adapter: EmbeddedMetadataAdapter,
        now: Callable[[], datetime],
        generate_id: Callable[[], str],
    ) -> None:
        super().__init__(
            modality="video",
            logger=logger,
            now=now,
            generate_id=generate_id,
        )
        self._settings = settings
        self._modality_acceptance = modality_acceptance
        self._pii_detector = pii_detector
        self._privacy_content_factory = privacy_content_factory
        self._video_reader = video_reader
        self._embedded_metadata_adapter = embedded_metadata_adapter
        self._max_duration_seconds = (
            float(max_duration_seconds)
            if max_duration_seconds is not None and max_duration_seconds > 0.0
            else 0.0
        )

    def _validate(self, *, item: PreprocessingInput) -> MediaValidationResult:
        if not self._settings.enabled:
            return accepted_media_result(signals={})
        reason, signals = validate_common_media_fields(
            item=item,
            allowed_mime_types=_ALLOWED_VIDEO_MIME_TYPES,
            min_bytes=self._settings.min_bytes,
            max_bytes=modality_preprocessing_limit(self._modality_acceptance),
        )
        media_path = resolve_media_path(item=item)
        path = resolve_path_object(media_path=media_path)
        probed = _probe_video(path=path, reader=self._video_reader)
        metadata = _resolve_video_metadata(item=item, probe=probed)
        duration = metadata["duration_seconds"]
        width = metadata["width"]
        height = metadata["height"]
        fps = metadata["fps"]
        keyframe_count = metadata["keyframe_count"]
        has_semantic_or_frames = _has_semantic_text_or_keyframes(
            item=item,
            keyframe_count=keyframe_count,
        )
        signals.update(
            {
                "duration_seconds": duration,
                "width": width,
                "height": height,
                "fps": fps,
                "keyframe_count": keyframe_count,
                "has_semantic_text_or_keyframes": has_semantic_or_frames,
                "decode_checked": probed.get("decode_checked") is True,
            }
        )
        if reason is not None:
            metadata_only = _accept_metadata_only_video(
                item=item, reason=reason, signals=signals, probed=probed
            )
            if metadata_only is not None:
                return metadata_only
            return rejected_media_result(reason=reason, signals=signals)
        if probed.get("decode_failed") is True:
            metadata_only = _accept_metadata_only_video(
                item=item,
                reason="decode_failed",
                signals=signals,
                probed=probed,
            )
            if metadata_only is not None:
                return metadata_only
            return rejected_media_result(
                reason="decode_failed", signals=signals
            )
        if duration is None:
            return rejected_media_result(
                reason="duration_missing", signals=signals
            )
        if duration < self._settings.min_duration_seconds:
            return rejected_media_result(
                reason="video_too_short", signals=signals
            )
        if (
            self._max_duration_seconds > 0.0
            and duration > self._max_duration_seconds
        ):
            return rejected_media_result(
                reason="video_too_long", signals=signals
            )
        if (
            width is None
            or height is None
            or width < self._settings.min_width
            or height < self._settings.min_height
        ):
            return rejected_media_result(
                reason="invalid_dimensions", signals=signals
            )
        if fps is None:
            return rejected_media_result(
                reason="video_fps_missing", signals=signals
            )
        if fps < self._settings.min_fps:
            return rejected_media_result(
                reason="video_fps_too_low", signals=signals
            )
        if self._settings.require_keyframes and keyframe_count <= 0:
            return rejected_media_result(
                reason="missing_keyframes", signals=signals
            )
        if (
            self._settings.require_semantic_text_or_keyframes
            and not has_semantic_or_frames
        ):
            return rejected_media_result(
                reason="needs_video_enrichment", signals=signals
            )
        return accepted_media_result(signals=signals)

    def _build_record(
        self,
        *,
        item: PreprocessingInput,
        validation: MediaValidationResult,
    ) -> PreprocessedVideo | PreprocessingQuarantineRecord:
        transcript_text = item.transcript_text or as_optional_text(
            item.payload.get("transcript_text")
        )
        frame_ocr_text = item.ocr_text or as_optional_text(
            item.payload.get("frame_ocr_text")
        )

        # Validators own resolved stream metadata; do not re-read item/payload.
        duration_seconds = as_optional_float(
            validation.signals.get("duration_seconds")
        )
        width = as_optional_int(validation.signals.get("width"))
        height = as_optional_int(validation.signals.get("height"))
        fps = as_optional_float(validation.signals.get("fps"))
        video_probe_metadata = _probe_metadata(
            item.payload.get("video_probe_metadata")
        )
        keyframe_paths = _normalize_keyframe_paths(
            item.payload.get("keyframes")
        )
        segments = normalize_segments(item.payload.get("transcript_segments"))
        fields = text_payload_fields(
            item=item,
            names=(
                "transcript_text",
                "transcript_preview",
                "frame_ocr_text",
                "frame_ocr_preview",
                "page_title",
                "surrounding_text",
                "html_context",
                "author",
                "creator",
                "description",
            ),
        )
        if transcript_text:
            fields["transcript_text"] = transcript_text
        if frame_ocr_text:
            fields["frame_ocr_text"] = frame_ocr_text
        for index, segment in enumerate(segments):
            text = as_optional_text(segment.get("text"))
            if text:
                fields[f"transcript_segment:{index}"] = text

        duration_ms = int((duration_seconds or 0.0) * 1000)
        embedded_fields, metadata_artifact, metadata_rejection = (
            self._prepare_embedded_metadata(
                item=item,
                adapter=self._embedded_metadata_adapter,
            )
        )
        if metadata_rejection is not None:
            return PreprocessingQuarantineRecord.from_input(
                item=item,
                reason=metadata_rejection,
                quality_signals={},
            )
        fields.update(embedded_fields)
        inspected_path = Path(
            metadata_artifact.path
            if metadata_artifact is not None
            else item.media_path or ""
        )
        inspection_content = self._privacy_content_factory.build(
            item=item,
            media_path=inspected_path,
            metadata={},
            duration_ms=duration_ms,
            transcript_segments=segments,
            residual=False,
        )
        inspection = inspect_video(
            inspection_content,
            self._pii_detector.registry,
        )
        privacy = inspect_media_privacy(
            item=item,
            object_id=self._media_id(item=item),
            detector=self._pii_detector,
            fields=fields,
            inspection=inspection,
            media_path=str(inspected_path),
            source_media_path=item.media_path,
            inspected_artifact=metadata_artifact,
            content_field_prefixes=("transcript", "frame_ocr"),
        )
        if privacy.rejection_reason is not None:
            return PreprocessingQuarantineRecord.from_input(
                item=item,
                reason=privacy.rejection_reason,
                quality_signals={
                    "privacy_status": privacy.clearance.status.value,
                    "privacy_reasons": list(privacy.clearance.reasons),
                },
            )
        transcript_text = privacy.fields.get("transcript_text")
        frame_ocr_text = privacy.fields.get("frame_ocr_text")
        segments = tuple(
            {
                **segment,
                "text": privacy.fields.get(
                    f"transcript_segment:{index}",
                    str(segment.get("text") or ""),
                ),
            }
            for index, segment in enumerate(segments)
        )
        semantic_text = (
            transcript_text
            or frame_ocr_text
            or _segments_text(segments=segments)
        )
        quality = self._quality_for_valid_item(
            item=item,
            validation=validation,
            semantic_text=semantic_text,
            has_alignment_material=bool(semantic_text or keyframe_paths),
            extra_signals={
                "segment_count": len(segments),
                "keyframe_count": len(keyframe_paths),
                "frame_ocr_available": bool(frame_ocr_text),
                "fps": fps,
            },
        )
        video_structure = canonical_video_structure(item.payload)
        subtitle_segments = normalize_segments(
            item.payload.get("subtitle_segments")
        )
        return PreprocessedVideo(
            media_id=self._media_id(item=item),
            source_id=item.source_id,
            source_url=item.source_url,
            normalized_url=item.normalized_url,
            domain=item.domain,
            media_path=privacy.media_path,
            mime_type=item.mime_type,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            transcript_text=transcript_text,
            transcript_language=item.resolved_language()
            or as_optional_text(item.payload.get("transcript_language")),
            transcript_segments=segments,
            frame_ocr_text=frame_ocr_text,
            keyframe_paths=keyframe_paths,
            quality=quality,
            normalized_video_path=privacy.media_path,
            video_probe_metadata=video_probe_metadata,
            dedupe_fingerprints=self._fingerprints(
                item=item,
                primary_text=semantic_text,
            ),
            alignment_signals={
                "transcript_available": bool(transcript_text),
                "frame_ocr_available": bool(frame_ocr_text),
                "keyframes_available": bool(keyframe_paths),
                **summarize_timeline(
                    segments=segments,
                    duration_seconds=duration_seconds,
                    payload=item.payload,
                ),
                **_video_alignment_summary(
                    payload=item.payload,
                    duration_seconds=duration_seconds,
                ),
            },
            safety_status="passed",
            privacy_clearance=privacy.clearance,
            privacy_evidence={
                "analysis": privacy.analysis_evidence.to_dict(),
                "residual": None,
            },
            timed_segments=canonical_transcript_segments(segments),
            keyframes=video_structure["keyframes"],
            shots=video_structure["shots"],
            object_tracks=video_structure["object_tracks"],
            temporal_events=video_structure["temporal_events"],
            subtitle_segments=canonical_transcript_segments(subtitle_segments),
            privacy_intervals=canonical_privacy_intervals(
                item.payload.get("privacy_intervals")
            ),
        )


def _segments_text(
    *,
    segments: tuple[dict[str, object], ...],
) -> str | None:
    text = " ".join(
        str(segment.get("text") or "") for segment in segments
    ).strip()
    return text or None


def _probe_metadata(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_keyframe_paths(value: object) -> tuple[str, ...]:
    """Return stable, de-duplicated keyframe paths."""

    paths: list[str] = []
    for frame in _normalize_keyframes(value):
        path = frame.get("frame_path")
        if isinstance(path, str):
            paths.append(path)
    return tuple(paths)


def _normalize_keyframes(
    value: object,
) -> tuple[dict[str, object], ...]:
    """Normalize keyframe dictionaries and timing metadata."""

    if not isinstance(value, list):
        return ()

    frames: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for frame in value:
        if not isinstance(frame, dict):
            continue
        path = as_optional_text(frame.get("frame_path"))
        if path is None or path in seen_paths:
            continue
        seen_paths.add(path)
        frames.append(
            {
                "frame_path": path,
                "timestamp_seconds": as_optional_float(
                    frame.get("timestamp_seconds")
                ),
                "ocr_text": as_optional_text(frame.get("ocr_text")),
                "confidence": as_optional_float(frame.get("confidence")),
            }
        )
    return tuple(
        sorted(
            frames,
            key=lambda item: (
                item.get("timestamp_seconds") is None,
                item.get("timestamp_seconds") or 0.0,
                item.get("frame_path") or "",
            ),
        )
    )


class _ResolvedVideoMetadata(TypedDict):
    duration_seconds: float | None
    width: int | None
    height: int | None
    fps: float | None
    keyframe_count: int


def _resolve_video_metadata(
    *,
    item: PreprocessingInput,
    probe: dict[str, object],
) -> _ResolvedVideoMetadata:
    """Resolve duration, dimensions, fps, and keyframe count.

    Item fields and payload metadata take precedence; the probe fills in
    what the item does not carry.
    """

    width_value = item.width or as_optional_float(
        payload_field(item=item, name="video_width")
    )
    width = (
        int(width_value) if width_value is not None else None
    ) or as_optional_int(probe.get("width"))
    height_value = item.height or as_optional_float(
        payload_field(item=item, name="video_height")
    )
    height = (
        int(height_value) if height_value is not None else None
    ) or as_optional_int(probe.get("height"))
    keyframes = item.payload.get("keyframes")
    if isinstance(keyframes, list):
        keyframe_count = len(keyframes)
    else:
        keyframe_count = int(
            as_optional_float(item.payload.get("keyframe_count")) or 0
        )
    return {
        "duration_seconds": (
            item.duration_seconds
            or as_optional_float(
                payload_field(item=item, name="video_duration_seconds")
            )
            or as_optional_float(probe.get("duration"))
        ),
        "width": width,
        "height": height,
        "fps": (
            as_optional_float(payload_field(item=item, name="video_fps"))
            or as_optional_float(probe.get("fps"))
        ),
        "keyframe_count": keyframe_count,
    }


def _has_semantic_text_or_keyframes(
    *,
    item: PreprocessingInput,
    keyframe_count: int,
) -> bool:
    if keyframe_count > 0:
        return True
    for value in (
        item.transcript_text,
        item.ocr_text,
        item.payload.get("transcript_text"),
        item.payload.get("frame_ocr_text"),
        item.payload.get("frame_ocr_preview"),
    ):
        if as_optional_text(value):
            return True
    return False


def _accept_metadata_only_video(
    *,
    item: PreprocessingInput,
    reason: str,
    signals: dict[str, object],
    probed: dict[str, object],
) -> MediaValidationResult | None:
    if reason not in {
        "too_large",
        "decode_failed",
        "partial_download",
        "file_not_found",
    }:
        return None
    if not is_metadata_fetch_mode(payload=item.payload):
        return None
    if not has_video_training_metadata(
        payload=item.payload,
        transcript_text=item.transcript_text,
        ocr_text=item.ocr_text,
    ):
        return None
    signals.update(
        {
            "metadata_only_accepted": True,
            "metadata_only_reason": reason,
            "decode_checked": probed.get("decode_checked") is True,
        }
    )
    return accepted_media_result(signals=signals)


def _probe_video(
    *,
    path: Path | None,
    reader: VideoReader,
) -> dict[str, object]:
    """Probe the exact local file used by privacy inspection."""

    if path is None or not path.is_file():
        return {"decode_checked": False}
    probed = reader.probe(path)
    if not probed:
        return {"decode_checked": True, "decode_failed": True}
    return {
        "duration": probed.get("duration_seconds"),
        "width": probed.get("width"),
        "height": probed.get("height"),
        "fps": probed.get("fps"),
        "frame_count": probed.get("frame_count"),
        "decode_checked": True,
        "decode_failed": False,
    }


def _video_alignment_summary(
    *,
    payload: dict[str, object],
    duration_seconds: float | None,
) -> dict[str, object]:
    keyframes = _normalize_keyframes(payload.get("keyframes"))
    timestamps = [
        timestamp
        for frame in keyframes
        if (timestamp := as_optional_float(frame.get("timestamp_seconds")))
        is not None
    ]
    scenes = payload.get("scene_boundaries")
    scene_count = len(scenes) if isinstance(scenes, (list, tuple)) else 0
    return {
        "keyframe_count": len(keyframes),
        "timed_keyframe_count": len(timestamps),
        "first_keyframe_seconds": min(timestamps) if timestamps else None,
        "last_keyframe_seconds": max(timestamps) if timestamps else None,
        "keyframe_timeline_ratio": (
            round((max(timestamps) - min(timestamps)) / duration_seconds, 4)
            if len(timestamps) >= 2
            and duration_seconds is not None
            and duration_seconds > 0.0
            else None
        ),
        "scene_boundary_count": scene_count,
        "shot_boundaries_available": scene_count > 0,
        "subtitle_alignment_available": bool(
            payload.get("subtitle_segments")
            or payload.get("subtitle_alignment")
        ),
        "object_tracks_available": bool(payload.get("object_tracks")),
        "privacy_intervals_available": bool(payload.get("privacy_intervals")),
    }
