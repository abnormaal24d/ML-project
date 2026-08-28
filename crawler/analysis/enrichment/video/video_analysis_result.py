"""Video analysis result DTO.

The analyzer builds this directly. All persisted mapping lives in
video_enrichment_payload.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crawler.analysis.enrichment.video.video_probe_resolver import (
        VideoProbeResult,
    )
    from preprocessing.media.ocr.ocr_result import (
        OpticalCharacterRecognitionResult,
    )
    from preprocessing.media.ports import (
        VideoAudioTrackResult,
        VideoNormalizationResult,
    )
    from preprocessing.media.speech.speaker_diarization_result import (
        SpeakerDiarizationResult,
    )
    from preprocessing.media.speech.transcription_result import (
        TranscriptionResult,
    )


@dataclass(frozen=True, slots=True)
class VideoAnalysisResult:
    """Raw/nested analysis output from VideoAnalyzer.

    Defaults allow simple construction for special cases (embed, head-only, timeout).
    """

    metadata: dict[str, object]
    metadata_status: str

    payload_path: str | None = None
    transcription: TranscriptionResult | None = None
    keyframes: tuple[dict[str, object], ...] = ()
    frame_ocr: OpticalCharacterRecognitionResult | None = None
    frame_ocr_results: tuple[dict[str, object], ...] = ()
    scene_graph: dict[str, object] | None = None
    action_result: dict[str, object] | None = None

    probe_result: VideoProbeResult | None = None

    audio_track_result: VideoAudioTrackResult | None = None
    audio_track_error_type: str | None = None

    speaker_diarization: SpeakerDiarizationResult | None = None
    speaker_diarization_error_type: str | None = None

    normalization_result: VideoNormalizationResult | None = None
    normalization_error_type: str | None = None

    # --- convenience properties for payload/handler consumers ---
    # These delegate to nested structures so callers can use flat names.

    @property
    def transcript_text(self) -> str | None:
        value = (
            self.transcription.text if self.transcription is not None else None
        )
        return str(value) if value is not None else None

    @property
    def transcript_confidence(self) -> float | None:
        return (
            self.transcription.confidence
            if self.transcription is not None
            else None
        )

    @property
    def transcript_source(self) -> str | None:
        if self.transcription is None:
            return None
        return self.transcription.provenance.producer_name

    @property
    def transcript_language(self) -> str | None:
        value = (
            self.transcription.language
            if self.transcription is not None
            else None
        )
        return str(value) if value is not None else None

    @property
    def transcript_segments(self) -> tuple[dict[str, object], ...]:
        segments = self.transcription.segments if self.transcription else ()
        return _object_tuple_as_dicts(segments)

    @property
    def transcription_status(self) -> str:
        value = (
            self.transcription.transcription_status
            if self.transcription is not None
            else None
        )
        return str(value) if value is not None else "not_run"

    @property
    def transcription_provenance(self) -> dict[str, object] | None:
        if self.transcription is None:
            return None
        payload = self.transcription.provenance.to_dict()
        return dict(payload) if isinstance(payload, dict) else None

    @property
    def frame_ocr_text(self) -> str | None:
        value = self.frame_ocr.text if self.frame_ocr is not None else None
        return str(value) if value is not None else None

    @property
    def duration_seconds(self) -> float | None:
        return _optional_float(self.metadata.get("duration_seconds"))

    @property
    def width(self) -> int | None:
        return _optional_int(self.metadata.get("width"))

    @property
    def height(self) -> int | None:
        return _optional_int(self.metadata.get("height"))

    @property
    def fps(self) -> float | None:
        return _optional_float(self.metadata.get("fps"))

    @property
    def frame_count(self) -> int | None:
        return _optional_int(self.metadata.get("frame_count"))

    # action / motion from action_result (dict or object)
    @property
    def action_segments(self) -> tuple[dict[str, object], ...]:
        return _object_tuple_as_dicts(
            (self.action_result or {}).get("action_segments")
        )

    @property
    def motion_segments(self) -> tuple[dict[str, object], ...]:
        return _object_tuple_as_dicts(
            (self.action_result or {}).get("motion_segments")
        )

    @property
    def action_label(self) -> str | None:
        val = (self.action_result or {}).get("action_label")
        return str(val) if val is not None else None

    @property
    def action_analysis_status(self) -> str | None:
        val = (self.action_result or {}).get("status")
        return str(val) if val is not None else None

    @property
    def action_analysis_reasons(self) -> tuple[str, ...]:
        val = (self.action_result or {}).get("analysis_reasons")
        if isinstance(val, (list, tuple)):
            return tuple(str(x) for x in val)
        return ()

    @property
    def action_proxy_backend(self) -> str | None:
        val = (self.action_result or {}).get("backend")
        return str(val) if val is not None else None

    @property
    def action_recognition_status(self) -> str | None:
        val = (self.action_result or {}).get("action_recognition_status")
        return str(val) if val is not None else None

    @property
    def action_recognition_available(self) -> bool:
        val = (self.action_result or {}).get("action_recognition_available")
        return bool(val)

    # speaker diarization
    @property
    def speaker_diarization_status(self) -> str | None:
        if self.speaker_diarization_error_type:
            return self.speaker_diarization_error_type
        if self.speaker_diarization:
            return "passed"
        return None

    @property
    def speaker_segments(self) -> tuple[dict[str, object], ...]:
        if self.speaker_diarization is None:
            return ()
        return _object_tuple_as_dicts(self.speaker_diarization.segments)

    @property
    def speaker_count(self) -> int:
        if self.speaker_diarization is None:
            return 0
        return self.speaker_diarization.speaker_count

    # normalization
    @property
    def normalized_video_path(self) -> str | None:
        if self.normalization_result is None:
            return None
        return self.normalization_result.normalized_path

    @property
    def video_normalization_status(self) -> str | None:
        if self.normalization_error_type:
            return "failed"
        if self.normalization_result:
            return "passed"
        return None

    # normalized probe result
    @property
    def video_probe_metadata(self) -> dict[str, object] | None:
        if self.probe_result is None:
            return None
        return self.probe_result.video_probe_metadata

    @property
    def video_probe_status(self) -> str | None:
        return (
            self.probe_result.video_probe_status
            if self.probe_result is not None
            else None
        )

    @property
    def video_probe_error_type(self) -> str | None:
        return (
            self.probe_result.video_probe_error_type
            if self.probe_result is not None
            else None
        )


def _optional_float(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_int(value: object | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return None


def _object_tuple_as_dicts(
    value: object | None,
) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()

    raw_items = value if isinstance(value, (list, tuple)) else ()
    items: list[dict[str, object]] = []

    for item in raw_items:
        if isinstance(item, dict):
            items.append(dict(item))
            continue

        if hasattr(item, "__dataclass_fields__"):
            items.append(
                {
                    name: getattr(item, name)
                    for name in item.__dataclass_fields__
                }
            )

    return tuple(items)
