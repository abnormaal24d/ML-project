"""Trusted in-process content construction for multimodal privacy inspection.

These factories derive inspection inputs from bytes opened by preprocessing
itself. They never consume ``privacy_analysis`` or
``privacy_residual_analysis``. Detector completion and coverage are therefore
local facts rather than caller-supplied claims.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, Sequence

from preprocessing.media.adapters.pillow_image import inspect_image_dimensions
from preprocessing.media.ocr.ocr_engine import OcrEngine
from preprocessing.media.ocr.ocr_result import OcrSpan
from preprocessing.media.ports import FrameProcessor, VideoReader
from preprocessing.preprocessing_input import PreprocessingInput
from preprocessing.privacy.inspection.content_readers.audio_content import (
    AudioContent,
    TranscriptSegment,
)
from preprocessing.privacy.inspection.content_readers.document_content import (
    DocumentContent,
    DocumentPage,
)
from preprocessing.privacy.inspection.content_readers.image_content import (
    ImageContent,
)
from preprocessing.privacy.inspection.content_readers.pdf_text_reader import (
    read_pdf_pages,
)
from preprocessing.privacy.inspection.content_readers.video_content import (
    FrameText,
    VideoContent,
)
from preprocessing.privacy.inspection.detector import VisualRegion
from preprocessing.privacy.inspection.local_visual_analysis import (
    OpenCvVisualPrivacyAnalyzer,
)


class ImagePrivacyContentFactory(Protocol):
    def build(
        self,
        *,
        item: PreprocessingInput,
        media_path: Path,
        metadata: dict[str, str],
        residual: bool,
    ) -> ImageContent: ...


class AudioPrivacyContentFactory(Protocol):
    def build(
        self,
        *,
        item: PreprocessingInput,
        media_path: Path,
        metadata: dict[str, str],
        duration_ms: int,
        transcript_segments: Sequence[dict[str, object]],
        full_decode_completed: bool,
        audio_fingerprint: str | None,
        residual: bool,
    ) -> AudioContent: ...


class VideoPrivacyContentFactory(Protocol):
    def build(
        self,
        *,
        item: PreprocessingInput,
        media_path: Path,
        metadata: dict[str, str],
        duration_ms: int,
        transcript_segments: Sequence[dict[str, object]],
        residual: bool,
    ) -> VideoContent: ...


class DocumentPrivacyContentFactory(Protocol):
    def build(
        self,
        *,
        item: PreprocessingInput,
        normalized_text: str,
        title: str | None,
        metadata: dict[str, str],
    ) -> DocumentContent: ...


class LocalImagePrivacyContentFactory:
    """Run local image decode, OCR and visual observation on exact bytes."""

    def __init__(
        self,
        *,
        ocr_engine: OcrEngine | None,
        visual_analyzer: OpenCvVisualPrivacyAnalyzer,
        max_decode_pixels: int,
    ) -> None:
        self._ocr_engine = ocr_engine
        self._visual_analyzer = visual_analyzer
        self._max_decode_pixels = max_decode_pixels

    def build(
        self,
        *,
        item: PreprocessingInput,
        media_path: Path,
        metadata: dict[str, str],
        residual: bool,
    ) -> ImageContent:
        del residual
        payload = _read_bytes(media_path)
        errors: list[str] = []
        ocr_text: str | None = None
        ocr_spans: tuple[OcrSpan, ...] = ()
        versions: dict[str, str] = {}

        try:
            dimensions = (
                inspect_image_dimensions(
                    path=media_path,
                    max_decode_pixels=self._max_decode_pixels,
                )
                if payload
                else None
            )
        except Exception as exc:
            dimensions = None
            errors.append(f"image_decode_failure:{type(exc).__name__}")
        decode_completed = dimensions is not None
        if not decode_completed:
            errors.append("image_decode_failed")

        ocr_completed = False
        if self._ocr_engine is None:
            errors.append("local_ocr_backend_unavailable")
        elif payload:
            try:
                ocr_result = self._ocr_engine.extract(image_bytes=payload)
            except Exception as exc:  # backend boundary; fail closed
                errors.append(f"local_ocr_failure:{type(exc).__name__}")
            else:
                ocr_completed = True
                if ocr_result is not None:
                    ocr_text = ocr_result.text
                    ocr_spans = ocr_result.spans
                    versions[f"ocr:{ocr_result.engine}"] = (
                        ocr_result.producer_revision
                    )

        try:
            visual = self._visual_analyzer.analyze_bytes(
                payload=payload,
                ocr_text=ocr_text,
            )
        except Exception as exc:
            visual_regions: tuple[VisualRegion, ...] = ()
            visual_completed = False
            visual_uncertainty_flags: tuple[str, ...] = ()
            errors.append(
                f"local_visual_analysis_failure:{type(exc).__name__}"
            )
        else:
            visual_regions = visual.regions
            visual_completed = visual.completed
            versions.update(visual.detector_versions)
            errors.extend(visual.errors)
            visual_uncertainty_flags = getattr(visual, "uncertainty_flags", ())

        return ImageContent(
            subject_bytes=payload,
            ocr_text=ocr_text,
            metadata=dict(metadata),
            visual_regions=visual_regions,
            media_decode_completed=decode_completed,
            ocr_analysis_completed=ocr_completed,
            visual_analysis_completed=visual_completed,
            metadata_analysis_completed=True,
            language=item.resolved_language(),
            country=None,
            detector_versions=versions,
            analysis_errors=tuple(dict.fromkeys(errors)),
            ocr_spans=ocr_spans,
            visual_uncertainty_flags=visual_uncertainty_flags,
        )


class LocalAudioPrivacyContentFactory:
    """Build exact-byte audio evidence and fail closed on missing analyzers.

    Existing transcript enrichment remains scanable content, but it does not
    prove that ASR, speaker, background-speech, or voice analysis ran in this
    preprocessing process. Those checks therefore remain incomplete until a
    trusted local backend is explicitly added.
    """

    def build(
        self,
        *,
        item: PreprocessingInput,
        media_path: Path,
        metadata: dict[str, str],
        duration_ms: int,
        transcript_segments: Sequence[dict[str, object]],
        full_decode_completed: bool,
        audio_fingerprint: str | None,
        residual: bool,
    ) -> AudioContent:
        del residual
        payload = _read_bytes(media_path)
        # Existing transcript enrichment is content, not proof that this run
        # inspected the complete waveform. Until an in-process ASR/speaker
        # backend is injected, all semantic audio checks remain incomplete.
        segments = _transcript_segments(transcript_segments)
        decode_completed = bool(payload) and full_decode_completed
        errors = []
        if not decode_completed:
            errors.append("audio_decode_failed")
        errors.extend(
            (
                "local_asr_backend_unavailable",
                "local_speaker_analysis_backend_unavailable",
                "local_background_speech_backend_unavailable",
                "local_voice_analysis_backend_unavailable",
            )
        )
        return AudioContent(
            subject_bytes=payload,
            duration_ms=duration_ms,
            transcript_segments=segments,
            transcript_checked_ranges_ms=(),
            transcript_analysis_completed=False,
            metadata=dict(metadata),
            metadata_analysis_completed=True,
            language=item.resolved_language(),
            country=None,
            full_decode_completed=decode_completed,
            speaker_analysis_completed=False,
            background_speech_analysis_completed=False,
            voice_analysis_completed=False,
            voice_identity_detected=False,
            voice_identity_authorized=False,
            audio_fingerprint=audio_fingerprint,
            detector_versions={"local_audio_decode": "v1"},
            analysis_errors=tuple(errors),
        )


class LocalVideoPrivacyContentFactory:
    """Decode and inspect every frame locally, subject to a hard safety cap."""

    def __init__(
        self,
        *,
        ocr_engine: OcrEngine | None,
        visual_analyzer: OpenCvVisualPrivacyAnalyzer,
        reader: VideoReader,
        frame_processor: FrameProcessor,
        audio_stream_probe: Callable[[Path], bool | None],
        max_frames: int = 10_000,
    ) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")
        self._ocr_engine = ocr_engine
        self._visual_analyzer = visual_analyzer
        self._reader = reader
        self._frame_processor = frame_processor
        self._audio_stream_probe = audio_stream_probe
        self._max_frames = max_frames

    def build(
        self,
        *,
        item: PreprocessingInput,
        media_path: Path,
        metadata: dict[str, str],
        duration_ms: int,
        transcript_segments: Sequence[dict[str, object]],
        residual: bool,
    ) -> VideoContent:
        del residual
        payload = _read_bytes(media_path)
        digest = hashlib.sha256(payload).hexdigest()
        errors: list[str] = []
        if payload:
            try:
                probe = self._reader.probe(media_path)
            except Exception as exc:
                probe = {}
                errors.append(f"video_probe_failure:{type(exc).__name__}")
        else:
            probe = {}
        frame_count = _positive_int(probe.get("frame_count")) or 0
        fps = _positive_float(probe.get("fps")) or 0.0
        probed_duration = _seconds_to_ms(probe.get("duration_seconds"))
        effective_duration = duration_ms or probed_duration
        versions: dict[str, str] = {}
        frame_text: list[FrameText] = []
        regions: list[VisualRegion] = []
        inspected_count = 0

        if not payload or frame_count <= 0 or fps <= 0.0:
            errors.append("video_decode_failed")
        elif frame_count > self._max_frames:
            errors.append("video_privacy_frame_limit_exceeded")
        elif self._ocr_engine is None:
            errors.append("local_frame_ocr_backend_unavailable")
        else:
            try:
                session = self._reader.open(media_path)
            except Exception as exc:
                session = None
                errors.append(f"video_open_failure:{type(exc).__name__}")
            if session is None:
                errors.append("video_decode_failed")
            else:
                try:
                    for index in range(frame_count):
                        try:
                            ok, frame = session.read_frame()
                        except Exception as exc:
                            errors.append(
                                f"video_frame_read_failure:{type(exc).__name__}"
                            )
                            break
                        if not ok or frame is None:
                            errors.append(f"video_frame_decode_failed:{index}")
                            break
                        timestamp_ms = int((index / fps) * 1000.0)
                        try:
                            encoded = self._frame_processor.encode_jpeg(frame)
                        except Exception as exc:
                            errors.append(
                                f"video_frame_encode_failure:{type(exc).__name__}"
                            )
                            break
                        if not encoded:
                            errors.append(f"video_frame_encode_failed:{index}")
                            break
                        try:
                            ocr_result = self._ocr_engine.extract(
                                image_bytes=encoded
                            )
                        except Exception as exc:
                            errors.append(
                                f"local_frame_ocr_failure:{type(exc).__name__}"
                            )
                            break
                        text = (
                            ocr_result.text if ocr_result is not None else ""
                        )
                        if ocr_result is not None:
                            versions[f"ocr:{ocr_result.engine}"] = (
                                ocr_result.producer_revision
                            )
                        try:
                            visual = self._visual_analyzer.analyze_frame(
                                frame=frame,
                                ocr_text=text,
                                frame_index=index,
                                timestamp_ms=timestamp_ms,
                            )
                        except Exception as exc:
                            errors.append(
                                "local_visual_analysis_failure:"
                                f"{type(exc).__name__}"
                            )
                            break
                        versions.update(visual.detector_versions)
                        if not visual.completed:
                            errors.extend(visual.errors)
                            break
                        regions.extend(visual.regions)
                        frame_text.append(
                            FrameText(
                                frame_index=index,
                                timestamp_ms=timestamp_ms,
                                text=text,
                                phash=hashlib.sha256(encoded).hexdigest(),
                            )
                        )
                        inspected_count += 1
                finally:
                    try:
                        session.close()
                    except Exception as exc:
                        errors.append(
                            f"video_close_failure:{type(exc).__name__}"
                        )

        all_frames_inspected = (
            frame_count > 0 and inspected_count == frame_count and not errors
        )
        full_range = (
            ((0, effective_duration),)
            if all_frames_inspected and effective_duration > 0
            else ()
        )
        uninspected = (
            ()
            if full_range
            else (((0, effective_duration),) if effective_duration > 0 else ())
        )
        # Caller-supplied transcript segments are retained as content only.
        # They cannot prove that the audio track was inspected by this run.
        segments = _transcript_segments(transcript_segments)
        has_audio = self._audio_stream_probe(media_path)
        if has_audio is False:
            audio_complete = True
            transcript_ranges = (
                ((0, effective_duration),) if effective_duration > 0 else ()
            )
        elif has_audio is True:
            audio_complete = False
            transcript_ranges = ()
            errors.append("local_video_audio_privacy_backend_unavailable")
        else:
            audio_complete = False
            transcript_ranges = ()
            errors.append("video_audio_stream_probe_failed")

        # One scene is a conservative lower bound; exhaustive frame inspection
        # means scene selection is not used to skip content.
        scene_count = 1 if all_frames_inspected else 0
        return VideoContent(
            subject_bytes=payload,
            subject_sha256=digest,
            duration_ms=effective_duration,
            decoded_frame_count=frame_count,
            inspected_frame_count=inspected_count,
            scene_count=scene_count,
            transcript_segments=segments,
            transcript_checked_ranges_ms=transcript_ranges,
            frame_text=tuple(frame_text),
            frame_ocr_checked_ranges_ms=full_range,
            visual_regions=tuple(regions),
            checked_video_ranges_ms=full_range,
            uninspected_intervals_ms=uninspected,
            tracking_completed=all_frames_inspected,
            audio_inspection_completed=audio_complete,
            metadata_inspection_completed=True,
            residual_scan_completed=all_frames_inspected,
            detector_versions=versions,
            visual_analysis_completed=all_frames_inspected,
            audio_fingerprint=None,
            metadata=dict(metadata),
            language=item.resolved_language(),
            country=None,
            analysis_errors=tuple(dict.fromkeys(errors)),
        )


class LocalDocumentPrivacyContentFactory:
    """Extract PDF pages locally; plain documents become one exact text page."""

    def build(
        self,
        *,
        item: PreprocessingInput,
        normalized_text: str,
        title: str | None,
        metadata: dict[str, str],
    ) -> DocumentContent:
        media_path = Path(item.media_path) if item.media_path else None
        if media_path is not None and media_path.is_file():
            payload = _read_bytes(media_path)
            try:
                pages = read_pdf_pages(payload)
            except (RuntimeError, OSError, ValueError):
                pages = ()
                expected_page_count = 1
            else:
                expected_page_count = len(pages)
            return DocumentContent(
                subject_bytes=payload,
                title=title,
                pages=pages,
                metadata=dict(metadata),
                language=item.resolved_language(),
                country=None,
                expected_page_count=expected_page_count,
            )

        encoded = normalized_text.encode("utf-8")
        return DocumentContent(
            subject_bytes=encoded,
            title=title,
            pages=(DocumentPage(page_number=1, text=normalized_text),),
            metadata=dict(metadata),
            language=item.resolved_language(),
            country=None,
            expected_page_count=1,
        )


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes() if path.is_file() else b""
    except OSError:
        return b""


def _transcript_segments(
    raw_segments: Sequence[dict[str, object]],
) -> tuple[TranscriptSegment, ...]:
    segments: list[TranscriptSegment] = []
    for raw in raw_segments:
        start = _segment_ms(raw, "start_ms", "start_seconds", "start")
        end = _segment_ms(raw, "end_ms", "end_seconds", "end")
        text = str(raw.get("text") or "").strip()
        if start is None or end is None or end <= start or not text:
            continue
        speaker = raw.get("speaker_id")
        segments.append(
            TranscriptSegment(
                start_ms=start,
                end_ms=end,
                text=text,
                speaker_id=(
                    str(speaker).strip()
                    if isinstance(speaker, str) and speaker.strip()
                    else None
                ),
            )
        )
    return tuple(
        sorted(segments, key=lambda item: (item.start_ms, item.end_ms))
    )


def _segment_ms(
    raw: dict[str, object],
    milliseconds_name: str,
    seconds_name: str,
    fallback_name: str,
) -> int | None:
    value = raw.get(milliseconds_name)
    multiplier = 1.0
    if value is None:
        value = raw.get(seconds_name)
        multiplier = 1000.0
    if value is None:
        value = raw.get(fallback_name)
        multiplier = 1000.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, int(float(value) * multiplier))


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number > 0 else None


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0.0 else None


def _seconds_to_ms(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(float(value) * 1000.0))


__all__ = [
    "AudioPrivacyContentFactory",
    "DocumentPrivacyContentFactory",
    "ImagePrivacyContentFactory",
    "LocalAudioPrivacyContentFactory",
    "LocalDocumentPrivacyContentFactory",
    "LocalImagePrivacyContentFactory",
    "LocalVideoPrivacyContentFactory",
    "VideoPrivacyContentFactory",
]
