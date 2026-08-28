"""Video frame OCR service."""

from __future__ import annotations

from collections.abc import Buffer, Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    cast,
)

from preprocessing.media.ocr.ocr_result import (
    OcrLine,
    OcrOrigin,
    OcrSpan,
    OcrWord,
    OpticalCharacterRecognitionResult,
)
from preprocessing.provenance import ProducerProvenance

if TYPE_CHECKING:
    from preprocessing.media.ocr.ocr_engine import OcrEngine
    from preprocessing.media.ports import FrameProcessor

FrameOcrBackend = Callable[
    [bytes],
    OpticalCharacterRecognitionResult | None,
]

# CONSOLIDATED from video/video_frame_ocr_models.py per SRP audit.
# Tiny result model + builder inlined here.


@dataclass(frozen=True, slots=True)
class VideoFrameOcrResult:
    frame_index: int
    timestamp_seconds: float
    text: str
    origin: OcrOrigin
    provenance: ProducerProvenance
    confidence: float | None = None
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    words: tuple[OcrWord, ...] = ()
    lines: tuple[OcrLine, ...] = ()
    frame_path: str | None = None
    engine: str = "unknown"
    language: str | None = None


def build_frame_ocr_results(
    *,
    frames_with_ocr: list[dict[str, Any]],
) -> tuple[VideoFrameOcrResult, ...]:
    """Convert raw frame OCR data to rich results. (CONSOLIDATED)"""
    results = []
    for f in frames_with_ocr:
        text = f.get("text", "") or ""
        if not text.strip():
            continue
        results.append(
            VideoFrameOcrResult(
                frame_index=int(f.get("frame_index", 0)),
                timestamp_seconds=float(f.get("timestamp_seconds", 0.0)),
                text=text,
                origin=f["origin"],
                provenance=f["provenance"],
                confidence=f.get("confidence"),
                boxes=tuple(f.get("boxes", ())),
                words=tuple(f.get("words", [])),
                lines=tuple(f.get("lines", [])),
                frame_path=f.get("frame_path"),
                engine=f.get("engine", "unknown"),
                language=f.get("language"),
            )
        )
    return tuple(results)


_VIDEO_OCR_MAX_FRAMES = 3
_VIDEO_OCR_MAX_SIDE = 960
_VIDEO_OCR_MIN_SCENE_CHANGE_SCORE = 0.35
_VIDEO_OCR_MIN_TEXT_LENGTH = 8


class VideoFrameTextExtractionService:
    """Extract OCR text from sampled video frames via the shared OCR engine."""

    def __init__(
        self,
        *,
        ocr_engine: OcrEngine | None = None,
        frame_processor: FrameProcessor | None = None,
    ) -> None:
        self._ocr_engine = ocr_engine
        self._frame_processor = frame_processor

    def extract(
        self,
        *,
        sampled_frames: list[dict[str, Any]],
    ) -> OpticalCharacterRecognitionResult | None:
        ocr_engine = self._ocr_engine
        frame_processor = self._frame_processor
        if not sampled_frames or ocr_engine is None or frame_processor is None:
            return None
        return _extract_video_frames_with_backend(
            sampled_frames=sampled_frames,
            backend=lambda body: ocr_engine.extract(image_bytes=body),
            frame_processor=frame_processor,
        )

    def extract_if_allowed(
        self,
        *,
        sampled_frames: list[dict[str, Any]],
        run_ocr: bool,
        duration_seconds: object,
        max_duration_seconds: float,
    ) -> OpticalCharacterRecognitionResult | None:
        if not run_ocr or not sampled_frames:
            return None

        if self._duration_exceeds_limit(
            duration_seconds=duration_seconds,
            max_duration_seconds=max_duration_seconds,
        ):
            return None

        return self.extract(sampled_frames=sampled_frames)

    @staticmethod
    def _duration_exceeds_limit(
        *,
        duration_seconds: object,
        max_duration_seconds: float,
    ) -> bool:
        if max_duration_seconds <= 0.0:
            return False

        if duration_seconds is None or isinstance(duration_seconds, bool):
            return False

        if not isinstance(
            duration_seconds,
            (str, bytes, bytearray, int, float),
        ):
            return False

        try:
            return float(duration_seconds) > max_duration_seconds
        except (ValueError, OverflowError):
            return False


def _extract_video_frames_with_backend(
    *,
    sampled_frames: list[dict[str, Any]],
    backend: FrameOcrBackend,
    frame_processor: FrameProcessor,
) -> OpticalCharacterRecognitionResult | None:
    processor = frame_processor
    if not processor.is_available():
        return None

    frames_with_ocr: list[dict[str, Any]] = []
    ocr_candidates = _select_scene_change_frames(
        sampled_frames=sampled_frames,
        max_frames=_VIDEO_OCR_MAX_FRAMES,
    )

    for fallback_index, item in enumerate(ocr_candidates):
        frame = _frame_from_item(item=item, frame_processor=processor)
        if frame is None:
            continue

        try:
            frame = _resize_frame_for_ocr(
                frame=frame,
                max_side=_VIDEO_OCR_MAX_SIDE,
                frame_processor=processor,
            )
            encoded = processor.encode_jpeg(frame)
            if not encoded:
                continue
        except (RuntimeError, OSError, ValueError):
            continue

        result = backend(encoded)
        if (
            result is None
            or len(result.text.strip()) < _VIDEO_OCR_MIN_TEXT_LENGTH
        ):
            continue

        boxes = _boxes_from_ocr_result(result=result)
        frames_with_ocr.append(
            {
                "frame_index": _safe_int(
                    item.get("frame_index"),
                    default=fallback_index,
                ),
                "timestamp_seconds": _safe_float(
                    item.get("timestamp_seconds"),
                    default=float(fallback_index),
                ),
                "text": result.text,
                "origin": result.origin,
                "provenance": result.provenance,
                "confidence": result.confidence,
                "boxes": boxes,
                "words": result.words,
                "lines": result.lines,
                "frame_path": _safe_optional_string(
                    item.get("frame_path"),
                ),
                "engine": result.engine,
                "language": result.language,
            }
        )

    return _build_video_ocr_result(frames_with_ocr=frames_with_ocr)


def _select_scene_change_frames(
    *,
    sampled_frames: list[dict[str, Any]],
    max_frames: int,
) -> list[dict[str, Any]]:
    if not sampled_frames:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in sampled_frames:
        score = _safe_float(
            item.get("scene_change_score"),
            default=0.0,
        )
        if score >= _VIDEO_OCR_MIN_SCENE_CHANGE_SCORE:
            scored.append((score, item))

    if scored:
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:max_frames]]

    return sampled_frames[:max_frames]


def _frame_from_item(
    *,
    item: dict[str, Any],
    frame_processor: FrameProcessor,
) -> Any:
    frame = item.get("frame")
    if frame is not None:
        return frame

    frame_path = item.get("frame_path")
    if frame_path is None:
        return None
    return frame_processor.read_image(str(frame_path))


def _build_video_ocr_result(
    *,
    frames_with_ocr: list[dict[str, Any]],
) -> OpticalCharacterRecognitionResult | None:
    frame_results = build_frame_ocr_results(frames_with_ocr=frames_with_ocr)
    if not frame_results:
        return None

    confidences = [
        result.confidence
        for result in frame_results
        if result.confidence is not None
    ]
    words = tuple(word for result in frame_results for word in result.words)
    lines = tuple(line for result in frame_results for line in result.lines)
    return OpticalCharacterRecognitionResult(
        text=" ".join(result.text for result in frame_results).strip(),
        confidence=(
            sum(confidences) / len(confidences) if confidences else None
        ),
        origin=frame_results[0].origin,
        provenance=frame_results[0].provenance,
        language=next(
            (result.language for result in frame_results if result.language),
            None,
        ),
        lines=lines,
        words=words,
        frame_results=tuple(
            _frame_result_to_dict(result=result) for result in frame_results
        ),
        engine="video_frame_ocr",
    )


def _frame_result_to_dict(
    *,
    result: VideoFrameOcrResult,
) -> dict[str, Any]:
    return {
        "frame_index": result.frame_index,
        "timestamp_seconds": result.timestamp_seconds,
        "text": result.text,
        "confidence": result.confidence,
        "boxes": result.boxes,
        "words": result.words,
        "lines": result.lines,
        "frame_path": result.frame_path,
        "engine": result.engine,
        "language": result.language,
    }


def _boxes_from_ocr_result(
    *,
    result: OpticalCharacterRecognitionResult,
) -> tuple[tuple[float, float, float, float], ...]:
    boxes: list[tuple[float, float, float, float]] = []
    for item in [*result.words, *result.lines]:
        box = _box_from_layout_item(item=item)
        if box is not None:
            boxes.append(box)
    return tuple(boxes)


def _box_from_layout_item(
    *,
    item: OcrSpan,
) -> tuple[float, float, float, float] | None:
    box = item.box
    if box is None or len(box) != 4:
        return None
    return (
        float(box[0]),
        float(box[1]),
        float(box[2]),
        float(box[3]),
    )


def _resize_frame_for_ocr(
    *,
    frame: Any,
    max_side: int,
    frame_processor: FrameProcessor,
) -> Any:
    """Resize large video frames before OCR to avoid ONNX bad allocation."""
    if max_side <= 0:
        return frame

    processor = frame_processor
    if not processor.is_available():
        return frame

    try:
        height, width = frame.shape[:2]
    except (AttributeError, RuntimeError, OSError, ValueError):
        return frame

    longest_side = max(width, height)
    if longest_side <= max_side:
        return frame

    scale = max_side / float(longest_side)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    try:
        resized = processor.resize(
            frame,
            width=new_width,
            height=new_height,
        )
        return frame if resized is None else resized
    except (RuntimeError, OSError, ValueError):
        return frame


def _safe_int(value: object, *, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(cast(str | Buffer | SupportsInt | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: object, *, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(cast(str | Buffer | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
