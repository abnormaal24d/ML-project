"""OCR normalization and frame text result building for video semantic outputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def _json_safe_value(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _json_safe_dict(value: object) -> dict[str, Any]:
    normalized = _json_safe_value(value)
    return dict(normalized) if isinstance(normalized, dict) else {}


def safe_float(value: object, *, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return default


def joined_text(values: Any) -> str:
    parts: list[str] = []
    if values is None:
        return ""
    if isinstance(values, str):
        if values:
            parts.append(values.strip())
    else:
        try:
            for v in values:
                if v:
                    parts.append(str(v).strip())
        except TypeError:
            if values:
                parts.append(str(values).strip())
    return " ".join(p for p in parts if p)


def _timestamp_inside_scene(
    *,
    timestamp_seconds: float,
    start_seconds: float,
    end_seconds: float,
) -> bool:
    return start_seconds <= timestamp_seconds < end_seconds


def build_frame_ocr_results(*, ocr: Any | None) -> tuple[dict[str, Any], ...]:
    if ocr is None:
        return ()

    per_frame = (
        getattr(ocr, "per_frame", None)
        or getattr(ocr, "frame_results", None)
        or []
    )
    if per_frame:
        return tuple(
            frame_payload
            for frame in per_frame
            if (frame_payload := _json_safe_dict(frame))
        )
    if getattr(ocr, "text", None):
        return (
            {
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "text": ocr.text,
                "confidence": getattr(ocr, "confidence", 0.0),
                "words": [],
                "lines": [],
                "boxes": [],
                "frame_path": None,
                "engine": "video_frame_ocr",
                "language": getattr(ocr, "language", None),
            },
        )
    return ()


def _ocr_text_for_scene(
    *,
    frame_ocr_results: list[dict[str, Any]],
    start_seconds: float,
    end_seconds: float,
) -> str:
    return joined_text(
        item.get("text")
        for item in frame_ocr_results
        if _timestamp_inside_scene(
            timestamp_seconds=safe_float(
                item.get("timestamp_seconds"),
                default=0.0,
            ),
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
    )
