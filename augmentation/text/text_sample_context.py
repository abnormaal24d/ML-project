"""Resolve canonical augmentation fields from samples and metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmcrawler_datasets.schema import MultimodalSample

_TEXT_SPAN_KEYS = (
    "text",
    "content",
    "body",
    "caption",
    "transcript",
    "ocr_text",
    "document_text",
    "page_text",
    "description",
    "alt_text",
    "value",
)


def sample_title(
    *,
    sample: MultimodalSample,
    metadata: dict[str, object],
) -> str | None:
    return _as_opt_str(metadata.get("title")) or _as_opt_str(sample.title)


def sample_modality(*, sample: MultimodalSample) -> str:
    metadata_modality = _as_opt_str((sample.metadata or {}).get("modality"))
    return metadata_modality or sample.modality


def sample_domain(*, metadata: dict[str, object]) -> str | None:
    return _as_opt_str(metadata.get("domain"))


def sample_task_type(*, metadata: dict[str, object]) -> str | None:
    return _as_opt_str(metadata.get("task_type"))


def sample_text_spans(
    *,
    metadata: dict[str, object],
) -> tuple[str, ...]:
    raw_value = metadata.get("text_spans")
    if not isinstance(raw_value, list):
        return ()
    spans: list[str] = []
    for item in raw_value:
        text = _text_span_value(item)
        if text:
            spans.append(text)
    return tuple(spans)


def _text_span_value(value: object) -> str | None:
    if isinstance(value, str):
        return _as_opt_str(value)
    if not isinstance(value, dict):
        return None
    for key in _TEXT_SPAN_KEYS:
        text = _as_opt_str(value.get(key))
        if text:
            return text
    return None


def _as_opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    return text
