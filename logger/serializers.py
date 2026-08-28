"""Helpers for turning logger payload values into renderable scalars."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from logger.redaction import redact_log_text, redact_log_value

if TYPE_CHECKING:
    import logging

STANDARD_LOG_RECORD_FIELDS = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "taskName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }
)


def serialize_log_value(value: Any) -> Any:
    """Serialize runtime values into JSON-compatible logger values."""

    value = redact_log_value(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, Path):
        return redact_log_text(str(value))

    if isinstance(value, Mapping):
        return {
            str(key): serialize_log_value(item) for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [serialize_log_value(item) for item in value]

    return str(value)


def render_plain_value(value: Any) -> str:
    """Render one value as a compact single-line log scalar."""

    if isinstance(value, str):
        if any(character.isspace() for character in value):
            return json.dumps(value, ensure_ascii=False)
        return value

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)) or value is None:
        return str(value)

    return json.dumps(serialize_log_value(value), ensure_ascii=False)


def extract_record_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Extract custom event fields from a log record."""

    record_fields = record.__dict__
    field_keys = record_fields.get("_project_field_keys")
    if field_keys is not None:
        return {
            str(key): record_fields[str(key)]
            for key in field_keys
            if not str(key).startswith("_")
            if str(key) in record_fields
        }

    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in STANDARD_LOG_RECORD_FIELDS
        if not key.startswith("_")
    }


def record_event_name(record: logging.LogRecord) -> str:
    """Return the structured event name without formatting message args."""

    value = record.__dict__.get("_project_event")
    if value is not None:
        return redact_log_text(str(value))

    raw_message = str(record.msg)
    return redact_log_text(raw_message or str(record.getMessage()))


def record_message(record: logging.LogRecord) -> str | None:
    """Return the optional human-readable message for a structured event."""

    value = record.__dict__.get("_project_message")
    if value is None:
        return None
    return redact_log_text(str(value))


def summarize_exception(record: logging.LogRecord) -> str:
    """Return a compact one-line exception summary."""

    if not record.exc_info:
        return ""

    error = record.exc_info[1]
    if error is None:
        return "unknown"

    return (
        f"{type(error).__name__}:"
        f"{render_plain_value(redact_log_text(str(error)))}"
    )
