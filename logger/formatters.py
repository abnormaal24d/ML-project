"""Human-readable and JSON project log formatters."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from .redaction import redact_log_text
from .serializers import (
    extract_record_fields,
    record_event_name,
    record_message,
    render_plain_value,
    serialize_log_value,
    summarize_exception,
)

MESSAGE_ONLY_CONSOLE_EVENTS = frozenset(
    {
        "runtime_dependency_status",
        "runtime_services_ready",
    }
)

MESSAGE_ONLY_CONTEXT_FIELDS = (
    "phase",
    "run_id",
    "crawl_session_id",
)

CONSOLE_CONTEXT_FIELDS = frozenset(
    {
        "workflow_id",
        "root_run_id",
        "parent_run_id",
        "crawl_attempt_id",
        "stage",
        "context_label",
        "logger_root",
        "component_path",
        "component",
        "focus_kinds",
        "workflow_missing_by_kind",
        "coverage_gaps",
    }
)

CONSOLE_LARGE_FIELDS = frozenset(
    {
        "audio_coverage",
        "coverage",
        "dedupe_stats",
        "expected_outputs",
        "missing_artifacts",
        "modality_counts",
        "modality_coverage",
        "rejected_by_reason",
        "required_inputs",
        "top_hosts",
        "validation_summary",
        "video_coverage",
    }
)

_CWD = Path.cwd().resolve()


def _single_line(
    value: object,
) -> str:
    """Collapse embedded newlines and whitespace."""

    return " ".join(str(value).split())


def _component_label(
    logger_name: str,
) -> str:
    """Create compact component label."""

    parts = logger_name.split(".")

    if len(parts) <= 2:
        return logger_name

    return ".".join(parts[-2:])


def _compact_console_value(key: str, value: object) -> object:
    """Shorten noisy values that are still fully available in file logs."""

    if isinstance(value, (list, tuple, set)):
        return _summarize_sequence(key, value)

    if isinstance(value, dict):
        return _summarize_mapping(key, value)

    if not isinstance(value, str):
        return value

    key_parts = key.lower()
    if ("fingerprint" in key_parts or "hash" in key_parts) and len(value) > 16:
        return f"{value[:12]}..."

    if (
        key_parts.endswith("_path")
        or key_parts.endswith("_directory")
        or key_parts in {"path", "directory"}
    ):
        return _compact_path(value)

    if key_parts in {"url", "final_url", "requested_url", "parent_url"}:
        return _compact_url(value)

    return value


def _compact_path(value: str) -> str:
    """Shorten path text for console logs while preserving readability."""

    path = Path(value)
    try:
        resolved = path.resolve()
        compact_path = resolved.relative_to(_CWD).as_posix()
        return compact_path
    except (OSError, ValueError):
        return value.replace("\\", "/")


def _compact_url(value: str) -> str:
    """Shorten URLs for readable console output."""

    parsed = urlsplit(value)
    if not parsed.netloc:
        return value

    path = parsed.path or "/"
    parts = [part for part in path.split("/") if part]
    if len(parts) > 2:
        path = f"/.../{'/'.join(parts[-2:])}"
    elif parts:
        path = "/" + "/".join(parts)

    query = "?..." if parsed.query else ""
    return f"{parsed.scheme}://{parsed.netloc}{path}{query}"


def _summarize_sequence(
    key: str,
    value: Sequence[object] | set[object],
) -> object:
    """
    Summarize large console sequences while keeping small values readable.
    """

    values = (
        tuple(value)
        if not isinstance(value, set)
        else tuple(sorted(value, key=str))
    )
    if key in CONSOLE_LARGE_FIELDS:
        preview = ", ".join(str(item) for item in values[:4])
        suffix = "" if len(values) <= 4 else ", ..."
        return f"{len(values)} items [{preview}{suffix}]"
    return values


def _summarize_mapping(key: str, value: dict[object, object]) -> object:
    """Summarize large mappings for console output."""

    if key in CONSOLE_LARGE_FIELDS or len(value) > 6:
        nonzero = {
            str(item_key): item_value
            for item_key, item_value in value.items()
            if item_value not in (0, None, "", False, {}, [], ())
        }
        if not nonzero:
            return "none"
        preview = ", ".join(
            f"{item_key}={item_value}"
            for item_key, item_value in tuple(nonzero.items())[:4]
        )
        suffix = "" if len(nonzero) <= 4 else ", ..."
        return f"{len(nonzero)} nonzero {{{preview}{suffix}}}"
    return value


def _render_fields(
    fields: dict[str, object],
    *,
    compact_context: bool,
) -> tuple[str, tuple[str, ...]]:
    """Render structured log fields."""

    inline_parts: list[str] = []
    extra_lines: list[str] = []

    for key, value in fields.items():
        if compact_context and key in CONSOLE_CONTEXT_FIELDS:
            continue

        if compact_context:
            value = _compact_console_value(key, value)

        if isinstance(
            value,
            (list, tuple, set),
        ):
            extra_lines.append(f"  {key}: {render_plain_value(value)}")
            continue

        if isinstance(value, dict):
            extra_lines.append(f"  {key}: {render_plain_value(value)}")
            continue

        rendered_value = _single_line(render_plain_value(value))

        inline_parts.append(f"{key}={rendered_value}")

    return (
        " ".join(inline_parts),
        tuple(extra_lines),
    )


class PlainFormatter(logging.Formatter):
    """Compact human-readable console formatter."""

    def __init__(
        self,
        *,
        datefmt: str | None = None,
        compact_context: bool = False,
    ) -> None:
        super().__init__(datefmt=datefmt)
        self._compact_context = compact_context

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        """Render log record."""

        timestamp = self.formatTime(
            record,
            self.datefmt,
        )

        level = f"{record.levelname:<7}"

        component = _component_label(record.name)

        event_name = _single_line(record_event_name(record))

        if self._compact_context:
            phase_line = _render_phase_line(
                timestamp=timestamp,
                level=level,
                component=component,
                event_name=event_name,
                record=record,
            )
            if phase_line is not None:
                return phase_line

        line = f"[{timestamp}] {level} {component} {event_name}"

        fields = extract_record_fields(record)
        explicit_message = record_message(record)
        if (
            self._compact_context
            and explicit_message is not None
            and event_name in MESSAGE_ONLY_CONSOLE_EVENTS
        ):
            fields = {
                key: fields[key]
                for key in MESSAGE_ONLY_CONTEXT_FIELDS
                if key in fields
            }

        rendered_fields, extra_lines = _render_fields(
            fields,
            compact_context=self._compact_context,
        )

        if explicit_message is not None:
            line = f"{line} {_single_line(explicit_message)}"

        if rendered_fields:
            line = f"{line} {rendered_fields}"

        exception_summary = summarize_exception(record)

        if exception_summary:
            extra_lines = (
                *extra_lines,
                (f"  exception: {_single_line(exception_summary)}"),
            )

        if record.stack_info:
            extra_lines = (
                *extra_lines,
                (
                    "  stack: "
                    f"{_single_line(self.formatStack(record.stack_info))}"
                ),
            )

        if not extra_lines:
            return line

        return "\n".join(
            (
                line,
                *extra_lines,
            )
        )


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render one structured log record as JSON."""

        event_name = record_event_name(record)
        message = record_message(record)

        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "event": event_name,
        }

        if message is not None:
            payload["message"] = message

        payload.update(
            {
                key: serialize_log_value(value)
                for key, value in extract_record_fields(record).items()
            }
        )

        if record.exc_info:
            payload["exception"] = redact_log_text(
                self.formatException(record.exc_info)
            )

        if record.stack_info:
            payload["stack"] = redact_log_text(
                self.formatStack(record.stack_info)
            )

        return json.dumps(payload, ensure_ascii=False)


def _render_phase_line(
    *,
    timestamp: str,
    level: str,
    component: str,
    event_name: str,
    record: logging.LogRecord,
) -> str | None:
    """Render compact high-value phase/training events for console output."""

    fields = extract_record_fields(record)
    stage = str(fields.get("action") or fields.get("stage") or "").upper()

    if event_name == "data_workflow_decision_resolved":
        action = str(fields.get("action", "")).upper()
        reason = _workflow_decision_reason_for_console(fields)
        suffix = _decision_context_suffix(fields)
        return f"[{timestamp}] {level} {component} [{action}] {reason}{suffix}"

    if event_name in {
        "data_workflow_completed",
        "data_workflow_completed_successfully",
    }:
        reason = str(fields.get("reason", "success"))
        summary = _compact_done_summary(fields)
        return f"[{timestamp}] {level} {component} [DONE] {reason}{summary}"

    if event_name.endswith("_phase_started") and stage:
        return f"[{timestamp}] {level} {component} [{stage}] started"

    if event_name.endswith("_phase_completed") and stage:
        return f"[{timestamp}] {level} {component} [{stage}] completed"

    if event_name == "crawl_phase_finalized":
        raw_dir = fields.get("raw_run_directory")
        suffix = f" raw={_compact_path(str(raw_dir))}" if raw_dir else ""
        return f"[{timestamp}] {level} {component} [CRAWL] finalized{suffix}"

    if event_name == "multimodal_training_epoch_started":
        epoch = fields.get("epoch")
        epochs = fields.get("epochs")
        train_batches = fields.get("train_batches")
        return (
            f"[{timestamp}] {level} {component} [TRAIN] "
            f"epoch {epoch}/{epochs} started batches={train_batches}"
        )

    if event_name == "multimodal_training_batch_progress":
        epoch = fields.get("epoch")
        epochs = fields.get("epochs")
        batch = fields.get("batch")
        batches_total = fields.get("batches_total")
        samples = fields.get("samples")
        average_loss = _format_metric(fields.get("average_loss"))
        elapsed_seconds = fields.get("elapsed_seconds")
        batches_per_second = fields.get("batches_per_second")
        return (
            f"[{timestamp}] {level} {component} [TRAIN] "
            f"epoch {epoch}/{epochs} batch {batch}/{batches_total} "
            f"samples={samples} avg_loss={average_loss} "
            f"elapsed={elapsed_seconds}s "
            f"batches_per_second={batches_per_second}"
        )

    if event_name == "multimodal_training_epoch_completed":
        epoch = fields.get("epoch")
        epochs = fields.get("epochs")
        train_loss = _format_metric(fields.get("train_loss"))
        val_loss = _format_metric(fields.get("val_loss"))
        test_loss = _format_metric(fields.get("test_loss"))
        batches = fields.get("batches")
        learning_rate = _format_metric(fields.get("learning_rate"))
        trend = fields.get("trend")

        return (
            f"[{timestamp}] {level} {component} [TRAIN] "
            f"epoch {epoch}/{epochs} "
            f"train_loss={train_loss} "
            f"val_loss={val_loss} "
            f"test_loss={test_loss} "
            f"batches={batches} "
            f"lr={learning_rate} "
            f"trend={trend}"
        )

    if event_name == "multimodal_training_completed":
        train_loss = _format_metric(fields.get("train_loss"))
        val_loss = _format_metric(fields.get("val_loss"))
        test_loss = _format_metric(fields.get("test_loss"))
        checkpoint = fields.get("checkpoint_path")
        checkpoint_part = (
            f" checkpoint={_compact_path(str(checkpoint))}"
            if checkpoint
            else ""
        )
        return (
            f"[{timestamp}] {level} {component} [TRAIN] completed "
            f"train={train_loss} val={val_loss} test={test_loss}"
            f"{checkpoint_part}"
        )

    return None


def _workflow_decision_reason_for_console(fields: dict[str, object]) -> str:
    """Render workflow decision reason using compact wording."""

    reason = str(fields.get("reason", ""))
    if reason == "crawl_output_missing" and fields.get("first_run") is True:
        return "crawl_artifacts_not_initialized"
    return reason


def _compact_done_summary(fields: dict[str, object]) -> str:
    """Render a compact workflow-completion summary."""

    parts: list[str] = []
    for key, label in (
        ("raw_output_file_count", "raw"),
        ("curated_document_count", "docs"),
        ("training_sample_count", "samples"),
        ("augmented_sample_count", "augmented"),
        ("physical_artifacts_missing", "missing"),
    ):
        value = fields.get(key)
        if value is not None:
            parts.append(f"{label}={value}")
    return "" if not parts else " " + " ".join(parts)


def _decision_context_suffix(fields: dict[str, object]) -> str:
    """Render compact workflow decision context."""

    parts: list[str] = []
    reason_detail = fields.get("decision_reason_detail")
    if reason_detail:
        parts.append(f"reason={reason_detail}")

    if "first_run" in fields:
        first_run = str(bool(fields["first_run"])).lower()
        parts.append(f"first_run={first_run}")

    suggested_action = fields.get("suggested_action")
    if suggested_action:
        parts.append(f"action={suggested_action}")

    return "" if not parts else " " + " ".join(parts)


def _format_metric(value: object) -> object:
    """Format numeric metrics compactly while preserving missing values."""

    if value is None:
        return None

    if isinstance(value, float):
        return round(value, 6)

    return value
