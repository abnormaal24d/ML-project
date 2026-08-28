"""Own rejected-sample report creation and validation for snapshots."""

from __future__ import annotations

import json
from pathlib import Path

PAIR_REJECTIONS_FILENAME = "pair_rejections.jsonl"
REJECTED_ROWS_FILENAME = "rejected_rows.jsonl"
REQUIRED_REJECTED_SAMPLE_REPORTS = (
    PAIR_REJECTIONS_FILENAME,
    REJECTED_ROWS_FILENAME,
)


class RejectedSampleReportError(RuntimeError):
    """Raised when mandatory rejected-sample reports are absent."""


def ensure_rejected_rows_report(*, training_directory: Path) -> Path:
    """Create the generic rejected-row report if the snapshot has none yet."""

    path = training_directory / REJECTED_ROWS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return path


def require_rejected_sample_reports(*, training_directory: Path) -> None:
    """Fail fast when mandatory rejected-sample report files are missing."""

    missing = tuple(
        filename
        for filename in REQUIRED_REJECTED_SAMPLE_REPORTS
        if not (training_directory / filename).is_file()
    )
    if missing:
        raise RejectedSampleReportError(
            "mandatory rejected-sample report missing: " + ", ".join(missing)
        )


def write_pair_rejections(
    *,
    training_directory: Path,
    rows: tuple[dict[str, object], ...],
) -> Path:
    """Persist pair rejection rows as JSONL."""

    path = training_directory / PAIR_REJECTIONS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return path


__all__ = [
    "PAIR_REJECTIONS_FILENAME",
    "REJECTED_ROWS_FILENAME",
    "REQUIRED_REJECTED_SAMPLE_REPORTS",
    "RejectedSampleReportError",
    "ensure_rejected_rows_report",
    "require_rejected_sample_reports",
    "write_pair_rejections",
]
