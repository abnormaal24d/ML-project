"""Release artifact presence, integrity, and resource checks."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from schemas.release import ReleaseReason, detail
from training.runtime.checkpoint.io import checkpoint_is_available

if TYPE_CHECKING:
    from config.settings.datasets import DatasetValidatorSettings

_REQUIRED_MODEL_CARD_SECTIONS = (
    "release identity",
    "architecture boundary",
    "intended use",
    "out-of-scope and disabled capabilities",
    "training data and provenance",
    "evaluation and acceptance",
    "limitations and risks",
    "release decision",
)

REQUIRED_REPORTS = (
    "coverage/task_coverage_report.json",
    "coverage/modality_coverage_report.json",
    "coverage/target_quality_report.json",
    "coverage/coverage_trend_report.json",
)


def check_checkpoint_and_metrics_target(
    *,
    checkpoint_path: Path,
    metrics_path: Path,
) -> tuple[str, ...]:
    """Validate checkpoint availability and the future metrics target path."""

    if not checkpoint_is_available(checkpoint_path):
        return (ReleaseReason.CHECKPOINT_MISSING,)
    if metrics_path.name == "":
        return (ReleaseReason.METRICS_PATH_INVALID,)
    return ()


def check_required_reports(
    *,
    dataset_root: Path,
    required: tuple[str, ...],
) -> tuple[str, ...]:
    """Return a reason when a mandatory report is absent."""

    missing = tuple(
        rel for rel in required if not (dataset_root / rel).exists()
    )
    return (ReleaseReason.REQUIRED_REPORTS_MISSING,) if missing else ()


def check_dataset_card(*, path: Path) -> tuple[str, ...]:
    """Check that the required dataset card exists."""
    if not path.exists():
        return (ReleaseReason.DATASET_CARD_MISSING,)
    return ()


def check_model_card(*, path: Path) -> tuple[str, ...]:
    """Check that the model card contains all required sections.

    Validates the canonical Markdown artifact produced by the exporter:
    training/export/model_card_template.md -> model_card.md

    Checks:
    - All 8 required ``##`` headings are present
    - Each heading has actual content (not empty)
    - No unresolved template placeholders ``<...>``
    """
    if not path.exists():
        return (ReleaseReason.MODEL_CARD_MISSING,)

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return (
            detail(
                ReleaseReason.RELEASE_ARTIFACT_INVALID,
                path.as_posix(),
                type(exc).__name__,
            ),
        )

    # Parse Markdown ``##`` headings
    sections = _model_card_sections(content)

    # Determine which required sections are missing
    missing: list[str] = []
    for section in _REQUIRED_MODEL_CARD_SECTIONS:
        if section not in sections:
            missing.append(section)

    # Determine which required sections are empty
    empty: list[str] = []
    for section in _REQUIRED_MODEL_CARD_SECTIONS:
        if section in sections and not sections[section].strip():
            empty.append(section)

    # Check for unresolved template placeholders
    placeholders = re.findall(r"<[^>]+>", content)
    unresolved = [p for p in placeholders]

    reasons: list[str] = []
    if missing:
        reasons.append(
            detail(
                ReleaseReason.MODEL_CARD_INCOMPLETE,
                f"missing_sections: {','.join(missing)}",
            )
        )
    if empty:
        reasons.append(
            detail(
                ReleaseReason.MODEL_CARD_INCOMPLETE,
                f"empty_sections: {','.join(empty)}",
            )
        )
    if unresolved:
        reasons.append(
            detail(
                ReleaseReason.MODEL_CARD_INCOMPLETE,
                "unresolved_placeholders",
            )
        )

    return tuple(reasons)


def _model_card_sections(text: str) -> dict[str, str]:
    """Parse ``##`` headings from model card Markdown and return
    ``{heading_key: heading_content}``.

    Only the first paragraph after each heading is captured as content.
    The heading key is normalised: lowercased, ``(required)`` suffix stripped,
    surrounding whitespace removed.
    Empty sections (heading present, no content) are registered with empty
    content so that callers can distinguish ``missing`` from ``empty``.
    """
    headings: dict[str, str] = {}
    current_heading: str | None = None
    current_content: list[str] = []

    for line in text.split("\n"):
        heading_match = re.match(r"^##\s+(.+)$", line)

        if heading_match:
            if current_heading is not None:
                headings[current_heading] = " ".join(current_content).strip()

            key = heading_match.group(1).strip()
            current_heading = key.lower().removesuffix("(required)").strip()
            current_content = []
        else:
            if current_heading is not None:
                current_content.append(line)

    # Save last heading
    if current_heading is not None:
        headings[current_heading] = " ".join(current_content).strip()

    return headings


def check_reproducibility(*, path: Path) -> tuple[str, ...]:
    """Check that the required reproducibility report exists."""
    if not path.exists():
        return (ReleaseReason.REPRODUCIBILITY_MISSING,)
    return ()


def check_cards(
    *,
    dataset_root: Path,
    export_directory: Path,
    report_directory: Path,
    dataset_card_filename: str,
    require_dataset_card: bool,
    require_model_card: bool,
    require_reproducibility: bool,
) -> tuple[str, ...]:
    """Return required card and reproducibility-report reasons."""

    reasons: list[str] = []
    if require_model_card:
        reasons.extend(
            check_model_card(path=export_directory / "model_card.md")
        )
    if require_dataset_card:
        reasons.extend(
            check_dataset_card(path=dataset_root / dataset_card_filename)
        )
    if require_reproducibility:
        reasons.extend(
            check_reproducibility(
                path=report_directory / "reproducibility_report.json"
            )
        )
    return tuple(reasons)


def check_validation(
    *,
    dataset_counts: dict[str, object],
    required: bool,
) -> tuple[str, ...]:
    """Return reasons from a required dataset-validation report."""

    if not required:
        return ()
    reasons: list[str] = []
    if dataset_counts.get("validation_valid") is False:
        reasons.append(ReleaseReason.DATASET_VALIDATION_INVALID)
    raw_errors = dataset_counts.get("validation_errors")
    if isinstance(raw_errors, (list, tuple)):
        reasons.extend(
            detail(ReleaseReason.DATASET_VALIDATION_ERROR, error)
            for error in raw_errors
        )
    return tuple(reasons)


def check_artifacts(
    *,
    settings: DatasetValidatorSettings,
    checkpoint_path: Path,
    dataset_root: Path,
    metrics_path: Path,
    evaluation: dict[str, object],
    dataset_counts: dict[str, object],
) -> tuple[str, ...]:
    """Return every release artifact and report reason."""

    return (
        *check_checkpoint_and_metrics_target(
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
        ),
        *check_required_reports(
            dataset_root=dataset_root,
            required=REQUIRED_REPORTS,
        ),
        *check_validation(
            dataset_counts=dataset_counts,
            required=settings.require_validation_report_clean,
        ),
        *_check_resources(
            dataset_root=dataset_root,
            metrics_path=metrics_path,
            evaluation=evaluation,
            dataset_counts=dataset_counts,
            require_coverage_report=(
                settings.require_coverage_report
                or settings.strict_production_acceptance
            ),
            require_evaluation_report=(
                settings.require_evaluation_report
                or settings.strict_production_acceptance
            ),
            min_active_modalities=settings.min_active_modalities,
            max_batch_latency_ms=settings.max_batch_latency_ms,
            max_peak_memory_mb=settings.max_peak_memory_mb,
        ),
    )


def _check_resources(
    *,
    dataset_root: Path,
    metrics_path: Path,
    evaluation: dict[str, object],
    dataset_counts: dict[str, object],
    require_coverage_report: bool,
    require_evaluation_report: bool,
    min_active_modalities: int,
    max_batch_latency_ms: float | None,
    max_peak_memory_mb: float | None,
) -> tuple[str, ...]:
    """Return runtime resource and supporting-report reasons."""

    reasons: list[str] = []

    if require_coverage_report:
        missing_reports = _missing_coverage_report_paths(
            dataset_root=dataset_root,
        )
        if missing_reports:
            reasons.append(ReleaseReason.COVERAGE_REPORT_MISSING)
            reasons.extend(
                detail(ReleaseReason.COVERAGE_REPORT_MISSING, path)
                for path in missing_reports
            )

    if require_evaluation_report and not _evaluation_report_exists(
        metrics_path=metrics_path,
        evaluation=evaluation,
    ):
        reasons.append(ReleaseReason.EVALUATION_REPORT_MISSING)

    if min_active_modalities > 0:
        active_modalities = active_modality_count(
            dataset_counts=dataset_counts
        )
        if active_modalities < int(min_active_modalities):
            reasons.append(
                detail(
                    ReleaseReason.ACTIVE_MODALITIES_LOW,
                    f"{active_modalities}/{int(min_active_modalities)}",
                )
            )

    latency = _optional_float(evaluation.get("max_batch_latency_ms"))
    if (
        latency is not None
        and max_batch_latency_ms is not None
        and latency > max_batch_latency_ms
    ):
        reasons.append(ReleaseReason.BATCH_LATENCY_HIGH)

    memory = _optional_float(evaluation.get("peak_memory_mb"))
    if (
        memory is not None
        and max_peak_memory_mb is not None
        and memory > max_peak_memory_mb
    ):
        reasons.append(ReleaseReason.PEAK_MEMORY_HIGH)

    return tuple(reasons)


def _missing_coverage_report_paths(*, dataset_root: Path) -> tuple[str, ...]:
    """Return mandatory coverage reports absent from the snapshot root."""

    return tuple(
        candidate
        for candidate in REQUIRED_REPORTS
        if not (dataset_root / candidate).is_file()
    )


def _evaluation_report_exists(
    *,
    metrics_path: Path,
    evaluation: dict[str, object],
) -> bool:
    """Return true when evaluation metrics or a sidecar report are present."""

    if evaluation.get("valid") is not None or evaluation.get("task_metrics"):
        return True
    for candidate in (
        metrics_path.with_name("evaluation_report.json"),
        metrics_path.parent / "evaluation_report.json",
    ):
        if candidate.exists():
            return True
    return False


def active_modality_count(*, dataset_counts: dict[str, object]) -> int:
    modalities = dataset_counts.get("modalities")
    if isinstance(modalities, dict):
        return sum(
            1 for value in modalities.values() if _nonnegative_int(value) > 0
        )

    coverage_payload = dataset_counts.get("modality_coverage")
    if isinstance(coverage_payload, dict):
        active = coverage_payload.get("active_modality_count")
        if active is not None:
            return _nonnegative_int(active)
    return 0


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0
