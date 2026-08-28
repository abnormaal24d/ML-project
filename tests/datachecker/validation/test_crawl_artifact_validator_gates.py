from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.coverage.settings import CoverageSettings
from crawler.coverage.gaps import CoverageGapAnalyzer
from datachecker.inventory.raw_run_inventory import RawInventory
from datachecker.manifests.crawl_state_manifest import CrawlStateManifest
from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus
from datachecker.validation.crawl_artifact_validator import (
    CrawlArtifactValidator,
)
from datachecker.workflow_decision import WorkflowDecisionReason


@dataclass(frozen=True, slots=True)
class _TestEvidence:
    object_records_total: int
    successful_requests_total: int
    quality_score: float


_IDENTITY = {
    "generation_id": "generation-1",
    "workflow_id": "workflow-1",
    "project_fingerprint": "project-fp",
    "config_fingerprint": "config-fp",
    "environment_name": "dev",
    "environment_fingerprint": "env-fp",
    "python_version": "3.13",
    "dependency_lock_fingerprint": "lock-fp",
}


def _state_manifest(
    *,
    status: WorkflowLifecycleStatus = WorkflowLifecycleStatus.COMPLETED,
) -> CrawlStateManifest:
    return CrawlStateManifest(
        **_IDENTITY,
        status=status,
        attempt_id=None,
        started_at=None,
        updated_at=None,
        completed_at=(
            None
            if status is WorkflowLifecycleStatus.RUNNING
            else "2026-07-19T00:00:00+00:00"
        ),
        raw_run_directory=None,
        run_summary_path=None,
        previous_status=None,
        previous_raw_run_directory=None,
        last_successful_completed_at=None,
        last_successful_manifest_path=None,
        error_type=None,
        error_message=None,
    )


def _inventory(
    *,
    modality_counts: dict[str, int] | None = None,
    file_count: int = 1,
) -> RawInventory:
    return RawInventory(
        directory=Path("/runs/run-1"),
        summary_path=Path("/runs/run-1/run_manifest.json"),
        records_path=Path("/runs/run-1/records.jsonl"),
        errors_path=Path("/runs/run-1/errors.jsonl"),
        fingerprint="fingerprint",
        file_count=file_count,
        fetched_url_count=1,
        failed_url_count=0,
        modality_counts=modality_counts or {},
        started_at=None,
        completed_at="2026-07-19T00:00:00+00:00",
        status="completed",
        final=True,
        schema_valid=True,
    )


def _validate(
    validator: CrawlArtifactValidator,
    *,
    inventory: RawInventory,
) -> object:
    return validator.validate(
        manifest=None,
        crawl_state_manifest=_state_manifest(),
        inventory=inventory,
        current_source_registry_hash="registry",
        current_crawl_settings_hash="settings",
    )


def _validator(
    *,
    minima: dict[str, int] | None = None,
    min_objects: int = 0,
    min_requests: int = 0,
    min_quality: float = 0.0,
    coverage: dict[str, int] | None = None,
    evidence: _TestEvidence | None = None,
) -> CrawlArtifactValidator:
    return CrawlArtifactValidator(
        minimum_output_files=1,
        minimum_modality_counts=minima or {},
        minimum_raw_objects_total=min_objects,
        minimum_successful_requests_total=min_requests,
        minimum_quality_score=min_quality,
        selected_coverage_provider=(
            (lambda: coverage) if coverage is not None else None
        ),
        selected_evidence_provider=(
            (lambda: evidence) if evidence is not None else None
        ),
    )


def test_modality_shortfall_reports_coverage_insufficient() -> None:
    validator = _validator(
        minima={"audio": 5},
        coverage={"audio": 4},
    )
    result = _validate(validator, inventory=_inventory())
    assert (
        result.reason
        is WorkflowDecisionReason.RAW_MODALITY_COVERAGE_INSUFFICIENT
    )


def test_selected_object_total_shortfall_reports_total_insufficient() -> None:
    validator = _validator(
        minima={"document": 1},
        min_objects=80,
        coverage={"document": 79},
    )
    result = _validate(validator, inventory=_inventory())
    assert (
        result.reason is WorkflowDecisionReason.RAW_OBJECT_TOTAL_INSUFFICIENT
    )
    assert "selected_total=79" in result.details
    assert "minimum=80" in result.details


def test_successful_requests_shortfall_reports_acquisition_insufficient() -> (
    None
):
    validator = _validator(
        minima={"document": 1},
        min_objects=1,
        min_requests=60,
        coverage={"document": 80},
        evidence=_TestEvidence(
            object_records_total=80,
            successful_requests_total=59,
            quality_score=0.9,
        ),
    )
    result = _validate(validator, inventory=_inventory())
    assert (
        result.reason
        is WorkflowDecisionReason.RAW_ACQUISITION_HEALTH_INSUFFICIENT
    )
    assert "successful_requests_total_low:59/60" in result.details


def test_quality_shortfall_reports_acquisition_insufficient() -> None:
    validator = _validator(
        minima={"document": 1},
        min_objects=1,
        min_requests=1,
        min_quality=0.45,
        coverage={"document": 80},
        evidence=_TestEvidence(
            object_records_total=80,
            successful_requests_total=60,
            quality_score=0.44,
        ),
    )
    result = _validate(validator, inventory=_inventory())
    assert (
        result.reason
        is WorkflowDecisionReason.RAW_ACQUISITION_HEALTH_INSUFFICIENT
    )
    assert any("quality_score_low" in detail for detail in result.details)


def test_all_gates_satisfied_is_up_to_date() -> None:
    validator = _validator(
        minima={"document": 1, "audio": 1},
        min_objects=2,
        min_requests=2,
        min_quality=0.45,
        coverage={"document": 80, "audio": 5},
        evidence=_TestEvidence(
            object_records_total=85,
            successful_requests_total=60,
            quality_score=0.9,
        ),
    )
    result = _validate(validator, inventory=_inventory())
    assert result.reason is WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE


def test_missing_evidence_fails_closed_when_gates_are_active() -> None:
    validator = _validator(
        minima={"document": 1},
        min_objects=1,
        min_requests=60,
        coverage={"document": 80},
    )
    result = _validate(validator, inventory=_inventory())
    assert (
        result.reason
        is WorkflowDecisionReason.RAW_ACQUISITION_HEALTH_INSUFFICIENT
    )
    assert "selected_crawl_evidence_missing_or_invalid" in result.details


def test_missing_evidence_is_up_to_date_when_no_gates_are_active() -> None:
    validator = _validator(
        minima={"document": 1},
        min_objects=1,
        coverage={"document": 80},
    )
    result = _validate(validator, inventory=_inventory())
    assert result.reason is WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE


def test_running_crawl_exposes_live_modality_gaps_before_finalization() -> (
    None
):
    validator = _validator(
        minima={"document": 15, "image": 20, "audio": 5, "video": 5},
    )
    result = validator.validate(
        manifest=None,
        crawl_state_manifest=_state_manifest(
            status=WorkflowLifecycleStatus.RUNNING,
        ),
        inventory=_inventory(
            modality_counts={"document": 2, "image": 5, "audio": 0},
        ),
        current_source_registry_hash="registry",
        current_crawl_settings_hash="settings",
    )

    assert result.reason is WorkflowDecisionReason.CRAWL_OUTPUT_MISSING
    assert "raw_modality_coverage_below_min:document:2/15" in result.details
    assert "raw_modality_coverage_below_min:image:5/20" in result.details
    assert "raw_modality_coverage_below_min:audio:0/5" in result.details
    assert "raw_modality_coverage_below_min:video:0/5" in result.details

    gaps = CoverageGapAnalyzer(
        settings=CoverageSettings(),
    ).gaps_from_validation_errors(result.details)
    assert gaps == {
        "modality:audio": 5,
        "modality:document": 13,
        "modality:image": 15,
        "modality:video": 5,
    }
