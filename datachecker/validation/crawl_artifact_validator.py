"""Validation for raw crawl workflow artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from datachecker.manifests.workflow_lifecycle import WorkflowLifecycleStatus
from datachecker.validation.acquisition_evidence import (
    CrawlAcquisitionEvidence,
)
from datachecker.validation.raw_coverage_validator import RawCoverageValidator
from datachecker.validation.shared_validation import ArtifactPathPresence
from datachecker.workflow_decision import (
    ValidationResult,
    WorkflowDecisionReason,
)

if TYPE_CHECKING:
    from datachecker.inventory.raw_run_inventory import RawInventory
    from datachecker.manifests.crawl_manifest import CrawlManifest
    from datachecker.manifests.crawl_state_manifest import CrawlStateManifest


class CrawlArtifactValidator:
    """Validate crawl manifest state against current crawl inputs."""

    def __init__(
        self,
        *,
        minimum_output_files: int,
        minimum_modality_counts: dict[str, int],
        minimum_raw_objects_total: int = 0,
        minimum_successful_requests_total: int = 0,
        minimum_quality_score: float = 0.0,
        selected_coverage_provider: (
            Callable[[], dict[str, int]] | None
        ) = None,
        selected_evidence_provider: (
            Callable[[], CrawlAcquisitionEvidence | None] | None
        ) = None,
    ) -> None:
        """Initialize with the minimum number of expected raw output files."""

        self._minimum_output_files = minimum_output_files
        self._minimum_modality_counts = {
            str(kind).strip().lower(): max(0, int(minimum))
            for kind, minimum in minimum_modality_counts.items()
            if str(kind).strip()
        }
        self._minimum_raw_objects_total = max(
            0, int(minimum_raw_objects_total)
        )
        self._minimum_successful_requests_total = max(
            0, int(minimum_successful_requests_total)
        )
        self._minimum_quality_score = max(0.0, float(minimum_quality_score))
        self._raw_validator = RawCoverageValidator(
            minimum_modality_counts=self._minimum_modality_counts,
        )
        self._selected_coverage_provider = selected_coverage_provider
        self._selected_evidence_provider = selected_evidence_provider

    def _selected_coverage_counts(
        self,
        *,
        inventory: RawInventory,
    ) -> dict[str, int]:
        """Return the raw-run selection that preprocessing will consume.

        Falls back to the current canonical run when no selection provider
        is wired or when no final raw runs are discoverable yet.
        """

        if self._selected_coverage_provider is not None:
            selected = self._selected_coverage_provider()
            if isinstance(selected, Mapping) and selected:
                return {
                    str(kind).strip().lower(): max(0, int(count))
                    for kind, count in selected.items()
                    if str(kind).strip()
                }
        return dict(inventory.modality_counts)

    def _selected_corpus_gate_result(
        self,
        *,
        inventory: RawInventory,
    ) -> ValidationResult | None:
        """Check modality minima, selected total objects, and acquisition health.

        The crawl-output gate is split into three judgments: modality
        coverage and the total selected corpus size (both derived from the
        same selected coverage provider), and acquisition-health evidence
        (successful requests and quality signal aggregated over the
        selected runs). Returns ``None`` when every gate is satisfied.
        """

        selected_counts = self._selected_coverage_counts(inventory=inventory)
        coverage_result = self._raw_validator.validate_counts(
            counts=selected_counts,
        )
        if coverage_result.errors:
            return ValidationResult.invalid(
                reason=(
                    WorkflowDecisionReason.RAW_MODALITY_COVERAGE_INSUFFICIENT
                ),
                details=coverage_result.errors,
            )

        selected_total = sum(selected_counts.values())
        if selected_total < self._minimum_raw_objects_total:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.RAW_OBJECT_TOTAL_INSUFFICIENT,
                details=(
                    "selected raw corpus is below the raw object minimum",
                    f"selected_total={selected_total}",
                    f"minimum={self._minimum_raw_objects_total}",
                ),
            )

        evidence = self._selected_evidence(inventory=inventory)
        if evidence is None:
            if (
                self._minimum_successful_requests_total > 0
                or self._minimum_quality_score > 0.0
            ):
                return ValidationResult.invalid(
                    reason=(
                        WorkflowDecisionReason.RAW_ACQUISITION_HEALTH_INSUFFICIENT
                    ),
                    details=("selected_crawl_evidence_missing_or_invalid",),
                )
            return None
        acquisition_errors: list[str] = []
        if (
            evidence.successful_requests_total
            < self._minimum_successful_requests_total
        ):
            acquisition_errors.append(
                "successful_requests_total_low:"
                f"{evidence.successful_requests_total}/"
                f"{self._minimum_successful_requests_total}"
            )
        if evidence.quality_score < self._minimum_quality_score:
            acquisition_errors.append(
                f"quality_score_low:{evidence.quality_score:.3f}/"
                f"{self._minimum_quality_score:.3f}"
            )
        if acquisition_errors:
            return ValidationResult.invalid(
                reason=(
                    WorkflowDecisionReason.RAW_ACQUISITION_HEALTH_INSUFFICIENT
                ),
                details=tuple(acquisition_errors),
            )
        return None

    def _live_inventory_coverage_errors(
        self,
        *,
        inventory: RawInventory,
    ) -> tuple[str, ...]:
        """Return modality gaps observable from the current raw run.

        The selected-corpus gate deliberately works from finalized crawl
        selection. During an active first crawl that selection may not exist
        yet, while the raw inventory already exposes useful modality counts.
        Preserve those errors alongside ``crawl_output_missing`` so the
        DataChecker can produce coverage focus for the next crawl attempt.
        """

        return self._raw_validator.validate_counts(
            counts=inventory.modality_counts,
        ).errors

    def _missing_output_result(
        self,
        *,
        inventory: RawInventory,
        details: tuple[str, ...],
    ) -> ValidationResult:
        """Report an unfinalized output without hiding its live modality gaps."""

        return ValidationResult.invalid(
            reason=WorkflowDecisionReason.CRAWL_OUTPUT_MISSING,
            details=(
                *details,
                *self._live_inventory_coverage_errors(inventory=inventory),
            ),
        )

    def _selected_evidence(
        self,
        *,
        inventory: RawInventory,
    ) -> CrawlAcquisitionEvidence | None:
        del inventory
        if self._selected_evidence_provider is None:
            return None
        evidence = self._selected_evidence_provider()
        if not isinstance(evidence, CrawlAcquisitionEvidence):
            return None
        return evidence

    def validate(
        self,
        *,
        manifest: CrawlManifest | None,
        crawl_state_manifest: CrawlStateManifest | None,
        inventory: RawInventory,
        current_source_registry_hash: str,
        current_crawl_settings_hash: str,
    ) -> ValidationResult:
        """Validate crawl output existence, hashes, and minimum file count."""

        if manifest is None:
            return self._validate_from_crawl_state(
                crawl_state_manifest=crawl_state_manifest,
                inventory=inventory,
            )
        artifact_validation = self._validate_manifest_artifacts(
            manifest=manifest,
            inventory=inventory,
        )
        if artifact_validation is not None:
            return artifact_validation
        if manifest.source_registry_hash != current_source_registry_hash:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.CRAWL_SOURCES_CHANGED,
                details=(
                    "source registry or seed configuration changed",
                    f"stored={manifest.source_registry_hash}",
                    f"current={current_source_registry_hash}",
                ),
            )
        if manifest.crawl_settings_hash != current_crawl_settings_hash:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.CRAWL_SETTINGS_CHANGED,
                details=(
                    "crawl settings fingerprint changed",
                    f"stored={manifest.crawl_settings_hash}",
                    f"current={current_crawl_settings_hash}",
                ),
            )
        if inventory.file_count < self._minimum_output_files:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.CRAWL_OUTPUT_INVALID,
                details=(
                    "crawl output file count is below the required minimum",
                    f"minimum={self._minimum_output_files}",
                    f"current={inventory.file_count}",
                ),
            )
        corpus_gate = self._selected_corpus_gate_result(inventory=inventory)
        if corpus_gate is not None:
            return corpus_gate
        if inventory.fingerprint != manifest.output_fingerprint:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.CRAWL_OUTPUT_INVALID,
                details=(
                    "crawl output fingerprint no longer matches stored "
                    "manifest",
                    f"stored={manifest.output_fingerprint}",
                    f"current={inventory.fingerprint}",
                ),
            )
        return ValidationResult.valid(
            reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
        )

    def _validate_manifest_artifacts(
        self,
        *,
        manifest: CrawlManifest,
        inventory: RawInventory,
    ) -> ValidationResult | None:
        if (
            inventory.directory is None
            or inventory.summary_path is None
            or inventory.records_path is None
            or inventory.errors_path is None
        ):
            return self._missing_output_result(
                inventory=inventory,
                details=(
                    "raw crawl output directory, summary, objects, "
                    "or errors manifest is missing",
                ),
            )
        if not inventory.schema_valid:
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.CRAWL_OUTPUT_INVALID,
                details=(
                    "raw crawl run is not a completed final run",
                    f"status={inventory.status}",
                    f"final={inventory.final}",
                    *inventory.raw_schema_errors,
                ),
            )
        if not self._is_completed_final(
            lifecycle_stage="raw",
            status=manifest.status,
            final=manifest.final,
        ):
            return ValidationResult.invalid(
                reason=WorkflowDecisionReason.CRAWL_OUTPUT_INVALID,
                details=(
                    "crawl manifest is not finalized",
                    f"lifecycle_stage={manifest.lifecycle_stage}",
                    f"status={manifest.status}",
                    f"final={manifest.final}",
                ),
            )
        absent_paths = ArtifactPathPresence.missing(
            manifest.raw_run_directory,
            manifest.raw_records_manifest_path,
            manifest.raw_errors_manifest_path,
            manifest.run_summary_path,
        )
        if absent_paths:
            return self._missing_output_result(
                inventory=inventory,
                details=(
                    "crawl manifest references missing physical artifacts",
                    *absent_paths,
                ),
            )
        return None

    @staticmethod
    def _is_completed_final(
        *,
        lifecycle_stage: str,
        status: str,
        final: bool,
    ) -> bool:
        return lifecycle_stage == "raw" and status == "completed" and final

    def _validate_from_crawl_state(
        self,
        *,
        crawl_state_manifest: CrawlStateManifest | None,
        inventory: RawInventory,
    ) -> ValidationResult:
        if crawl_state_manifest is None:
            return self._missing_output_result(
                inventory=inventory,
                details=("top-level crawl manifest is missing",),
            )

        status = crawl_state_manifest.status
        if status is WorkflowLifecycleStatus.COMPLETED:
            corpus_gate = self._selected_corpus_gate_result(
                inventory=inventory,
            )
            if corpus_gate is not None:
                return corpus_gate
            if inventory.schema_valid:
                return ValidationResult.valid(
                    reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
                )

        if status is WorkflowLifecycleStatus.RECOVERING:
            return self._missing_output_result(
                inventory=inventory,
                details=(
                    "interrupted_run_requires_recovery",
                    "crawl_state_manifest reports recovering",
                ),
            )

        if (
            status is WorkflowLifecycleStatus.FAILED
            and str(crawl_state_manifest.error_type or "").strip()
            == "abandoned_interrupted_run"
        ):
            return self._missing_output_result(
                inventory=inventory,
                details=("interrupted_run_was_abandoned",),
            )

        return self._missing_output_result(
            inventory=inventory,
            details=self._missing_manifest_details(
                crawl_state_manifest=crawl_state_manifest,
                inventory=inventory,
            ),
        )

    @staticmethod
    def _missing_manifest_details(
        *,
        crawl_state_manifest: CrawlStateManifest | None,
        inventory: RawInventory,
    ) -> tuple[str, ...]:
        details: list[str] = ["top-level crawl manifest is missing"]

        if crawl_state_manifest is not None:
            status = crawl_state_manifest.status
            if status is WorkflowLifecycleStatus.RUNNING:
                details.append(
                    "last crawl attempt was left in progress before canonical "
                    "finalization"
                )
            elif status is WorkflowLifecycleStatus.RECOVERING:
                details.append("last crawl attempt requires reconciliation")
            elif status is WorkflowLifecycleStatus.CANCELLED:
                details.append(
                    "last crawl attempt was cancelled before the canonical "
                    "manifest was written"
                )
            elif status is WorkflowLifecycleStatus.FAILED:
                details.append(
                    "last crawl attempt failed before the canonical manifest "
                    "was written"
                )

            if crawl_state_manifest.last_successful_completed_at:
                details.append(
                    "a previous crawl was finalized successfully before this "
                    "missing manifest state"
                )

        if (
            inventory.directory is not None
            or inventory.summary_path is not None
        ):
            details.append(
                "latest raw crawl output exists but finalization did not "
                "complete"
            )
            details.extend(inventory.raw_schema_errors)

        return tuple(details)
