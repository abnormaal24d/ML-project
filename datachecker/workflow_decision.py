"""Workflow decision types and logic for the next pipeline step."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class WorkflowAction(StrEnum):
    """Canonical workflow actions used in DataChecker execution plans."""

    NOOP = "noop"
    CRAWL = "crawl"
    PREPROCESS = "preprocess"
    AUGMENT = "augment"
    TRAIN = "train"
    BLOCKED = "blocked"


class WorkflowDecisionReason(StrEnum):
    """Machine-readable reasons for workflow decisions and phase results."""

    CRAWL_OUTPUT_MISSING = "crawl_output_missing"
    CRAWL_OUTPUT_INVALID = "crawl_output_invalid"
    RAW_MODALITY_COVERAGE_INSUFFICIENT = "raw_modality_coverage_insufficient"
    RAW_OBJECT_TOTAL_INSUFFICIENT = "raw_object_total_insufficient"
    RAW_ACQUISITION_HEALTH_INSUFFICIENT = "raw_acquisition_health_insufficient"
    CRAWL_SETTINGS_CHANGED = "crawl_settings_changed"
    CRAWL_SOURCES_CHANGED = "crawl_sources_changed"

    PREPROCESSING_OUTPUT_MISSING = "preprocessing_output_missing"
    PREPROCESSING_OUTPUT_INVALID = "preprocessing_output_invalid"
    PREPROCESSING_SETTINGS_CHANGED = "preprocessing_settings_changed"
    PREPROCESSING_INPUT_CHANGED = "preprocessing_input_changed"

    AUGMENTATION_OUTPUT_MISSING = "augmentation_output_missing"
    AUGMENTATION_OUTPUT_INVALID = "augmentation_output_invalid"
    AUGMENTATION_SETTINGS_CHANGED = "augmentation_settings_changed"
    AUGMENTATION_INPUT_CHANGED = "augmentation_input_changed"

    TRAINING_OUTPUT_MISSING = "training_output_missing"
    TRAINING_OUTPUT_FOR_SELECTED_DATASET_MISSING = (
        "training_output_for_selected_dataset_missing"
    )
    TRAINING_OUTPUT_INVALID = "training_output_invalid"
    TRAINING_SNAPSHOT_INVALID = "training_snapshot_invalid"
    TRAINING_SNAPSHOT_VALIDATION_FAILED = "training_snapshot_validation_failed"
    TRAINING_SETTINGS_CHANGED = "training_settings_changed"
    TRAINING_INPUT_CHANGED = "training_input_changed"

    CONFIG_NO_SEED_URLS = "config_no_seed_urls"
    CONFIG_AUGMENTATION_REQUIRED_BUT_DISABLED = (
        "config_augmentation_required_but_disabled"
    )
    CONFIG_TRAINING_INPUT_UNAVAILABLE = "config_training_input_unavailable"
    WORKFLOW_STATE_INCONSISTENT = "workflow_state_inconsistent"
    LAST_PHASE_FAILED = "last_phase_failed"

    PREPROCESSING_BLOCKED_BY_CRAWL = "preprocessing_blocked_by_crawl"
    AUGMENTATION_BLOCKED_BY_PREPROCESSING = (
        "augmentation_blocked_by_preprocessing"
    )
    TRAINING_BLOCKED_BY_DATASET_SELECTION = (
        "training_blocked_by_dataset_selection"
    )

    COVERAGE_TARGETS_NOT_MET = "coverage_targets_not_met"
    WORKFLOW_IS_UP_TO_DATE = "workflow_is_up_to_date"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result contract consumed by the workflow decision rules."""

    is_valid: bool
    reason: WorkflowDecisionReason
    details: tuple[str, ...] = ()

    @classmethod
    def valid(
        cls,
        *,
        reason: WorkflowDecisionReason,
        details: tuple[str, ...] = (),
    ) -> ValidationResult:
        return cls(is_valid=True, reason=reason, details=details)

    @classmethod
    def invalid(
        cls,
        *,
        reason: WorkflowDecisionReason,
        details: tuple[str, ...] = (),
    ) -> ValidationResult:
        return cls(is_valid=False, reason=reason, details=details)


@dataclass(frozen=True, slots=True)
class WorkflowExecutionPlan:
    """Complete DataChecker-owned plan for the next workflow step."""

    action: WorkflowAction
    reason: WorkflowDecisionReason

    raw_run_directory: Path | None = None
    raw_records_manifest_path: Path | None = None

    training_snapshot_id: str | None = None
    training_root: Path | None = None
    dataset_manifest_hash: str | None = None

    coverage_gaps: dict[str, int] = field(default_factory=dict)
    details: tuple[str, ...] = ()


_PREPROCESSING_REASONS_ROUTED_TO_TRAINING: frozenset[
    WorkflowDecisionReason
] = frozenset(
    {
        WorkflowDecisionReason.TRAINING_SNAPSHOT_VALIDATION_FAILED,
        WorkflowDecisionReason.TRAINING_OUTPUT_INVALID,
        WorkflowDecisionReason.TRAINING_SNAPSHOT_INVALID,
    }
)


@dataclass(frozen=True, slots=True)
class _WorkflowPlanContext:
    """Inputs copied unchanged into every plan produced by one decision."""

    raw_run_directory: Path | None
    raw_records_manifest_path: Path | None
    training_snapshot_id: str | None
    training_root: Path | None
    dataset_manifest_hash: str | None
    coverage_gaps: dict[str, int]


def decide_workflow_action(
    *,
    crawl: ValidationResult,
    preprocessing: ValidationResult,
    augmentation: ValidationResult,
    training: ValidationResult,
    ordered_actions: tuple[WorkflowAction, ...],
    optional_actions: tuple[WorkflowAction, ...],
    seed_url_count: int,
    require_seed_urls: bool,
    augmentation_enabled: bool,
    training_input_mode_is_augmented_required: bool,
    raw_run_directory: Path | None = None,
    raw_records_manifest_path: Path | None = None,
    training_snapshot_id: str | None = None,
    training_root: Path | None = None,
    dataset_manifest_hash: str | None = None,
    coverage_gaps: dict[str, int] | None = None,
) -> WorkflowExecutionPlan:
    """Evaluate decision rules in their documented precedence to select the next action."""

    results = {
        WorkflowAction.CRAWL: crawl,
        WorkflowAction.PREPROCESS: preprocessing,
        WorkflowAction.AUGMENT: augmentation,
        WorkflowAction.TRAIN: training,
    }
    context = _WorkflowPlanContext(
        raw_run_directory=raw_run_directory,
        raw_records_manifest_path=raw_records_manifest_path,
        training_snapshot_id=training_snapshot_id,
        training_root=training_root,
        dataset_manifest_hash=dataset_manifest_hash,
        coverage_gaps={} if coverage_gaps is None else coverage_gaps,
    )
    configuration_plan = _configuration_block_plan(
        context=context,
        seed_url_count=seed_url_count,
        require_seed_urls=require_seed_urls,
        augmentation_enabled=augmentation_enabled,
        training_input_mode_is_augmented_required=(
            training_input_mode_is_augmented_required
        ),
    )
    if configuration_plan is not None:
        return configuration_plan

    upstream_plan = _required_upstream_plan(
        context=context,
        results=results,
        ordered_actions=ordered_actions,
        optional_actions=optional_actions,
    )
    if upstream_plan is not None:
        return upstream_plan

    coverage_plan = _coverage_recovery_plan(context=context)
    if coverage_plan is not None:
        return coverage_plan

    training_plan = _required_training_plan(
        context=context,
        results=results,
        ordered_actions=ordered_actions,
        optional_actions=optional_actions,
    )
    if training_plan is not None:
        return training_plan

    return _build_plan_with_constraints(
        context=context,
        action=WorkflowAction.NOOP,
        reason=WorkflowDecisionReason.WORKFLOW_IS_UP_TO_DATE,
        details=(),
    )


def _configuration_block_plan(
    *,
    context: _WorkflowPlanContext,
    seed_url_count: int,
    require_seed_urls: bool,
    augmentation_enabled: bool,
    training_input_mode_is_augmented_required: bool,
) -> WorkflowExecutionPlan | None:
    """Return a fail-closed plan for impossible configured workflows."""
    if require_seed_urls and seed_url_count < 1:
        return _build_plan_with_constraints(
            context=context,
            action=WorkflowAction.BLOCKED,
            reason=WorkflowDecisionReason.CONFIG_NO_SEED_URLS,
            details=("sources.active.seed_urls is empty",),
        )
    if training_input_mode_is_augmented_required and not augmentation_enabled:
        return _build_plan_with_constraints(
            context=context,
            action=WorkflowAction.BLOCKED,
            reason=(
                WorkflowDecisionReason.CONFIG_AUGMENTATION_REQUIRED_BUT_DISABLED
            ),
            details=(
                "training_input_mode=augmented_required",
                "augmentation.enabled is false",
            ),
        )
    return None


def _required_upstream_plan(
    *,
    context: _WorkflowPlanContext,
    results: dict[WorkflowAction, ValidationResult],
    ordered_actions: tuple[WorkflowAction, ...],
    optional_actions: tuple[WorkflowAction, ...],
) -> WorkflowExecutionPlan | None:
    """Return the first invalid required non-training phase in configured order."""
    for action in ordered_actions:
        if action is WorkflowAction.TRAIN or action in optional_actions:
            continue
        result = results.get(action)
        if result is None or result.is_valid:
            continue
        selected_action: WorkflowAction = action
        if (
            selected_action is WorkflowAction.PREPROCESS
            and result.reason in _PREPROCESSING_REASONS_ROUTED_TO_TRAINING
        ):
            selected_action = WorkflowAction.TRAIN
        return _build_plan_with_constraints(
            context=context,
            action=selected_action,
            reason=result.reason,
            details=result.details,
        )
    return None


def _coverage_recovery_plan(
    *, context: _WorkflowPlanContext
) -> WorkflowExecutionPlan | None:
    """Schedule a crawl when the raw modality coverage target is not met."""
    if not context.coverage_gaps:
        return None
    details = tuple(
        f"need:{name}+{missing}"
        for name, missing in sorted(context.coverage_gaps.items())
    )
    return _build_plan_with_constraints(
        context=context,
        action=WorkflowAction.CRAWL,
        reason=WorkflowDecisionReason.COVERAGE_TARGETS_NOT_MET,
        details=details,
    )


def _required_training_plan(
    *,
    context: _WorkflowPlanContext,
    results: dict[WorkflowAction, ValidationResult],
    ordered_actions: tuple[WorkflowAction, ...],
    optional_actions: tuple[WorkflowAction, ...],
) -> WorkflowExecutionPlan | None:
    """Return the training plan when training is required and invalid."""
    if (
        WorkflowAction.TRAIN not in ordered_actions
        or WorkflowAction.TRAIN in optional_actions
    ):
        return None
    result = results[WorkflowAction.TRAIN]
    if result.is_valid:
        return None
    return _build_plan_with_constraints(
        context=context,
        action=WorkflowAction.TRAIN,
        reason=result.reason,
        details=result.details,
    )


def _build_plan_with_constraints(
    *,
    context: _WorkflowPlanContext,
    action: WorkflowAction,
    reason: WorkflowDecisionReason,
    details: tuple[str, ...],
) -> WorkflowExecutionPlan:
    """Apply context-derived availability constraints to the plan."""
    plan_details = list(details)

    if action is WorkflowAction.AUGMENT:
        if context.training_root is None:
            action = WorkflowAction.BLOCKED
            reason = WorkflowDecisionReason.CONFIG_TRAINING_INPUT_UNAVAILABLE
            plan_details.append(
                "selected preprocessed training root is unavailable"
            )

    if action is WorkflowAction.TRAIN:
        if (
            reason
            is WorkflowDecisionReason.TRAINING_BLOCKED_BY_DATASET_SELECTION
        ):
            action = WorkflowAction.BLOCKED
            plan_details.append(
                "selected training dataset does not meet configured minima"
            )
        elif context.training_root is None or not (
            isinstance(context.dataset_manifest_hash, str)
            and bool(context.dataset_manifest_hash.strip())
        ):
            action = WorkflowAction.BLOCKED
            reason = WorkflowDecisionReason.CONFIG_TRAINING_INPUT_UNAVAILABLE
            plan_details.append("selected training dataset is unavailable")

    if action is WorkflowAction.PREPROCESS:
        if (
            context.raw_run_directory is None
            or context.raw_records_manifest_path is None
        ):
            action = WorkflowAction.BLOCKED
            reason = WorkflowDecisionReason.PREPROCESSING_BLOCKED_BY_CRAWL
            plan_details.append("selected raw crawl output is unavailable")

    return WorkflowExecutionPlan(
        action=action,
        reason=reason,
        raw_run_directory=context.raw_run_directory,
        raw_records_manifest_path=context.raw_records_manifest_path,
        training_snapshot_id=context.training_snapshot_id,
        training_root=context.training_root,
        dataset_manifest_hash=context.dataset_manifest_hash,
        coverage_gaps=context.coverage_gaps,
        details=tuple(plan_details),
    )
