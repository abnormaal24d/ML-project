"""Enforce machine-readable production release requirements."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from schemas.release import ReleaseReason, detail

if TYPE_CHECKING:
    from config.settings.root import Settings


class ReleaseConfigurationError(ValueError):
    """Raised when release scope, tasks, or stages are inconsistent."""


@dataclass(frozen=True, slots=True)
class MetricRequirement:
    """One release quality metric with exactly one hard bound."""

    name: str
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ReleaseConfigurationError(
                "metric requirement name must be non-empty"
            )
        if (self.minimum is None) == (self.maximum is None):
            raise ReleaseConfigurationError(
                f"metric {self.name!r} requires exactly one of minimum/maximum"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class TaskRequirement:
    """One release task: identity, data floor, and metric bounds."""

    name: str
    min_samples: int
    metrics: tuple[MetricRequirement, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ReleaseConfigurationError(
                "task requirement name must be non-empty"
            )
        if self.min_samples < 0:
            raise ReleaseConfigurationError(
                f"task {self.name!r} min_samples must be >= 0"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "min_samples": self.min_samples,
            "metrics": [metric.to_payload() for metric in self.metrics],
        }


@dataclass(frozen=True, slots=True)
class BlockedCapability:
    """Explicit out-of-scope capability declaration."""

    capability: str
    production_status: str
    reason: str

    def to_payload(self) -> dict[str, str]:
        return {
            "capability": self.capability,
            "production_status": self.production_status,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ReproducibilityRequirements:
    """Multi-run reproducibility contract for one release.

    ``seeds`` are the campaign seeds that must each produce one immutable run
    receipt before release evidence is accepted. ``metric_tolerances`` names
    the only metrics compared for seed-to-seed stability; a metric without an
    explicit tolerance is never implicitly compared.

    ``policy_id`` is the authoritative release contract identity and
    ``policy_sha256`` is its canonical content digest. Both are bound into
    every reproducibility report, evidence bundle, and release manifest so a
    report produced under one policy can never be accepted under another.
    """

    policy_id: str
    seeds: tuple[int, ...]
    require_deterministic_execution: bool
    metric_tolerances: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ReleaseConfigurationError(
                "reproducibility policy identity must be non-empty"
            )
        if not self.seeds:
            raise ReleaseConfigurationError(
                "reproducibility policy must declare at least one seed"
            )
        if any(isinstance(seed, bool) for seed in self.seeds):
            raise ReleaseConfigurationError(
                "reproducibility policy seeds must be integers"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "seeds": sorted(int(seed) for seed in self.seeds),
            "require_deterministic_execution": (
                self.require_deterministic_execution
            ),
            "metric_tolerances": {
                str(name): float(tolerance)
                for name, tolerance in self.metric_tolerances.items()
            },
        }

    @property
    def policy_sha256(self) -> str:
        """Return the canonical content digest of this reproducibility policy."""

        from schemas.canonical import stable_payload_fingerprint

        return stable_payload_fingerprint(self.to_payload())


@dataclass(frozen=True, slots=True)
class RuntimeLimitRequirement:
    """Hard runtime resource limits declared by the release contract."""

    max_batch_latency_ms: float | None = None
    max_peak_memory_mb: float | None = None

    def is_required(self) -> bool:
        return (
            self.max_batch_latency_ms is not None
            or self.max_peak_memory_mb is not None
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "max_batch_latency_ms": self.max_batch_latency_ms,
            "max_peak_memory_mb": self.max_peak_memory_mb,
        }


@dataclass(frozen=True, slots=True)
class ReleaseRequirements:
    """Single source of truth for one production release scope."""

    release_id: str
    release_stage: str
    required_modalities: tuple[str, ...]
    optional_modalities: tuple[str, ...]
    blocked_modalities: tuple[str, ...]
    required_tasks: tuple[str, ...]
    optional_tasks: tuple[str, ...]
    blocked_capabilities: tuple[BlockedCapability, ...]
    task_requirements: tuple[TaskRequirement, ...] = ()
    runtime_limits: RuntimeLimitRequirement | None = None
    reproducibility: ReproducibilityRequirements | None = None
    require_benchmark: bool = False
    require_baseline: bool = False

    @property
    def allowed_modalities(self) -> frozenset[str]:
        return frozenset(
            (*self.required_modalities, *self.optional_modalities)
        )

    @property
    def allowed_tasks(self) -> frozenset[str]:
        return frozenset((*self.required_tasks, *self.optional_tasks))

    def scope_summary(self) -> dict[str, object]:
        """Return a stable summary for status and model-card consumers."""

        return {
            "release_id": self.release_id,
            "release_stage": self.release_stage,
            "in_scope": {
                "modalities": list(self.required_modalities),
                "optional_modalities": list(self.optional_modalities),
                "tasks": list(self.required_tasks),
                "optional_tasks": list(self.optional_tasks),
            },
            "out_of_scope": [
                capability.to_payload()
                for capability in self.blocked_capabilities
            ],
            "blocked_modalities": list(self.blocked_modalities),
        }


@dataclass(frozen=True, slots=True)
class RequiredTaskEvidence:
    """Evidence a required release task must provide before promotion."""

    task_name: str
    require_sample_builder: bool = True
    require_collation: bool = True
    require_model_output: bool = True
    require_loss: bool = True
    require_metric: bool = True
    require_inference: bool = True

    def to_payload(self) -> dict[str, object]:
        return {
            "task_name": self.task_name,
            "require_sample_builder": self.require_sample_builder,
            "require_collation": self.require_collation,
            "require_model_output": self.require_model_output,
            "require_loss": self.require_loss,
            "require_metric": self.require_metric,
            "require_inference": self.require_inference,
        }


@dataclass(frozen=True, slots=True)
class TaskImplementationEvidence:
    """Observed implementation coverage for one task."""

    task_name: str
    has_definition: bool
    has_sample_builder: bool
    has_target_fields: bool
    has_collation: bool
    has_dataset_coverage: bool
    has_model_output: bool
    has_loss: bool
    has_metric: bool
    has_inference: bool
    maturity_sufficient: bool

    def missing_requirements(
        self,
        required: RequiredTaskEvidence,
    ) -> tuple[str, ...]:
        """Return the evidence labels missing for a release task."""

        missing: list[str] = []
        checks = (
            (
                "definition",
                required.require_sample_builder,
                self.has_definition,
            ),
            (
                "sample_builder",
                required.require_sample_builder,
                self.has_sample_builder,
            ),
            (
                "target_fields",
                required.require_sample_builder,
                self.has_target_fields,
            ),
            (
                "collation",
                required.require_collation,
                self.has_collation,
            ),
            (
                "dataset_coverage",
                required.require_sample_builder,
                self.has_dataset_coverage,
            ),
            (
                "model_output",
                required.require_model_output,
                self.has_model_output,
            ),
            ("loss", required.require_loss, self.has_loss),
            ("metric", required.require_metric, self.has_metric),
            ("inference", required.require_inference, self.has_inference),
            (
                "maturity",
                required.require_model_output,
                self.maturity_sufficient,
            ),
        )
        for label, required_flag, observed in checks:
            if required_flag and not observed:
                missing.append(label)
        return tuple(missing)

    def to_payload(self) -> dict[str, object]:
        return {
            "task_name": self.task_name,
            "has_definition": self.has_definition,
            "has_sample_builder": self.has_sample_builder,
            "has_target_fields": self.has_target_fields,
            "has_collation": self.has_collation,
            "has_dataset_coverage": self.has_dataset_coverage,
            "has_model_output": self.has_model_output,
            "has_loss": self.has_loss,
            "has_metric": self.has_metric,
            "has_inference": self.has_inference,
            "maturity_sufficient": self.maturity_sufficient,
        }


def required_task_evidence(
    release_requirements: ReleaseRequirements,
) -> tuple[RequiredTaskEvidence, ...]:
    """Return the default evidence requirements for every required task."""

    return tuple(
        RequiredTaskEvidence(task_name=task_name)
        for task_name in release_requirements.required_tasks
    )


_RELEASE_REQUIREMENTS_STAGES = frozenset({"candidate", "production_model"})


def release_requirements_from_settings(
    settings: Settings,
) -> ReleaseRequirements | None:
    """Derive the release contract from the validated settings tree.

    No external TOML is loaded; the contract comes entirely from the profile
    settings tree.
    """

    stage = settings.training.release_stage
    if stage not in _RELEASE_REQUIREMENTS_STAGES:
        return None

    release = settings.release
    environment = settings.application.resolved_environment()
    if environment != "prod":
        return None

    required_tasks = tuple(task.name for task in release.tasks)

    task_requirements = tuple(
        TaskRequirement(
            name=task.name,
            min_samples=int(task.min_samples),
            metrics=tuple(
                MetricRequirement(
                    name=metric.name,
                    minimum=metric.min,
                    maximum=metric.max,
                )
                for metric in task.metrics
            ),
        )
        for task in release.tasks
    )

    blocked_capabilities = tuple(
        BlockedCapability(
            capability=policy.capability,
            production_status=(
                "blocked" if policy.status == "blocked" else "research_only"
            ),
            reason=policy.reason,
        )
        for policy in release.blocked_capabilities
    )

    release_id = release.release_id or (
        "production_v1" if stage == "production_model" else "candidate_v1"
    )
    runtime_limits = RuntimeLimitRequirement(
        max_batch_latency_ms=(
            float(release.limits.max_batch_latency_ms)
            if release.limits.max_batch_latency_ms > 0
            else None
        ),
        max_peak_memory_mb=(
            int(release.limits.max_peak_memory_mb)
            if release.limits.max_peak_memory_mb > 0
            else None
        ),
    )
    reproducibility = ReproducibilityRequirements(
        policy_id=release_id,
        seeds=tuple(release.reproducibility.seeds),
        require_deterministic_execution=(
            release.reproducibility.require_deterministic_execution
        ),
        metric_tolerances=dict(release.reproducibility.metric_tolerances),
    )

    return ReleaseRequirements(
        release_id=release_id,
        release_stage=stage,
        required_modalities=tuple(settings.multimodal.enabled_modalities),
        optional_modalities=(),
        blocked_modalities=(),
        required_tasks=required_tasks,
        optional_tasks=tuple(release.optional_tasks),
        blocked_capabilities=blocked_capabilities,
        task_requirements=task_requirements,
        runtime_limits=runtime_limits,
        reproducibility=reproducibility,
        require_benchmark=release.require_benchmark,
        require_baseline=release.require_baseline,
    )


def validate_release_requirements(
    *,
    release_requirements: ReleaseRequirements,
    enabled_modalities: Sequence[str],
    enabled_tasks: Sequence[str],
    active_release_stage: str,
) -> None:
    """Fail closed when runtime config drifts from release requirements."""

    enabled_modality_set = _normalized_name_set(
        enabled_modalities,
        field_name="enabled_modalities",
    )
    enabled_task_set = _normalized_name_set(
        enabled_tasks,
        field_name="enabled_tasks",
    )
    runtime_release_stage = str(active_release_stage or "").strip().lower()
    required_release_stage = release_requirements.release_stage.strip().lower()

    if not runtime_release_stage:
        raise ReleaseConfigurationError(
            "active release stage must not be empty"
        )
    if runtime_release_stage != required_release_stage:
        raise ReleaseConfigurationError(
            "active release stage must exactly match release requirements "
            f"{release_requirements.release_id!r}: "
            f"active={runtime_release_stage!r} "
            f"contract={required_release_stage!r}"
        )

    missing_modalities = sorted(
        set(release_requirements.required_modalities) - enabled_modality_set
    )
    if missing_modalities:
        raise ReleaseConfigurationError(
            f"Required modalities are disabled: {missing_modalities}"
        )

    blocked_enabled = sorted(
        enabled_modality_set & set(release_requirements.blocked_modalities)
    )
    if blocked_enabled:
        raise ReleaseConfigurationError(
            "Blocked modalities are enabled for this release: "
            f"{blocked_enabled}"
        )

    disallowed_modalities = sorted(
        enabled_modality_set - release_requirements.allowed_modalities
    )
    if disallowed_modalities:
        raise ReleaseConfigurationError(
            "Enabled modalities are outside release scope: "
            f"{disallowed_modalities}"
        )

    missing_tasks = sorted(
        set(release_requirements.required_tasks) - enabled_task_set
    )
    if missing_tasks:
        raise ReleaseConfigurationError(
            f"Required tasks are disabled: {missing_tasks}"
        )

    disallowed_tasks = sorted(
        enabled_task_set - release_requirements.allowed_tasks
    )
    if disallowed_tasks:
        raise ReleaseConfigurationError(
            f"Enabled tasks are outside release scope: {disallowed_tasks}"
        )


def check_release_task_requirements(
    *,
    task_requirements: Sequence[TaskRequirement],
    task_counts: Mapping[str, int],
    task_metrics: Mapping[str, Mapping[str, float]],
) -> tuple[str, ...]:
    """Fail closed on release task sample floors and metric bounds.

    ``task_counts`` are the total release/dataset counts per task;
    ``task_metrics`` are the final test-task metrics from the evaluator.
    A metric that is missing, non-finite, below its minimum, or above its
    maximum produces one unique reason.
    """

    reasons: list[str] = []
    for requirement in task_requirements:
        observed_samples = int(task_counts.get(requirement.name, 0))
        if observed_samples < requirement.min_samples:
            reasons.append(
                detail(
                    ReleaseReason.TASK_MIN_SAMPLES,
                    requirement.name,
                    observed_samples,
                    requirement.min_samples,
                )
            )
        task_metrics_for_task = task_metrics.get(requirement.name) or {}
        for metric in requirement.metrics:
            value = task_metrics_for_task.get(metric.name)
            if value is None:
                reasons.append(
                    detail(
                        ReleaseReason.EVALUATION_METRIC_MISSING,
                        requirement.name,
                        metric.name,
                    )
                )
                continue
            if not math.isfinite(float(value)):
                reasons.append(
                    detail(
                        ReleaseReason.EVALUATION_METRIC_NON_FINITE,
                        requirement.name,
                        metric.name,
                        value,
                    )
                )
                continue
            if metric.minimum is not None and value < metric.minimum:
                reasons.append(
                    detail(
                        ReleaseReason.EVALUATION_METRIC_LOW,
                        requirement.name,
                        metric.name,
                        value,
                        metric.minimum,
                    )
                )
            if metric.maximum is not None and value > metric.maximum:
                reasons.append(
                    detail(
                        ReleaseReason.EVALUATION_METRIC_HIGH,
                        requirement.name,
                        metric.name,
                        value,
                        metric.maximum,
                    )
                )
    return tuple(reasons)


def check_release_runtime_limits(
    *,
    runtime_limits: RuntimeLimitRequirement | None,
    max_batch_latency_ms: float | None,
    peak_memory_mb: float | None,
) -> tuple[str, ...]:
    """Fail closed on release runtime limits.

    A declared limit with a missing or non-finite measurement is a
    violation (fail-closed), so a release can never pass just because the
    runtime was not measured.
    """

    if runtime_limits is None or not runtime_limits.is_required():
        return ()

    reasons: list[str] = []
    latency_limit = runtime_limits.max_batch_latency_ms
    if latency_limit is not None:
        if max_batch_latency_ms is None or not math.isfinite(
            float(max_batch_latency_ms)
        ):
            reasons.append(
                detail(
                    ReleaseReason.RUNTIME_MEASUREMENT_MISSING,
                    "max_batch_latency_ms",
                )
            )
        elif max_batch_latency_ms > latency_limit:
            reasons.append(
                detail(
                    ReleaseReason.BATCH_LATENCY_HIGH,
                    max_batch_latency_ms,
                    latency_limit,
                )
            )
    memory_limit = runtime_limits.max_peak_memory_mb
    if memory_limit is not None:
        if peak_memory_mb is None or not math.isfinite(float(peak_memory_mb)):
            reasons.append(
                detail(
                    ReleaseReason.RUNTIME_MEASUREMENT_MISSING,
                    "peak_memory_mb",
                )
            )
        elif peak_memory_mb > memory_limit:
            reasons.append(
                detail(
                    ReleaseReason.PEAK_MEMORY_HIGH,
                    peak_memory_mb,
                    memory_limit,
                )
            )
    return tuple(reasons)


def capability_status_payload(
    release_requirements: ReleaseRequirements,
) -> tuple[dict[str, str], ...]:
    """Return explicit production statuses for blocked capabilities."""

    return tuple(
        capability.to_payload()
        for capability in release_requirements.blocked_capabilities
    )


def _normalized_name_tuple(
    value: object,
    field_name: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip().lower(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(
            dict.fromkeys(
                str(item).strip().lower()
                for item in value
                if str(item).strip()
            )
        )
    raise ReleaseConfigurationError(f"{field_name} must be a string array")


def _normalized_name_set(
    value: object,
    field_name: str,
) -> set[str]:
    return set(_normalized_name_tuple(value, field_name=field_name))
