"""Release contract settings: tasks, metric floors, and resource limits.

This is the single generic task-requirement schema. Production requirements
live in prod.toml only; there is no separate release configuration layer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from config.base.settings_model import SettingsModel


class MetricReq(SettingsModel):
    """One quality metric with exactly one hard bound."""

    name: str = Field(min_length=1)
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_exactly_one_bound(self) -> MetricReq:
        if (self.min is None) == (self.max is None):
            raise ValueError(
                f"metric {self.name!r} requires exactly one of min/max"
            )
        return self


class TaskReq(SettingsModel):
    """One task requirement: identity, data floor, metric floors.

    Task maturity is governed exclusively by the multimodal task registry;
    this schema does not re-declare it.
    """

    name: str = Field(min_length=1)
    min_samples: int = Field(default=0, ge=0)
    metrics: tuple[MetricReq, ...] = ()


class RuntimeLimits(SettingsModel):
    """Hard runtime resource limits (production requires both)."""

    max_batch_latency_ms: float = Field(default=0.0, ge=0.0)
    max_peak_memory_mb: int = Field(default=0, ge=0)


class CapabilityPolicy(SettingsModel):
    """Explicit out-of-scope capability declaration."""

    capability: str = Field(min_length=1)
    status: Literal["blocked", "research_only"]
    reason: str = Field(min_length=1)


class ReproducibilitySettings(SettingsModel):
    """Multi-run reproducibility contract for one release.

    ``seeds`` are the campaign seeds that must each produce one immutable run
    receipt before release evidence is accepted. ``metric_tolerances`` names
    the only metrics compared for seed-to-seed stability; a metric without an
    explicit tolerance is never implicitly compared.
    """

    seeds: tuple[int, ...] = ()
    require_deterministic_execution: bool = False
    metric_tolerances: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reproducibility(self) -> ReproducibilitySettings:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("reproducibility seeds must be unique")
        for metric, tolerance in self.metric_tolerances.items():
            if tolerance < 0:
                raise ValueError(
                    f"negative reproducibility tolerance for {metric!r}"
                )
        return self


class ReleaseSettings(SettingsModel):
    """Task requirements and runtime limits for one profile."""

    release_id: str | None = None

    tasks: tuple[TaskReq, ...] = ()
    optional_tasks: tuple[str, ...] = ()

    blocked_capabilities: tuple[CapabilityPolicy, ...] = ()

    reproducibility: ReproducibilitySettings = ReproducibilitySettings()

    limits: RuntimeLimits = RuntimeLimits()

    require_benchmark: bool = False
    require_baseline: bool = False
