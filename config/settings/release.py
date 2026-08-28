"""Release contract settings: tasks, metric floors, and resource limits.

This is the single generic task-requirement schema. Production requirements
live in the release settings/profile rather than validation implementation.
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
    """One task requirement: identity, data floor, metric floors."""

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
    """Multi-run reproducibility contract for one release."""

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
    """Task, quality, capacity, and runtime requirements for one profile."""

    release_id: str | None = None
    tasks: tuple[TaskReq, ...] = ()
    optional_tasks: tuple[str, ...] = ()
    blocked_capabilities: tuple[CapabilityPolicy, ...] = ()
    reproducibility: ReproducibilitySettings = Field(
        default_factory=ReproducibilitySettings
    )
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)

    # Production model-capacity policy. Validation consumes these values; it
    # does not own separate threshold constants.
    min_model_fusion_dim: int = Field(default=512, ge=0)
    min_model_projection_dim: int = Field(default=512, ge=0)
    min_model_raw_text_vocab_size: int = Field(default=32768, ge=0)
    min_model_raw_text_max_tokens: int = Field(default=512, ge=0)
    min_model_raw_image_size: int = Field(default=224, ge=0)
    min_model_raw_audio_num_samples: int = Field(default=160000, ge=0)
    min_model_raw_video_frames: int = Field(default=8, ge=0)

    require_benchmark: bool = False
    require_baseline: bool = False
