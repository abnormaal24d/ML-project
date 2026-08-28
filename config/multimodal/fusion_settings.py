"""Fusion, runtime, and modality-routing settings."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from config.base.settings_model import SettingsModel


class RuntimeSettings(SettingsModel):
    """Inference runtime settings, including latency objectives."""

    profile: Literal[
        "dev_cpu",
        "gpu_train",
        "multi_gpu_pretrain",
        "realtime_inference",
        "research",
    ] = "dev_cpu"
    latency_target_ms: int = Field(default=1000, gt=0)
    streaming_enabled: bool = False
    streaming_chunk_ms: int = Field(default=40, gt=0)
    cache_kv: bool = False
    max_batch_tokens: int = Field(default=4096, gt=0)


class ModalityRoutingSettings(SettingsModel):
    """Optional task-to-modality overrides over the registry defaults.

    Canonical schemas live in ``multimodal.tasks.TASKS``. These maps only
    override registry inputs/outputs when a deployment needs a temporary
    deviation.
    """

    strict_required_inputs: bool = True
    task_input_overrides: dict[str, tuple[str, ...]] = Field(
        default_factory=dict
    )
    task_output_overrides: dict[str, tuple[str, ...]] = Field(
        default_factory=dict
    )


class FusionSettings(SettingsModel):
    """Multimodal fusion dimension and runtime routing."""

    fusion_dim: int = Field(default=256, gt=0)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    modality_routing: ModalityRoutingSettings = Field(
        default_factory=ModalityRoutingSettings
    )
