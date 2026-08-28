"""Optimizer update, gradient validation, and clipping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

from training.runtime.loop.state import TrainingLoopState
from training.runtime.precision import GradScaler
from training.runtime.training_batch_processor import (
    raise_for_non_finite_gradients,
)


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    """Observability values measured once for each optimizer update."""

    gradient_norm: float | None
    gradient_was_clipped: bool


def _optimizer_step(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    scheduler_interval: Literal["step", "epoch"],
    grad_scaler: GradScaler | None,
    gradient_clip_max_norm: float | None,
) -> OptimizerStepResult:
    """Apply one update after an accumulated gradient is complete."""

    try:
        if grad_scaler is not None:
            grad_scaler.unscale_(optimizer)
        raise_for_non_finite_gradients(model)
        gradient_norm, was_clipped = _clip_gradients(
            model=model,
            max_norm=gradient_clip_max_norm,
        )
        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None and scheduler_interval == "step":
        scheduler.step()
    return OptimizerStepResult(
        gradient_norm=gradient_norm,
        gradient_was_clipped=was_clipped,
    )


def _clip_gradients(
    *,
    model: Any,
    max_norm: float | None,
) -> tuple[float | None, bool]:
    if max_norm is None:
        return None, False
    if isinstance(max_norm, bool) or float(max_norm) <= 0.0:
        raise ValueError("gradient_clip_max_norm must be positive or null")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(max_norm),
    )
    norm_value = float(gradient_norm.detach().cpu())
    if not math.isfinite(norm_value):
        raise ValueError("non-finite gradient norm encountered")
    return norm_value, norm_value > float(max_norm)


def _record_optimizer_step(
    *,
    state: TrainingLoopState,
    result: OptimizerStepResult,
) -> None:
    state.completed_optimizer_steps += 1
    state.last_gradient_norm = result.gradient_norm
    if result.gradient_was_clipped:
        state.gradient_clip_count += 1


__all__ = ["OptimizerStepResult"]
