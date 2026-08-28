"""Best-model selection and distributed early stopping."""

from __future__ import annotations

import math
from typing import Any, Literal

import torch.distributed as dist

from training.runtime.loop.state import (
    TrainingLoopState,
    _resume_float,
    _resume_int_value,
)

MonitorMode = Literal["min", "max"]


def is_improvement(
    *,
    current: float,
    best: float | None,
    mode: MonitorMode,
    min_delta: float,
) -> bool:
    """Compare a monitored metric using the shared selection semantics."""

    if not math.isfinite(current):
        raise ValueError("monitored metric must be finite")
    if best is None:
        return True
    if not math.isfinite(best):
        raise ValueError("best monitored metric must be finite")
    if min_delta < 0 or not math.isfinite(min_delta):
        raise ValueError("min_delta must be a finite non-negative value")
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"unsupported monitor_mode: {mode!r}")


def _monitor_value(*, validation_loss: float, settings: Any) -> float:
    metric = settings.monitor_metric
    if metric == "validation_loss":
        return validation_loss
    raise ValueError(f"unsupported monitor_metric: {metric!r}")


def _update_selection_state(
    *,
    state: TrainingLoopState,
    epoch: int,
    current_metric: float,
    settings: Any,
    distributed_context: dict[str, object],
) -> tuple[bool, bool]:
    if distributed_context.get("enabled"):
        return _synchronize_selection_state(
            state=state,
            epoch=epoch,
            current_metric=current_metric,
            settings=settings,
            distributed_context=distributed_context,
        )
    return _apply_selection_update(
        state=state,
        epoch=epoch,
        current_metric=current_metric,
        settings=settings,
    )


def _synchronize_selection_state(
    *,
    state: TrainingLoopState,
    epoch: int,
    current_metric: float,
    settings: Any,
    distributed_context: dict[str, object],
) -> tuple[bool, bool]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "distributed early stopping requires a process group"
        )
    rank = distributed_context.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError("distributed context rank must be an integer")
    payload: list[dict[str, object] | None] = [None]
    if rank == 0:
        improved, should_stop = _apply_selection_update(
            state=state,
            epoch=epoch,
            current_metric=current_metric,
            settings=settings,
        )
        payload[0] = {
            "improved": improved,
            "should_stop": should_stop,
            "best_metric": state.best_metric,
            "best_epoch": state.best_epoch,
            "epochs_without_improvement": state.epochs_without_improvement,
            "stop_reason": state.stop_reason,
        }
    dist.broadcast_object_list(payload, src=0)
    synchronized = payload[0]
    if not isinstance(synchronized, dict):
        raise RuntimeError("rank zero did not broadcast selection state")
    synchronized_improved = synchronized.get("improved")
    synchronized_should_stop = synchronized.get("should_stop")
    best_metric = synchronized.get("best_metric")
    best_epoch = synchronized.get("best_epoch")
    epochs_without_improvement = synchronized.get("epochs_without_improvement")
    stop_reason = synchronized.get("stop_reason")
    if not isinstance(synchronized_improved, bool) or not isinstance(
        synchronized_should_stop,
        bool,
    ):
        raise RuntimeError("distributed selection flags are invalid")
    state.best_metric = (
        _resume_float(best_metric, field="best_metric")
        if best_metric is not None
        else None
    )
    state.best_epoch = (
        _resume_int_value(best_epoch, field="best_epoch")
        if best_epoch is not None
        else None
    )
    if not isinstance(epochs_without_improvement, int):
        raise RuntimeError("distributed patience state is invalid")
    state.epochs_without_improvement = epochs_without_improvement
    if stop_reason is not None and not isinstance(stop_reason, str):
        raise RuntimeError("distributed stop reason is invalid")
    state.stop_reason = stop_reason
    return synchronized_improved, synchronized_should_stop


def _apply_selection_update(
    *,
    state: TrainingLoopState,
    epoch: int,
    current_metric: float,
    settings: Any,
) -> tuple[bool, bool]:
    mode = settings.monitor_mode
    min_delta = float(settings.early_stopping_min_delta)
    improved = is_improvement(
        current=current_metric,
        best=state.best_metric,
        mode=mode,
        min_delta=min_delta,
    )
    if improved:
        state.best_metric = current_metric
        state.best_epoch = epoch
        state.epochs_without_improvement = 0
    else:
        state.epochs_without_improvement += 1

    patience = settings.early_stopping_patience
    should_stop = (
        patience is not None
        and state.epochs_without_improvement >= int(patience)
    )
    state.stop_reason = "early_stopping" if should_stop else None
    return improved, should_stop


__all__ = ["is_improvement"]
