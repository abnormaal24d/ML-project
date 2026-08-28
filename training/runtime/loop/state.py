"""Resumable state for the training loop."""

from __future__ import annotations

import math


class TrainingLoopState:
    """Resumable state for training, selection, and optimizer-step metrics."""

    def __init__(self) -> None:
        self.completed_epochs: int = 0
        self.total_batches: int = 0
        self.completed_optimizer_steps: int = 0
        self.final_loss: float = 0.0
        self.last_val_loss: float | None = None
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.epochs_without_improvement: int = 0
        self.stop_reason: str | None = None
        self.last_gradient_norm: float | None = None
        self.gradient_clip_count: int = 0
        self.epoch_losses: list[float] = []
        self.epoch_history: list[dict[str, object]] = []
        self.cumulative_loss_sum: float = 0.0
        self.test_loss: float | None = None

    @classmethod
    def from_resume_state(
        cls,
        state: dict[str, object] | None,
    ) -> "TrainingLoopState":
        restored = cls()
        if state is None:
            return restored
        restored.completed_epochs = _resume_int(state, field="epoch")
        restored.total_batches = _resume_int(state, field="total_batches")
        restored.completed_optimizer_steps = _resume_int(
            state,
            field="global_step",
        )
        if "best_metric" not in state:
            raise ValueError("checkpoint lacks best_metric")
        best_metric = state["best_metric"]
        restored.best_metric = (
            _resume_float(best_metric, field="best_metric")
            if best_metric is not None
            else None
        )
        if "best_epoch" not in state:
            raise ValueError("checkpoint lacks best_epoch")
        best_epoch = state["best_epoch"]
        restored.best_epoch = (
            _resume_int_value(best_epoch, field="best_epoch")
            if best_epoch is not None
            else None
        )
        if "epochs_without_improvement" not in state:
            raise ValueError("checkpoint lacks epochs_without_improvement")
        restored.epochs_without_improvement = _resume_int(
            state,
            field="epochs_without_improvement",
        )
        if "stop_reason" not in state:
            raise ValueError("checkpoint lacks stop_reason")
        stop_reason = state["stop_reason"]
        if stop_reason is not None and not isinstance(stop_reason, str):
            raise ValueError("checkpoint stop_reason must be a string or null")
        if stop_reason not in {None, "early_stopping"}:
            raise ValueError(
                f"unsupported checkpoint stop_reason: {stop_reason}"
            )
        restored.stop_reason = stop_reason
        restored.last_val_loss = _resume_optional_float(
            state,
            field="last_val_loss",
        )
        restored.final_loss = _resume_float(
            _required_state_value(state, field="final_loss"),
            field="final_loss",
        )
        restored.cumulative_loss_sum = _resume_float(
            _required_state_value(state, field="cumulative_loss_sum"),
            field="cumulative_loss_sum",
        )
        restored.last_gradient_norm = _resume_optional_float(
            state,
            field="last_gradient_norm",
        )
        restored.gradient_clip_count = _resume_int(
            state,
            field="gradient_clip_count",
        )
        epoch_losses = _required_state_value(state, field="epoch_losses")
        if not isinstance(epoch_losses, list):
            raise ValueError("checkpoint epoch_losses must be a list")
        restored.epoch_losses = [
            _resume_float(value, field="epoch_losses")
            for value in epoch_losses
        ]
        epoch_history = _required_state_value(state, field="epoch_history")
        if not isinstance(epoch_history, list) or not all(
            isinstance(record, dict) for record in epoch_history
        ):
            raise ValueError(
                "checkpoint epoch_history must be a list of objects"
            )
        restored.epoch_history = [dict(record) for record in epoch_history]
        return restored


def _resume_int(state: dict[str, object], *, field: str) -> int:
    return _resume_int_value(
        _required_state_value(state, field=field),
        field=field,
    )


def _required_state_value(
    state: dict[str, object],
    *,
    field: str,
) -> object:
    if field not in state:
        raise ValueError(f"checkpoint lacks {field}")
    return state[field]


def _resume_optional_float(
    state: dict[str, object],
    *,
    field: str,
) -> float | None:
    value = _required_state_value(state, field=field)
    return _resume_float(value, field=field) if value is not None else None


def _resume_int_value(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"checkpoint {field} must be a non-negative integer")
    return value


def _resume_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"checkpoint {field} must be numeric")
    candidate = float(value)
    if not math.isfinite(candidate):
        raise ValueError(f"checkpoint {field} must be finite")
    return candidate


__all__ = ["TrainingLoopState"]
