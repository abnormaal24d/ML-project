"""Epoch and batch orchestration for model training."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from evaluator.loss import evaluate_final_losses, evaluate_loader_loss
from logger.project_logger import ProjectLogger
from training.runtime.cancellations import TrainingCancellationRequested
from training.runtime.loop.model_selection import (
    _monitor_value,
    _update_selection_state,
)
from training.runtime.loop.optimizer_step import (
    _optimizer_step,
    _record_optimizer_step,
)
from training.runtime.loop.state import TrainingLoopState
from training.runtime.precision import (
    PrecisionRuntime,
    autocast_context,
)
from training.runtime.training_batch_processor import TrainingBatchProcessor

if TYPE_CHECKING:
    import torch

    from training.losses.objective import SupervisedOrSelfSupervisedLoss
    from training.runtime.signal import TrainingSignalTracker


class _SchedulerIntervalSettings(Protocol):
    """Minimal settings contract for scheduler cadence."""

    @property
    def scheduler_interval(self) -> object: ...


def set_epoch(*, loader: Any, epoch: int) -> None:
    """Propagate the epoch to samplers and collators that support it."""

    for component in (
        getattr(loader, "sampler", None),
        getattr(loader, "batch_sampler", None),
        getattr(loader, "collate_fn", None),
    ):
        setter = getattr(component, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))


def run_training_loop(
    *,
    settings: Any,
    device: torch.device,
    logger: ProjectLogger,
    model: torch.nn.Module,
    loss_fn: SupervisedOrSelfSupervisedLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    signal_tracker: TrainingSignalTracker,
    precision_runtime: PrecisionRuntime,
    loop_state: TrainingLoopState | None = None,
    grad_scaler: Any | None = None,
    distributed_context: dict[str, object],
    last_epoch_checkpoint: Callable[[TrainingLoopState], None] | None = None,
    best_epoch_checkpoint: Callable[[TrainingLoopState], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[TrainingLoopState, tuple[dict[str, object], ...]]:
    """Run training with correct update, selection, and resume semantics."""
    state = loop_state or TrainingLoopState()
    grad_accum = int(settings.gradient_accumulation_steps)
    log_interval = int(settings.progress_log_interval_batches)
    scheduler_interval = _scheduler_interval(settings)
    precision = precision_runtime
    context = distributed_context
    total_loss_sum = state.cumulative_loss_sum
    total_batches = state.total_batches
    batch_processor = TrainingBatchProcessor(
        device=device,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        signal_tracker=signal_tracker,
        gradient_accumulation_steps=grad_accum,
        precision_runtime=precision,
        grad_scaler=grad_scaler,
    )
    model.train()
    start_epoch = _next_epoch_to_run(state=state, settings=settings)
    for epoch in range(start_epoch, int(settings.epochs)):
        _raise_for_cancellation(cancel_event)
        set_epoch(loader=train_loader, epoch=epoch)
        epoch_loss = 0.0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            _raise_for_cancellation(cancel_event)
            progress = batch_processor.process(
                batch,
                batch_count=batch_count,
                total_batches=total_batches,
                epoch_loss=epoch_loss,
                total_loss_sum=total_loss_sum,
            )
            batch_count = progress.batch_count
            total_batches = progress.total_batches
            epoch_loss = progress.epoch_loss
            total_loss_sum = progress.total_loss_sum
            if batch_count % grad_accum == 0:
                _record_optimizer_step(
                    state=state,
                    result=_optimizer_step(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scheduler_interval=scheduler_interval,
                        grad_scaler=grad_scaler,
                        gradient_clip_max_norm=settings.gradient_clip_max_norm,
                    ),
                )
            if logger is not None and batch_count % max(1, log_interval) == 0:
                logger.info(
                    "training_batch",
                    epoch=epoch,
                    batch=batch_count,
                    loss=float(progress.loss.detach().cpu()),
                )

        if batch_count == 0:
            raise ValueError("training loader produced no batches")
        if batch_count % grad_accum != 0:
            _record_optimizer_step(
                state=state,
                result=_optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scheduler_interval=scheduler_interval,
                    grad_scaler=grad_scaler,
                    gradient_clip_max_norm=settings.gradient_clip_max_norm,
                ),
            )

        if scheduler is not None and scheduler_interval == "epoch":
            scheduler.step()

        avg_epoch_loss = epoch_loss / batch_count
        state.epoch_losses.append(avg_epoch_loss)
        state.completed_epochs = epoch + 1
        state.total_batches = total_batches
        state.final_loss = avg_epoch_loss
        state.cumulative_loss_sum = total_loss_sum

        val_loss = evaluate_loader_loss(
            model=model,
            loss_fn=loss_fn,
            loader=val_loader,
            device=device,
            autocast_factory=lambda: autocast_context(precision),
        )
        state.last_val_loss = val_loss
        monitor_value = _monitor_value(
            validation_loss=val_loss,
            settings=settings,
        )
        improved, should_stop = _update_selection_state(
            state=state,
            epoch=epoch,
            current_metric=monitor_value,
            settings=settings,
            distributed_context=context,
        )
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "val_loss": val_loss,
            "batches": batch_count,
            "optimizer_steps": state.completed_optimizer_steps,
            "monitor_metric": settings.monitor_metric,
            "monitor_value": monitor_value,
            "improved": improved,
            "epochs_without_improvement": state.epochs_without_improvement,
            "gradient_norm": state.last_gradient_norm,
            "gradient_clip_count": state.gradient_clip_count,
        }
        state.epoch_history.append(epoch_record)
        if logger is not None:
            logger.info(
                "epoch_completed",
                epoch=epoch,
                train_loss=avg_epoch_loss,
                val_loss=val_loss,
                monitor_value=monitor_value,
                improved=improved,
                epochs_without_improvement=(state.epochs_without_improvement),
                gradient_norm=state.last_gradient_norm,
                gradient_clip_count=state.gradient_clip_count,
            )
        if improved and best_epoch_checkpoint is not None:
            best_epoch_checkpoint(state)
        if last_epoch_checkpoint is not None:
            last_epoch_checkpoint(state)
        if should_stop:
            break

    _raise_for_cancellation(cancel_event)
    final_train_loss, final_val_loss, final_test_loss = evaluate_final_losses(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        autocast_factory=lambda: autocast_context(precision),
    )
    state.final_loss = final_train_loss
    state.last_val_loss = final_val_loss
    state.test_loss = final_test_loss
    return state, tuple(state.epoch_history)


def _scheduler_interval(
    settings: _SchedulerIntervalSettings,
) -> Literal["step", "epoch"]:
    interval = settings.scheduler_interval
    if not isinstance(interval, str) or interval not in {"step", "epoch"}:
        raise ValueError(f"unsupported scheduler_interval: {interval!r}")
    return cast(Literal["step", "epoch"], interval)


def _next_epoch_to_run(*, state: TrainingLoopState, settings: Any) -> int:
    if state.stop_reason == "early_stopping":
        return int(settings.epochs)
    return state.completed_epochs


def _raise_for_cancellation(cancel_event: threading.Event | None) -> None:
    """Raise when cooperative training cancellation has been requested."""

    if cancel_event is not None and cancel_event.is_set():
        raise TrainingCancellationRequested(
            "training cancellation was requested"
        )


__all__ = ["run_training_loop", "set_epoch"]
