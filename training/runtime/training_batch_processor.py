"""One-batch training execution with strict loss and gradient validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from mmcrawler_datasets.collation.multimodal import move_batch_to_device
from multimodal.model.contracts import CollatedBatch
from training.runtime.device import unwrap_model
from training.runtime.precision import (
    GradScaler,
    PrecisionRuntime,
    autocast_context,
)
from training.runtime.preparation import (
    dense_batch_requires_causal_targets,
    validate_dense_decoder_batch,
)

if TYPE_CHECKING:
    from training.losses.objective import SupervisedOrSelfSupervisedLoss
    from training.runtime.signal import TrainingSignalTracker


@dataclass(frozen=True, slots=True)
class BatchProgress:
    loss: torch.Tensor
    batch_count: int
    total_batches: int
    epoch_loss: float
    total_loss_sum: float


class TrainingBatchProcessor:
    def __init__(
        self,
        *,
        device: torch.device,
        model: torch.nn.Module,
        loss_fn: SupervisedOrSelfSupervisedLoss,
        optimizer: torch.optim.Optimizer,
        signal_tracker: TrainingSignalTracker,
        gradient_accumulation_steps: int,
        precision_runtime: PrecisionRuntime,
        grad_scaler: GradScaler | None = None,
    ) -> None:
        if not isinstance(loss_fn, torch.nn.Module):
            raise TypeError(
                "loss_fn must be a torch.nn.Module implementing the training loss contract"
            )
        self._device = device
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._signal_tracker = signal_tracker
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._precision_runtime = precision_runtime
        self._grad_scaler = grad_scaler

    def process(
        self,
        batch: object,
        *,
        batch_count: int,
        total_batches: int,
        epoch_loss: float,
        total_loss_sum: float,
    ) -> BatchProgress:
        if not isinstance(batch, CollatedBatch):
            raise TypeError(
                "training loader must yield the canonical CollatedBatch"
            )
        batch = move_batch_to_device(batch=batch, device=self._device)
        model_batch = _preference_primary_batch(batch)
        concrete_model = unwrap_model(self._model)
        if getattr(
            concrete_model, "training_backend", None
        ) == "dense_transformer" and dense_batch_requires_causal_targets(
            batch=model_batch
        ):
            validate_dense_decoder_batch(batch=model_batch)
        with autocast_context(self._precision_runtime):
            try:
                outputs = self._model(model_batch)
            except (
                Exception
            ) as exc:  # exception-rules: boundary-wrap-and-raise
                raise RuntimeError(
                    f"Model forward failed for batch: {exc}"
                ) from exc

            loss = self._compute_loss(outputs=outputs, batch=batch)
        if loss.ndim > 0:
            loss = loss.mean()
        if not torch.isfinite(loss).all():
            self._optimizer.zero_grad(set_to_none=True)
            raise ValueError("non-finite training loss encountered")

        scaled_loss = loss / self._gradient_accumulation_steps
        if self._grad_scaler is None:
            torch.autograd.backward(scaled_loss)
        else:
            torch.autograd.backward(self._grad_scaler.scale(scaled_loss))
        self._signal_tracker.record_after_backward()
        scalar_loss = float(loss.detach().cpu())
        return BatchProgress(
            loss=loss,
            batch_count=batch_count + 1,
            total_batches=total_batches + 1,
            epoch_loss=epoch_loss + scalar_loss,
            total_loss_sum=total_loss_sum + scalar_loss,
        )

    def _compute_loss(
        self, *, outputs: object, batch: CollatedBatch
    ) -> torch.Tensor:
        normalized_outputs: dict[str, torch.Tensor]
        if isinstance(outputs, dict):
            normalized_outputs = {}
            for name, value in outputs.items():
                if not isinstance(name, str) or not torch.is_tensor(value):
                    raise TypeError(
                        "model outputs must map string names to tensors"
                    )
                normalized_outputs[name] = value
        elif torch.is_tensor(outputs):
            normalized_outputs = {"logits": outputs}
        else:
            raise TypeError(
                "model output must be a tensor or a mapping of tensors"
            )
        loss_result = self._loss_fn(
            model_output=normalized_outputs,
            batch=batch,
            require_targets_for_generation=True,
        )
        if isinstance(loss_result, dict):
            total = loss_result.get("total")
            if torch.is_tensor(total):
                return total
            tensor_losses = [
                value
                for key, value in loss_result.items()
                if key != "total" and torch.is_tensor(value)
            ]
            if tensor_losses:
                return torch.stack(tensor_losses).sum()
            raise RuntimeError(
                "Loss computation failed; no dummy loss allowed"
            )
        if torch.is_tensor(loss_result):
            return loss_result
        raise RuntimeError("Loss computation failed; no dummy loss allowed")


def raise_for_non_finite_gradients(model: Any) -> None:
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            raise ValueError(f"non-finite gradient encountered: {name}")


def _preference_primary_batch(batch: CollatedBatch) -> CollatedBatch:
    """Use the chosen response as the primary decoder stream when present."""

    if batch.chosen_input_ids is None:
        return batch
    if batch.chosen_labels is None or batch.chosen_attention_mask is None:
        raise ValueError("preference batches require complete chosen tensors")
    return replace(
        batch,
        decoder_input_ids=batch.chosen_input_ids,
        decoder_labels=batch.chosen_labels,
        decoder_attention_mask=batch.chosen_attention_mask,
    )
