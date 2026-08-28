"""Random, optimizer-step, and resumable training state."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from config.environment.runtime_environment import cublas_workspace_config
from training.runtime.checkpoint.io import (
    checkpoint_is_available,
    safe_torch_load,
)

if TYPE_CHECKING:
    from config.multimodal.training_settings import TrainingSettings
    from training.runtime.loop.state import TrainingLoopState
    from training.runtime.planner import TrainingScalePlan

_LOGGER = logging.getLogger(__name__)


def _import_numpy() -> Any:
    try:
        import numpy as np

        return np
    except ImportError:
        return None


def resume_optimizer_steps(*, settings: TrainingSettings) -> int:
    """Read the saved optimizer-step count needed to construct a scheduler."""

    checkpoint_value = settings.resume_from_checkpoint
    if checkpoint_value is None or not str(checkpoint_value).strip():
        return 0
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_is_available(checkpoint_path):
        raise FileNotFoundError(
            f"resume checkpoint not found: {checkpoint_path}"
        )
    payload = safe_torch_load(checkpoint_path)
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint payload must be a dictionary")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("resume checkpoint lacks complete training_state")
    validate_epoch_resume_state(training_state)
    global_step = training_state.get("global_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError(
            "resume checkpoint global_step must be a non-negative integer"
        )
    return global_step


def set_reproducible_seed(*, settings: TrainingSettings) -> None:
    """Seed Python, NumPy, and torch according to training settings."""

    seed = int(settings.seed)
    random.seed(seed)
    np = _import_numpy()
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = bool(settings.deterministic)

    torch.use_deterministic_algorithms(deterministic)

    torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        required_workspace = ":4096:8"
        actual_workspace = cublas_workspace_config()
        if (
            torch.cuda.is_available()
            and actual_workspace != required_workspace
        ):
            raise RuntimeError(
                "deterministic training requires "
                f"CUBLAS_WORKSPACE_CONFIG={required_workspace}"
            )
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    else:
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True


@contextmanager
def reproducibility_runtime(*, settings: TrainingSettings) -> Iterator[None]:
    """Run one reproducible training region and restore RNG on exit.

    The caller-skewed random state is captured, the region starts from the
    configured seed, and the previous state is restored afterwards so the
    deterministic region is self-contained and repeatable regardless of how
    many RNG draws happened before it in the same process.

    Also restores torch deterministic algorithm flags to their previous state.
    """

    previous_state = capture_random_state()
    previous_deterministic_algorithms = (
        torch.are_deterministic_algorithms_enabled()
    )
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cuda_matmul_allow_tf32 = (
        torch.backends.cuda.matmul.allow_tf32
        if hasattr(torch.backends.cuda.matmul, "allow_tf32")
        else None
    )
    previous_cudnn_allow_tf32 = (
        torch.backends.cudnn.allow_tf32
        if hasattr(torch.backends.cudnn, "allow_tf32")
        else None
    )
    set_reproducible_seed(settings=settings)
    try:
        yield
    finally:
        _restore_random_state(state=previous_state)
        torch.use_deterministic_algorithms(previous_deterministic_algorithms)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        if previous_cuda_matmul_allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = (
                previous_cuda_matmul_allow_tf32
            )
        if previous_cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = previous_cudnn_allow_tf32


def capture_random_state() -> dict[str, Any]:
    """Capture random generator state for resumable checkpoints."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    np = _import_numpy()
    if np is not None:
        state["numpy"] = _serialize_numpy_random_state(np.random.get_state())
    return state


def build_training_state_payload(
    *,
    state: TrainingLoopState,
    val_loss: float | None,
    random_state: dict[str, object],
    training_plan: TrainingScalePlan,
    stage_state: object | None = None,
    sampler_position: int = 0,
    gradient_accumulation_position: int = 0,
    rank: int = 0,
) -> dict[str, object]:
    """Return resumable training state stored inside a checkpoint."""

    if sampler_position != 0 or gradient_accumulation_position != 0:
        raise ValueError(
            "mid-epoch checkpoint state is unsupported; checkpoints must be "
            "created on epoch boundaries"
        )

    return {
        "resume_granularity": "epoch",
        "epoch": state.completed_epochs,
        "global_step": state.completed_optimizer_steps,
        "total_batches": state.total_batches,
        "final_loss": state.final_loss,
        "cumulative_loss_sum": state.cumulative_loss_sum,
        "epoch_losses": list(state.epoch_losses),
        "epoch_history": [dict(record) for record in state.epoch_history],
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
        "epochs_without_improvement": state.epochs_without_improvement,
        "stop_reason": state.stop_reason,
        "last_val_loss": val_loss,
        "last_gradient_norm": state.last_gradient_norm,
        "gradient_clip_count": state.gradient_clip_count,
        "random_state": random_state,
        "training_scale_plan": training_plan.to_dict(),
        "sampler_position": int(sampler_position),
        "gradient_accumulation_position": int(gradient_accumulation_position),
        "rank": int(rank),
        "training_stage": (
            stage_state.to_payload()
            if hasattr(stage_state, "to_payload")
            else stage_state
        ),
    }


def validate_epoch_resume_state(training_state: dict[str, object]) -> None:
    """Reject checkpoint state that claims unsupported mid-epoch progress."""

    granularity = training_state.get("resume_granularity", "epoch")
    if granularity != "epoch":
        raise ValueError(
            "unsupported checkpoint resume granularity: "
            f"{granularity!r}; only epoch-boundary resume is supported"
        )
    for field in ("sampler_position", "gradient_accumulation_position"):
        value = training_state.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"resume checkpoint {field} must be an integer")
        if value != 0:
            raise ValueError(
                "mid-epoch resume is unsupported: "
                f"{field}={value}; use an epoch-boundary checkpoint"
            )


def _restore_random_state(*, state: dict[str, object]) -> None:
    python_state = state.get("python")
    if isinstance(python_state, tuple):
        random.setstate(python_state)

    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state)

    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        torch.cuda.set_rng_state_all(cuda_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np = _import_numpy()
        if np is not None:
            try:
                numpy_state = _coerce_numpy_random_state(
                    numpy_state=numpy_state
                )
                np.random.set_state(numpy_state)
            except (TypeError, ValueError) as exc:
                _LOGGER.warning(
                    "numpy_random_state_restore_failed",
                    extra={"error_type": type(exc).__name__},
                )


def _serialize_numpy_random_state(numpy_state: object) -> object:
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        return numpy_state

    algorithm, keys, position, has_gauss, cached_gaussian = numpy_state
    if hasattr(keys, "tolist"):
        keys = keys.tolist()
    return (
        algorithm,
        keys,
        int(position),
        int(has_gauss),
        float(cached_gaussian),
    )


def _coerce_numpy_random_state(*, numpy_state: object) -> Any:
    if not isinstance(numpy_state, (list, tuple)) or len(numpy_state) != 5:
        return numpy_state

    algorithm, keys, position, has_gauss, cached_gaussian = numpy_state
    try:
        import numpy as np

        if isinstance(keys, list):
            keys = np.array(keys, dtype="uint32")
    except ImportError:
        return numpy_state
    return (
        algorithm,
        keys,
        int(position),
        int(has_gauss),
        float(cached_gaussian),
    )


__all__ = [
    "build_training_state_payload",
    "capture_random_state",
    "reproducibility_runtime",
    "resume_optimizer_steps",
    "set_reproducible_seed",
    "validate_epoch_resume_state",
]
