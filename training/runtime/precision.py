"""Precision policy shared by training, evaluation, and memory planning."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Literal, Protocol, cast

import torch

PrecisionName = Literal["fp32", "fp16", "bf16"]


class _PrecisionSettings(Protocol):
    """Minimal settings contract needed to resolve the precision policy."""

    @property
    def precision(self) -> object: ...


class GradScaler(Protocol):
    """Torch scaler operations used by the training loop."""

    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...

    def step(self, optimizer: torch.optim.Optimizer) -> object: ...

    def update(self) -> None: ...


class _GradScalerFactory(Protocol):
    def __call__(self, *, device: str, enabled: bool) -> GradScaler: ...


class UnsupportedPrecisionError(RuntimeError):
    """Raised when configured precision cannot run on the selected device."""


@dataclass(frozen=True, slots=True)
class PrecisionRuntime:
    """Resolved precision behavior for one concrete training device."""

    name: PrecisionName
    device_type: str
    autocast_enabled: bool
    autocast_dtype: torch.dtype | None
    uses_grad_scaler: bool


def precision_from_settings(settings: _PrecisionSettings) -> PrecisionName:
    """Return the explicit configured precision."""

    value = settings.precision
    if not isinstance(value, str) or value not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"unsupported precision: {value!r}")
    return cast(PrecisionName, value)


def bytes_per_element(precision: PrecisionName) -> int:
    """Return the storage size used by the configured compute precision."""

    return {"fp32": 4, "fp16": 2, "bf16": 2}[precision]


def resolve_precision_runtime(
    *,
    settings: _PrecisionSettings,
    device: torch.device,
) -> PrecisionRuntime:
    """Validate precision/device compatibility and return runtime behavior.

    Mixed precision is intentionally fail-closed: a requested half precision
    mode never silently becomes FP32 while the memory plan still assumes two
    bytes per element.
    """

    precision = precision_from_settings(settings)
    if precision == "fp32":
        return PrecisionRuntime(
            name="fp32",
            device_type=device.type,
            autocast_enabled=False,
            autocast_dtype=None,
            uses_grad_scaler=False,
        )

    _require_cuda_precision(precision=precision, device=device)
    if precision == "fp16":
        return PrecisionRuntime(
            name="fp16",
            device_type="cuda",
            autocast_enabled=True,
            autocast_dtype=torch.float16,
            uses_grad_scaler=True,
        )

    _require_bfloat16_support(device=device)
    return PrecisionRuntime(
        name="bf16",
        device_type="cuda",
        autocast_enabled=True,
        autocast_dtype=torch.bfloat16,
        uses_grad_scaler=False,
    )


def autocast_context(runtime: PrecisionRuntime) -> ContextManager[object]:
    """Return the single autocast policy used by all model forward passes."""

    if not runtime.autocast_enabled:
        return nullcontext()
    dtype = runtime.autocast_dtype
    if dtype is None:  # Defensive invariant guard for future precision modes.
        raise RuntimeError("autocast precision is missing its dtype")
    return torch.autocast(
        device_type=runtime.device_type,
        dtype=dtype,
        enabled=True,
    )


def build_grad_scaler(runtime: PrecisionRuntime) -> GradScaler | None:
    """Create a scaler only for CUDA FP16 training."""

    if not runtime.uses_grad_scaler:
        return None
    scaler_factory = vars(torch.amp).get("GradScaler")
    if not callable(scaler_factory):
        raise RuntimeError("torch.amp.GradScaler is unavailable")
    return cast(_GradScalerFactory, scaler_factory)(
        device=runtime.device_type,
        enabled=True,
    )


def _require_cuda_precision(
    *,
    precision: PrecisionName,
    device: torch.device,
) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise UnsupportedPrecisionError(
            f"precision={precision!r} requires an available CUDA device; "
            f"resolved device is {device!s}"
        )


def _require_bfloat16_support(*, device: torch.device) -> None:
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(checker) or not bool(checker()):
        raise UnsupportedPrecisionError(
            "precision='bf16' is not supported by the selected CUDA device "
            f"{device!s}"
        )


__all__ = [
    "PrecisionName",
    "PrecisionRuntime",
    "GradScaler",
    "UnsupportedPrecisionError",
    "autocast_context",
    "build_grad_scaler",
    "bytes_per_element",
    "precision_from_settings",
    "resolve_precision_runtime",
]
