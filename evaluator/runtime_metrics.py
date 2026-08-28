"""Runtime metrics: CUDA sync, memory, latency."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    pass


class CUDARuntimeObserver:
    """Observe CUDA latency and peak memory."""

    def __init__(
        self,
        *,
        device: Any,
        peak_memory_mb_fn: Callable[[Any], float | None] | None = None,
    ) -> None:
        self._cuda_device = _cuda_device(device)
        self._max_latency_ms: float | None = None
        self._started_ns: int = 0
        self._peak_memory_mb_fn = peak_memory_mb_fn or _peak_memory_mb

    def reset(self, *, device: Any) -> None:
        _reset_cuda_peak_memory(device=device)

    def start_batch(self, *, device: Any) -> None:
        _synchronize_cuda(device=device)
        self._started_ns = time.perf_counter_ns()

    def end_batch(self, *, device: Any) -> float:
        _synchronize_cuda(device=device)
        elapsed_ms = (time.perf_counter_ns() - self._started_ns) / 1_000_000
        if self._max_latency_ms is None or elapsed_ms > self._max_latency_ms:
            self._max_latency_ms = elapsed_ms
        return self._max_latency_ms

    def peak_memory_mb(self, *, device: Any) -> float | None:
        return self._peak_memory_mb_fn(device)


class NoOpRuntimeObserver:
    """No-op observer when runtime metrics are disabled."""

    def reset(self, *, device: Any) -> None:
        pass

    def start_batch(self, *, device: Any) -> None:
        pass

    def end_batch(self, *, device: Any) -> float:
        return 0.0

    def peak_memory_mb(self, *, device: Any) -> float | None:
        return None


def create_runtime_observer(
    *,
    device: Any,
    enabled: bool = True,
    peak_memory_mb_fn: Callable[[Any], float | None] | None = None,
) -> Any:
    """Create a runtime observer for the given device."""
    if not enabled:
        return NoOpRuntimeObserver()
    cuda_device = _cuda_device(device)
    if cuda_device is None:
        if peak_memory_mb_fn is not None:
            return CUDARuntimeObserver(
                device=device, peak_memory_mb_fn=peak_memory_mb_fn
            )
        return NoOpRuntimeObserver()
    return CUDARuntimeObserver(
        device=device, peak_memory_mb_fn=peak_memory_mb_fn
    )


def _cuda_device(device: Any) -> torch.device | None:
    """Return a CUDA device only when CUDA runtime metrics are available."""
    if device is None or not torch.cuda.is_available():
        return None
    try:
        resolved = torch.device(device)
    except (TypeError, ValueError):
        return None
    return resolved if resolved.type == "cuda" else None


def _synchronize_cuda(*, device: Any) -> None:
    """Synchronize CUDA before/after one latency measurement when applicable."""
    cuda_device = _cuda_device(device)
    if cuda_device is None:
        return
    torch.cuda.synchronize(device=cuda_device)


def _reset_cuda_peak_memory(*, device: Any) -> None:
    """Start CUDA peak accounting at the beginning of this evaluation."""
    cuda_device = _cuda_device(device)
    if cuda_device is None:
        return
    torch.cuda.reset_peak_memory_stats(device=cuda_device)


def _peak_memory_mb(device: Any) -> float | None:
    """Return the peak device memory in MiB, or None when unmeasurable."""
    try:
        cuda_device = _cuda_device(device)
        if cuda_device is not None:
            return float(
                torch.cuda.max_memory_allocated(device=cuda_device)
                / (1024 * 1024)
            )
    except (TypeError, ValueError, RuntimeError):
        return None
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


__all__ = [
    "create_runtime_observer",
    "_cuda_device",
    "_synchronize_cuda",
    "_reset_cuda_peak_memory",
    "_peak_memory_mb",
]
