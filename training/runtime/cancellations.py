"""Cooperative cancellation for the training runtime."""

from __future__ import annotations


class TrainingCancellationRequested(BaseException):
    """Raised when cooperative cancellation of training is requested.

    The training loop observes a shared ``threading.Event`` and raises this
    exception on an epoch or batch boundary so the worker can finish its
    current step before unwinding. The orchestrator maps the exception to a
    cancelled attempt rather than a failed one.
    """


__all__ = ["TrainingCancellationRequested"]
