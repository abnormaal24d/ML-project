from __future__ import annotations

import math

import pytest

from crawler.numeric import coerce_finite_float
from crawler.runtime.metrics.metrics_snapshot import clamp, ewma
from crawler.scheduling.host_control.discovery_signal_scorer import (
    coerce_score,
)
from crawler.scheduling.scheduling_value_parser import coerce_float, coerce_int
from crawler.worker.pool.worker_task_counters import WorkerTaskCounters


@pytest.mark.parametrize("value", ("nan", float("nan"), "inf", "-inf"))
def test_finite_float_coercion_rejects_non_finite_values(
    value: object,
) -> None:
    assert coerce_finite_float(value, default=0.25) == 0.25


def test_finite_float_coercion_clamps_valid_values() -> None:
    assert coerce_finite_float(2.0, default=0.0, maximum=1.0) == 1.0
    assert coerce_finite_float(-1.0, default=0.0, minimum=0.0) == 0.0
    assert math.isfinite(
        coerce_finite_float(
            0.5,
            default=0.0,
            minimum=float("inf"),
            maximum=float("-inf"),
        )
    )


def test_host_feedback_coercion_and_ewma_cannot_propagate_nan() -> None:
    assert coerce_score(float("nan"), default=0.5) == 0.5
    assert math.isfinite(ewma(float("nan"), 0.75))
    assert clamp(float("nan"), minimum=0.0, maximum=1.0) == 0.0


def test_worker_metrics_reject_non_finite_external_values() -> None:
    assert coerce_float(float("nan"), default=0.0) == 0.0
    assert coerce_int(float("inf"), default=0) == 0

    counters = WorkerTaskCounters()
    assert (
        counters.record_task_completed(processing_seconds=float("nan")) == 0.0
    )
    assert counters.average_processing_seconds == 0.0
