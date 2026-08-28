from __future__ import annotations

import math

import pytest

from config.releases.release_requirements import (
    MetricRequirement,
    RuntimeLimitRequirement,
    TaskRequirement,
    check_release_runtime_limits,
    check_release_task_requirements,
)


def _requirement(
    *,
    name: str = "image_text_pair",
    min_samples: int = 100,
    metrics: tuple[MetricRequirement, ...] = (
        MetricRequirement(name="recall_at_1", minimum=0.20),
        MetricRequirement(name="embedding_similarity_mean", maximum=0.99),
    ),
) -> TaskRequirement:
    return TaskRequirement(
        name=name,
        min_samples=min_samples,
        metrics=metrics,
    )


def test_passing_samples_and_metrics_produce_no_reasons() -> None:
    reasons = check_release_task_requirements(
        task_requirements=(_requirement(),),
        task_counts={"image_text_pair": 250},
        task_metrics={
            "image_text_pair": {
                "recall_at_1": 0.35,
                "embedding_similarity_mean": 0.90,
            }
        },
    )
    assert reasons == ()


def test_too_few_samples_fails_closed() -> None:
    reasons = check_release_task_requirements(
        task_requirements=(_requirement(),),
        task_counts={"image_text_pair": 99},
        task_metrics={
            "image_text_pair": {
                "recall_at_1": 0.35,
                "embedding_similarity_mean": 0.90,
            }
        },
    )
    assert reasons == ("task_min_samples_not_met:image_text_pair:99:100",)


def test_missing_task_counts_fail_closed() -> None:
    reasons = check_release_task_requirements(
        task_requirements=(_requirement(),),
        task_counts={},
        task_metrics={
            "image_text_pair": {
                "recall_at_1": 0.35,
                "embedding_similarity_mean": 0.90,
            }
        },
    )
    assert reasons == ("task_min_samples_not_met:image_text_pair:0:100",)


def test_missing_metric_fails_closed() -> None:
    reasons = check_release_task_requirements(
        task_requirements=(_requirement(),),
        task_counts={"image_text_pair": 250},
        task_metrics={
            "image_text_pair": {
                "recall_at_1": 0.35,
            }
        },
    )
    assert reasons == (
        "evaluation_metric_missing:image_text_pair:embedding_similarity_mean",
    )


def test_metric_below_minimum_fails_closed() -> None:
    reasons = check_release_task_requirements(
        task_requirements=(_requirement(),),
        task_counts={"image_text_pair": 250},
        task_metrics={
            "image_text_pair": {
                "recall_at_1": 0.19,
                "embedding_similarity_mean": 0.90,
            }
        },
    )
    assert reasons == (
        "evaluation_metric_below_threshold:"
        "image_text_pair:recall_at_1:0.19:0.2",
    )


def test_metric_above_maximum_fails_closed() -> None:
    reasons = check_release_task_requirements(
        task_requirements=(_requirement(),),
        task_counts={"image_text_pair": 250},
        task_metrics={
            "image_text_pair": {
                "recall_at_1": 0.35,
                "embedding_similarity_mean": 0.995,
            }
        },
    )
    assert reasons == (
        "evaluation_metric_above_threshold:"
        "image_text_pair:embedding_similarity_mean:0.995:0.99",
    )


@pytest.mark.parametrize(
    ("value", "rendered"),
    ((math.nan, "nan"), (math.inf, "inf"), (-math.inf, "-inf")),
)
def test_non_finite_metric_fails_closed(value: float, rendered: str) -> None:
    reasons = check_release_task_requirements(
        task_requirements=(_requirement(),),
        task_counts={"image_text_pair": 250},
        task_metrics={
            "image_text_pair": {
                "recall_at_1": value,
                "embedding_similarity_mean": 0.90,
            }
        },
    )
    assert reasons == (
        f"evaluation_metric_non_finite:image_text_pair:recall_at_1:{rendered}",
    )


def test_every_violation_is_reported_for_one_task() -> None:
    reasons = check_release_task_requirements(
        task_requirements=(
            _requirement(
                min_samples=200,
                metrics=(
                    MetricRequirement(name="recall_at_1", minimum=0.30),
                    MetricRequirement(
                        name="embedding_similarity_mean", minimum=0.50
                    ),
                ),
            ),
        ),
        task_counts={"image_text_pair": 150},
        task_metrics={
            "image_text_pair": {
                "recall_at_1": 0.25,
                "embedding_similarity_mean": math.nan,
            }
        },
    )
    assert reasons == (
        "task_min_samples_not_met:image_text_pair:150:200",
        "evaluation_metric_below_threshold:"
        "image_text_pair:recall_at_1:0.25:0.3",
        "evaluation_metric_non_finite:image_text_pair:"
        "embedding_similarity_mean:nan",
    )


def test_metric_requirement_requires_exactly_one_bound() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MetricRequirement(name="recall_at_1", minimum=0.2, maximum=0.9)
    with pytest.raises(ValueError, match="exactly one"):
        MetricRequirement(name="recall_at_1")


def _limits(
    *,
    latency_ms: float | None = 250.0,
    memory_mb: int | None = 16384,
) -> RuntimeLimitRequirement:
    return RuntimeLimitRequirement(
        max_batch_latency_ms=latency_ms,
        max_peak_memory_mb=memory_mb,
    )


def test_runtime_limits_pass_within_bounds() -> None:
    reasons = check_release_runtime_limits(
        runtime_limits=_limits(),
        max_batch_latency_ms=200.0,
        peak_memory_mb=8192.0,
    )
    assert reasons == ()


def test_runtime_limits_fail_closed_when_unrequired() -> None:
    assert (
        check_release_runtime_limits(
            runtime_limits=RuntimeLimitRequirement(),
            max_batch_latency_ms=None,
            peak_memory_mb=None,
        )
        == ()
    )
    assert (
        check_release_runtime_limits(
            runtime_limits=None,
            max_batch_latency_ms=999.0,
            peak_memory_mb=999.0,
        )
        == ()
    )


def test_latency_above_limit_fails_closed() -> None:
    reasons = check_release_runtime_limits(
        runtime_limits=_limits(latency_ms=250.0),
        max_batch_latency_ms=251.0,
        peak_memory_mb=8192.0,
    )
    assert reasons == ("batch_latency_above_max:251:250",)


def test_memory_above_limit_fails_closed() -> None:
    reasons = check_release_runtime_limits(
        runtime_limits=_limits(memory_mb=16384),
        max_batch_latency_ms=200.0,
        peak_memory_mb=16385.0,
    )
    assert reasons == ("peak_memory_above_max:16385:16384",)


def test_missing_latency_measurement_fails_closed() -> None:
    reasons = check_release_runtime_limits(
        runtime_limits=_limits(latency_ms=250.0),
        max_batch_latency_ms=None,
        peak_memory_mb=8192.0,
    )
    assert reasons == ("runtime_measurement_missing:max_batch_latency_ms",)


def test_missing_memory_measurement_fails_closed() -> None:
    reasons = check_release_runtime_limits(
        runtime_limits=_limits(memory_mb=16384),
        max_batch_latency_ms=200.0,
        peak_memory_mb=None,
    )
    assert reasons == ("runtime_measurement_missing:peak_memory_mb",)


def test_non_finite_latency_measurement_fails_closed() -> None:
    reasons = check_release_runtime_limits(
        runtime_limits=_limits(latency_ms=250.0),
        max_batch_latency_ms=math.nan,
        peak_memory_mb=8192.0,
    )
    assert reasons == ("runtime_measurement_missing:max_batch_latency_ms",)


def test_both_limits_violated_are_reported() -> None:
    reasons = check_release_runtime_limits(
        runtime_limits=_limits(),
        max_batch_latency_ms=None,
        peak_memory_mb=20000.0,
    )
    assert reasons == (
        "runtime_measurement_missing:max_batch_latency_ms",
        "peak_memory_above_max:20000:16384",
    )
