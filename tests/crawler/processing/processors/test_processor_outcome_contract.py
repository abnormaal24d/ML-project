from __future__ import annotations

import pytest

from crawler.processing.outcomes.processor_outcome import (
    RESERVED_PROCESSOR_METADATA_KEYS,
    ProcessorOutcome,
)

# --- status / stage / reason / detail invariants --------------------------


def test_only_the_four_processor_status_values_are_valid() -> None:
    with pytest.raises(
        ValueError, match="unsupported processor outcome status"
    ):
        ProcessorOutcome(status="skipped", stage="analysis")  # type: ignore[arg-type]


def test_stage_is_required_and_non_empty() -> None:
    with pytest.raises(ValueError, match="stage must be a non-empty string"):
        ProcessorOutcome.success(stage="")

    with pytest.raises(ValueError, match="stage must be a non-empty string"):
        ProcessorOutcome.success(stage="  ")


def test_non_success_status_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="dropped outcome requires a reason"):
        ProcessorOutcome.dropped(stage="quality", reason="")

    with pytest.raises(ValueError, match="deferred outcome requires a reason"):
        ProcessorOutcome.deferred(
            stage="fetch",
            reason="",
            retry_after_seconds=5.0,
        )

    with pytest.raises(ValueError, match="failure outcome requires a reason"):
        ProcessorOutcome.failure(stage="analysis", reason="")

    assert ProcessorOutcome.success(stage="persistence").reason == ""


def test_deferred_requires_retry_after_seconds() -> None:
    with pytest.raises(
        ValueError, match="deferred outcome requires retry_after"
    ):
        ProcessorOutcome.deferred(
            stage="fetch",
            reason="host_not_ready",
            retry_after_seconds=None,  # type: ignore[arg-type]
        )


def test_non_deferred_statuses_reject_retry_after_seconds() -> None:
    for outcome in (
        ProcessorOutcome.dropped(stage="quality", reason="quality_rejected"),
        ProcessorOutcome.success(stage="analysis"),
        ProcessorOutcome.failure(stage="analysis", reason="analysis_failed"),
    ):
        with pytest.raises(ValueError, match="only valid for deferred"):
            ProcessorOutcome(
                status=outcome.status,
                stage=outcome.stage,
                reason=outcome.reason,
                retry_after_seconds=1.0,
            )


def test_retry_after_seconds_validation() -> None:
    with pytest.raises(TypeError, match="must be numeric"):
        ProcessorOutcome.deferred(
            stage="fetch",
            reason="retryable",
            retry_after_seconds=True,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="must be numeric"):
        ProcessorOutcome.deferred(
            stage="fetch",
            reason="retryable",
            retry_after_seconds="almost",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        ProcessorOutcome.deferred(
            stage="fetch",
            reason="retryable",
            retry_after_seconds=-1.0,
        )

    with pytest.raises(ValueError, match="must be finite"):
        ProcessorOutcome.deferred(
            stage="fetch",
            reason="retryable",
            retry_after_seconds=float("nan"),
        )

    with pytest.raises(ValueError, match="must be finite"):
        ProcessorOutcome.deferred(
            stage="fetch",
            reason="retryable",
            retry_after_seconds=float("inf"),
        )


def test_retry_after_seconds_is_normalized_to_float() -> None:
    outcome = ProcessorOutcome.deferred(
        stage="fetch",
        reason="host_not_ready",
        retry_after_seconds=5,
    )
    assert outcome.retry_after_seconds == 5.0
    assert isinstance(outcome.retry_after_seconds, float)


def test_succeeded_property() -> None:
    assert ProcessorOutcome.success(stage="x").succeeded is True
    assert ProcessorOutcome.dropped(stage="x", reason="y").succeeded is False
    assert (
        ProcessorOutcome.deferred(
            stage="x",
            reason="y",
            retry_after_seconds=1.0,
        ).succeeded
        is False
    )
    assert ProcessorOutcome.failure(stage="x", reason="y").succeeded is False


def test_malformed_role_strings_are_rejected() -> None:
    with pytest.raises(TypeError, match="retry_class must be a string"):
        ProcessorOutcome.dropped(
            stage="x",
            reason="y",
            retry_class=5,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="error must be a string"):
        ProcessorOutcome.dropped(
            stage="x",
            reason="y",
            error=object(),  # type: ignore[arg-type]
        )


# --- metadata invariants ------------------------------------------------


def test_metadata_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        ProcessorOutcome.dropped(
            stage="quality",
            reason="quality_rejected",
            metadata=["quality_score"],  # type: ignore[arg-type]
        )


def test_metadata_keys_must_be_non_empty_strings() -> None:
    with pytest.raises(ValueError, match="metadata keys must be non-empty"):
        ProcessorOutcome.dropped(
            stage="x",
            reason="y",
            metadata={5: "value"},  # type: ignore[dict-item]
        )

    with pytest.raises(ValueError, match="metadata keys must be non-empty"):
        ProcessorOutcome.dropped(
            stage="x",
            reason="y",
            metadata={"": "value"},
        )


def test_reserved_keys_are_refused_in_metadata() -> None:
    assert "reason" in RESERVED_PROCESSOR_METADATA_KEYS
    assert "wait_seconds" not in RESERVED_PROCESSOR_METADATA_KEYS

    with pytest.raises(ValueError, match="reserved processor key 'reason'"):
        ProcessorOutcome.dropped(
            stage="quality",
            reason="quality_rejected",
            metadata={"reason": "forged"},
        )


def test_extension_metadata_is_preserved() -> None:
    outcome = ProcessorOutcome.dropped(
        stage="quality",
        reason="quality_rejected",
        metadata={
            "quality_score": 0.1,
            "detector_name": "det-1",
            "quality_threshold": 0.35,
        },
    )
    assert outcome.metadata["quality_score"] == 0.1
    assert outcome.metadata["detector_name"] == "det-1"
    assert outcome.metadata["quality_threshold"] == 0.35


# --- immutability --------------------------------------------------------


def test_metadata_mapping_is_defensively_copied() -> None:
    source: dict[str, object] = {"quality_score": 0.4}
    outcome = ProcessorOutcome.dropped(
        stage="quality",
        reason="quality_rejected",
        metadata=source,
    )

    source["quality_score"] = 1.0

    assert outcome.metadata["quality_score"] == 0.4


def test_direct_metadata_mutation_fails() -> None:
    outcome = ProcessorOutcome.dropped(
        stage="x",
        reason="y",
        metadata={"quality_score": 0.1},
    )

    with pytest.raises(TypeError):
        outcome.metadata["quality_score"] = 0.9  # type: ignore[index]


def test_dict_and_kwargs_expansion_of_metadata() -> None:
    outcome = ProcessorOutcome.dropped(
        stage="x",
        reason="y",
        metadata={"quality_score": 0.1},
    )

    flat = dict(outcome.metadata)
    assert flat == {"quality_score": 0.1}

    payload: dict[str, object] = {}
    payload.update(outcome.metadata)
    assert payload == {"quality_score": 0.1}


def test_nested_values_are_not_deep_frozen() -> None:
    outcome = ProcessorOutcome.dropped(
        stage="x",
        reason="y",
        metadata={"labels": ["a", "b"]},
    )

    assert isinstance(outcome.metadata["labels"], list)
    outcome.metadata["labels"].append("c")
    assert outcome.metadata["labels"] == ["a", "b", "c"]


# --- reserved namespace ownership ---------------------------------------


def test_reserved_set_owns_all_policy_and_fact_names() -> None:
    expected = {
        "status",
        "stage",
        "reason",
        "detail",
        "retry_after_seconds",
        "retry_class",
        "retry_error_kind",
        "counts_toward_task_retry_budget",
        "terminal_eligible",
        "error_type",
        "error",
        "task_id",
        "url",
        "requested_kind",
        "target_kind",
        "kind",
        "result_kind",
        "content_type",
        "mime_type",
        "bytes",
        "stored",
        "category",
        "relevance_score",
    }
    assert RESERVED_PROCESSOR_METADATA_KEYS == frozenset(expected)


def test_metadata_never_carries_task_facts() -> None:
    outcome = ProcessorOutcome.dropped(
        stage="quality",
        reason="quality_rejected",
        metadata={"quality_score": 0.1},
    )

    for name in (
        "task_id",
        "kind",
        "retry_after_seconds",
    ):
        assert name not in outcome.metadata
