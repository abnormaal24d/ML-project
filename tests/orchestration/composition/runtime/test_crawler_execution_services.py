"""Contracts for the CrawlerExecutionServices return aggregate.

The aggregate contains only services that graph assembly actually reads;
runtime-only collaborators stay internal to the execution subgraph.
"""

from __future__ import annotations

from orchestration.composition.runtime.crawler_execution import (
    CrawlerExecutionServices,
)


def test_execution_services_contract_contains_only_consumed_fields() -> None:
    fields = set(CrawlerExecutionServices.__dataclass_fields__)

    assert fields == {
        "scheduler",
        "worker_pool",
        "worker_scaler",
        "dataset_writer",
        "seed_enqueuer",
        "build_runtime_session",
        "control_directory",
        "task_feedback",
    }


def test_runtime_state_pair_stays_internal() -> None:
    """The runtime reader/writer pair must not leak into the aggregate."""

    fields = CrawlerExecutionServices.__dataclass_fields__

    assert "state_writer" not in fields
    assert "state_reader" not in fields
    assert "task_processor" not in fields


def test_aggregate_is_frozen() -> None:
    assert CrawlerExecutionServices.__dataclass_params__.frozen is True
