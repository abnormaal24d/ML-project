"""Architecture contracts for crawler dead-letter composition.

These tests verify the wiring invariants in the crawler state and execution
subgraphs where the actual composition now resides.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from orchestration.composition.runtime.crawler_execution import (
    state_runtime as state_runtime_module,
)
from orchestration.composition.runtime import crawler_state as state_module
from orchestration.composition.runtime import scheduler as scheduler_module


def _source_tree(callable_object: object) -> ast.Module:
    return ast.parse(
        textwrap.dedent(inspect.getsource(callable_object)),
    )


def _named_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr:
    matches = [item.value for item in call.keywords if item.arg == name]
    assert len(matches) == 1
    return matches[0]


def _assert_name(node: ast.expr, expected: str) -> None:
    assert isinstance(node, ast.Name)
    assert node.id == expected


def test_writer_precedes_scheduler_and_reader_uses_same_file() -> None:
    # Dead letter writer is created in build_crawler_state
    state_tree = _source_tree(state_module.build_crawler_state)
    (writer_call,) = _named_calls(state_tree, "CrawlerDeadLetterWriter")

    # Scheduler is built via build_scheduler from runtime.scheduler,
    # which receives dead_letter_writer from the state persistence subgraph.
    scheduler_tree = _source_tree(scheduler_module.build_scheduler)
    (scheduler_call,) = _named_calls(scheduler_tree, "SchedulerCompletionHandler")

    # Reader is built in build_execution_state_runtime (extracted subgraph)
    state_runtime_tree = _source_tree(
        state_runtime_module.build_execution_state_runtime
    )
    (reader_call,) = _named_calls(state_runtime_tree, "CrawlerDeadLetterReader")

    # Verify SchedulerCompletionHandler receives dead_letter_writer keyword
    # argument (the shared instance built by the state persistence subgraph).
    scheduler_kwarg = _keyword(scheduler_call, "dead_letter_writer")
    if isinstance(scheduler_kwarg, ast.Attribute):
        _assert_name(scheduler_kwarg.value, "state")
        assert scheduler_kwarg.attr == "dead_letter_writer"
    else:
        assert isinstance(scheduler_kwarg, ast.Name)

    # Verify writer has dead_letter_path keyword argument
    writer_path = _keyword(writer_call, "dead_letter_path")
    assert isinstance(writer_path, ast.Name)

    # Verify reader has dead_letter_path keyword argument (the shared
    # persistence instance from the state subgraph).
    reader_path = _keyword(reader_call, "dead_letter_path")
    if isinstance(reader_path, ast.Attribute):
        _assert_name(reader_path.value, "state")
        assert reader_path.attr == "dead_letter_path"
    else:
        assert isinstance(reader_path, ast.Name)

    deserializer = _keyword(reader_call, "task_deserializer")
    assert isinstance(deserializer, ast.Call)
    assert isinstance(deserializer.func, ast.Attribute)
    assert deserializer.func.attr == "create_checkpoint_task_deserializer"
    _assert_name(deserializer.func.value, "scheduler")


def test_scheduler_injects_dead_letter_writer_into_completion_handler() -> (
    None
):
    tree = _source_tree(scheduler_module.build_scheduler)
    (completion_call,) = _named_calls(tree, "SchedulerCompletionHandler")

    _assert_name(
        _keyword(completion_call, "dead_letter_writer"),
        "dead_letter_writer",
    )


def test_dead_letter_acknowledgement_uses_runtime_checkpoint_writer() -> None:
    # State writer/reader are built in build_execution_state_runtime
    tree = _source_tree(state_runtime_module.build_execution_state_runtime)
    (state_writer_call,) = _named_calls(tree, "CrawlStateWriter")
    (state_reader_call,) = _named_calls(tree, "CrawlStateReader")

    assert state_writer_call.lineno < state_reader_call.lineno
    _assert_name(
        _keyword(state_reader_call, "checkpoint_writer"),
        "state_writer",
    )
