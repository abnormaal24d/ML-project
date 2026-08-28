"""Architecture tests for crawler execution boundary invariants.

Enforces that build_crawler_execution() receives semantic aggregates
and explicit composition dependencies, not hidden parameter bags.
"""

from __future__ import annotations

import inspect

from orchestration.composition.runtime.crawler_execution import (
    CrawlerExecutionOverrides,
    build_crawler_execution,
)


def test_execution_signature_has_semantic_aggregates() -> None:
    """build_crawler_execution() must receive semantic aggregates and explicit deps."""
    sig = inspect.signature(build_crawler_execution)
    params = set(sig.parameters.keys())

    # Semantic aggregates
    assert "settings" in params
    assert "infrastructure" in params
    assert "governance" in params
    assert "state" in params
    assert "seed_plan" in params

    # Explicit composition dependencies (no hidden parameter bag)
    assert "logger_factory" in params
    assert "shutdown_manager" in params
    assert "run_context" in params
    assert "control_directory" in params
    assert "id_generator" in params

    # No generic context/dependencies bag
    assert "runtime" not in params
    assert "context" not in params

    # overrides must be optional with None default
    assert sig.parameters["overrides"].default is None, (
        "overrides must default to None"
    )


def test_execution_overrides_is_optional_with_safe_defaults() -> None:
    """CrawlerExecutionOverrides must have safe defaults for all fields."""
    # Create with no arguments - all fields should have defaults
    overrides = CrawlerExecutionOverrides()

    assert overrides.crawl_attempt_id is None
    assert overrides.crawl_state_manifest_writer is None
    assert overrides.processing_activity_registry is None
    assert overrides.page_settings is None


def test_execution_overrides_reserves_the_manifest_writer_slot() -> None:
    """The workflow manifest writer must not be confused with runtime state."""
    annotations = CrawlerExecutionOverrides.__annotations__
    overrides = CrawlerExecutionOverrides()

    assert annotations["crawl_state_manifest_writer"] == (
        "CrawlStateManifestWriter | None"
    )
    assert "crawl_state_writer" not in annotations
    assert not hasattr(overrides, "crawl_state_writer")
