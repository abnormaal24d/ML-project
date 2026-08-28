"""Ownership contracts for the canonical crawler HostExtractor.

Proves exactly one ``HostExtractor`` exists per crawler graph: built once
in ``CrawlerInfrastructure``, required by host scheduling controls,
reused by governance and execution, and never re-created downstream.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from config.load import load_settings
from crawler.extraction.hosts_extractor import HostExtractor
from crawler.governance.domains.host_normalizer import HostNormalizer
from orchestration.resource_shutdown import ResourceShutdownManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _NoopLogger:
    def __getattr__(self, name: str):
        def _log(*_args: object, **_kwargs: object) -> None:
            return None

        return _log


class _AsyncCloseable:
    async def aclose(self) -> None:
        return None


def _logger_factory() -> Any:
    return SimpleNamespace(
        get_logger=lambda *_a, **_k: _NoopLogger(),
        get_logger_for=lambda *_a, **_k: _NoopLogger(),
    )


def _dev_settings(tmp_path: Path):
    return load_settings(
        "dev",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )


def test_infrastructure_builds_one_extractor_and_feeds_scheduling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from orchestration.composition.runtime import (
        crawler_infrastructure as infra_module,
    )

    scheduling_calls: list[dict[str, object]] = []

    def fake_build_host_scheduling_controls(**kwargs: object):
        scheduling_calls.append(kwargs)
        return (object(), object(), object())

    monkeypatch.setattr(
        infra_module,
        "build_http_transport",
        lambda **_k: (_AsyncCloseable(), _AsyncCloseable()),
    )
    monkeypatch.setattr(
        infra_module, "build_rate_limiter", lambda **_k: object()
    )
    monkeypatch.setattr(
        infra_module,
        "build_host_suppression_store",
        lambda **_k: object(),
    )
    monkeypatch.setattr(
        infra_module,
        "build_host_scheduling_controls",
        fake_build_host_scheduling_controls,
    )

    from orchestration.composition.runtime.crawler_infrastructure import (
        build_crawler_infrastructure,
    )
    from orchestration.resource_shutdown import ResourceShutdownManager

    host_normalizer = HostNormalizer()
    infrastructure = build_crawler_infrastructure(
        settings=_dev_settings(tmp_path),
        logger_factory=_logger_factory(),
        shutdown_manager=ResourceShutdownManager(
            resource_shutdown_timeout_seconds=10.0
        ),
        clock=SimpleNamespace(),
        url_normalizer=SimpleNamespace(),
        host_normalizer=host_normalizer,
    )

    assert isinstance(infrastructure.host_extractor, HostExtractor)
    assert infrastructure.host_normalizer is host_normalizer

    assert len(scheduling_calls) == 1
    assert (
        scheduling_calls[0]["host_extractor"] is infrastructure.host_extractor
    )
    assert (
        scheduling_calls[0]["host_normalizer"]
        is infrastructure.host_normalizer
    )


def test_governance_reuses_the_infrastructure_extractor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from orchestration.composition.runtime import (
        crawler_governance as gov_module,
    )
    from orchestration.composition.runtime.crawler_governance import (
        build_crawler_governance,
    )
    from orchestration.composition.runtime.crawler_infrastructure import (
        CrawlerInfrastructure,
    )

    received: dict[str, dict[str, object]] = {}

    class RecordingRedirector:
        def __init__(self, **kwargs: object) -> None:
            received["redirector"] = kwargs

    def fake_build_url_filter(**kwargs: object) -> object:
        received["url_filter"] = kwargs
        return object()

    def fake_build_blacklist_repository(**_kwargs: object) -> object:
        return object()

    def fake_build_robots_checker(**_kwargs: object) -> object:
        return _AsyncCloseable()

    def fake_build_source_scope_registry(**_kwargs: object) -> object:
        return SimpleNamespace()

    monkeypatch.setattr(
        gov_module, "RedirectRulesValidator", RecordingRedirector
    )
    monkeypatch.setattr(gov_module, "build_url_filter", fake_build_url_filter)
    monkeypatch.setattr(
        gov_module,
        "build_blacklist_repository",
        fake_build_blacklist_repository,
    )
    monkeypatch.setattr(
        gov_module, "build_robots_checker", fake_build_robots_checker
    )
    monkeypatch.setattr(
        gov_module,
        "build_source_scope_registry",
        fake_build_source_scope_registry,
    )

    host_extractor = HostExtractor(logger=_NoopLogger())

    infrastructure = CrawlerInfrastructure(
        host_normalizer=HostNormalizer(),
        host_extractor=host_extractor,
        metrics=SimpleNamespace(),
        prometheus_exporter=None,
        network_access_guard=SimpleNamespace(),
        session_manager=SimpleNamespace(),
        rate_limiter=SimpleNamespace(),
        coverage_tracker=SimpleNamespace(),
        host_budget_tracker=SimpleNamespace(),
        host_media_byte_budget=SimpleNamespace(),
        priority_resolver=SimpleNamespace(),
        host_suppression_store=SimpleNamespace(),
        conditional_representation_cache=SimpleNamespace(),
        clock=SimpleNamespace(),
        url_normalizer=SimpleNamespace(),
    )

    governance = build_crawler_governance(
        settings=_dev_settings(tmp_path),
        infrastructure=infrastructure,
        logger_factory=_logger_factory(),
        shutdown_manager=ResourceShutdownManager(
            resource_shutdown_timeout_seconds=10.0
        ),
    )

    assert received["redirector"]["host_extractor"] is host_extractor
    assert received["url_filter"]["host_extractor"] is host_extractor

    # Governance must not own or re-export the extractor.
    assert not hasattr(governance, "host_extractor")


def test_processing_runtime_share_conditional_representation_cache() -> None:
    """Fetch and dataset persistence must receive the one infrastructure cache.

    Verifies by AST that ``build_processing_runtime`` passes
    ``infrastructure.conditional_representation_cache`` to both
    ``build_fetcher`` and ``build_dataset_writer``.
    """
    import ast
    import inspect
    import textwrap

    from orchestration.composition.runtime.crawler_execution.processing import (
        build_processing_runtime,
    )

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(build_processing_runtime)),
    )

    fetcher_call = None
    dataset_writer_call = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name):
            continue
        if func.id == "build_fetcher":
            fetcher_call = node
        elif func.id == "build_dataset_writer":
            dataset_writer_call = node

    assert fetcher_call is not None, "build_fetcher call not found"
    assert dataset_writer_call is not None, "build_dataset_writer call not found"

    def _find_keyword(call: ast.Call, name: str) -> ast.expr | None:
        for kw in call.keywords:
            if kw.arg == name:
                return kw.value
        return None

    fetcher_cache = _find_keyword(fetcher_call, "conditional_representation_cache")
    writer_cache = _find_keyword(
        dataset_writer_call, "conditional_representation_cache"
    )

    assert fetcher_cache is not None, (
        "build_fetcher missing conditional_representation_cache"
    )
    assert writer_cache is not None, (
        "build_dataset_writer missing conditional_representation_cache"
    )

    assert isinstance(fetcher_cache, ast.Attribute)
    assert isinstance(fetcher_cache.value, ast.Name)
    assert fetcher_cache.value.id == "infrastructure"
    assert fetcher_cache.attr == "conditional_representation_cache"

    assert isinstance(writer_cache, ast.Attribute)
    assert isinstance(writer_cache.value, ast.Name)
    assert writer_cache.value.id == "infrastructure"
    assert writer_cache.attr == "conditional_representation_cache"


@pytest.mark.asyncio
async def test_seed_feedback_works_with_real_extractor() -> None:
    """The exact code path broken by ``host_extractor=None``."""

    from crawler.scheduling.host_control.host_feedback_aggregator import (
        HostFeedbackAggregator,
    )

    aggregator = HostFeedbackAggregator(
        max_hosts=None,
        default_info_gain=0.5,
        default_host_quality=0.5,
        host_extractor=HostExtractor(logger=_NoopLogger()),
        host_normalizer=HostNormalizer(),
    )

    assert aggregator.get_or_create("example.com") is not None
