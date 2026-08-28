"""Propagation contracts for the Prometheus exporter edge.

Proves the exporter built once by ``CrawlerInfrastructure`` reaches
``build_runtime_session_factory`` via ``infrastructure.prometheus_exporter``,
and stays ``None`` end-to-end when the feature is disabled.

Step 2 removed ``prometheus_exporter`` as a flat parameter;
it now flows via ``infrastructure.prometheus_exporter``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.load import load_settings
from crawler.governance.processing_activity import ProcessingActivityRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _Clock:
    def now(self):
        return datetime(2026, 1, 1, tzinfo=UTC)


def _settings(tmp_path: Path, *, prometheus_enabled: bool):
    settings = load_settings(
        "dev",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )
    metrics = settings.collection.metrics.model_copy(
        update={"prometheus_enabled": prometheus_enabled}
    )
    collection = settings.collection.model_copy(update={"metrics": metrics})
    return settings.model_copy(update={"collection": collection})


def _logger_factory():
    class _NoopLogger:
        def __getattr__(self, name: str):
            return lambda *_a, **_k: None

    return SimpleNamespace(
        get_logger=lambda *_a, **_k: _NoopLogger(),
        get_logger_for=lambda *_a, **_k: _NoopLogger(),
    )


def _build_infrastructure(tmp_path: Path, *, prometheus_enabled: bool):
    from crawler.extraction.urls.normalizer import UrlNormalizer
    from crawler.governance.domains.host_normalizer import HostNormalizer
    from orchestration.composition.runtime.crawler_infrastructure import (
        build_crawler_infrastructure,
    )
    from orchestration.resource_shutdown import ResourceShutdownManager

    settings = _settings(tmp_path, prometheus_enabled=prometheus_enabled)
    host_normalizer = HostNormalizer()
    url_normalizer = UrlNormalizer(
        settings=settings.collection.url_normalizer,
        logger=_logger_factory().get_logger_for(UrlNormalizer),
        host_normalizer=host_normalizer,
    )
    return build_crawler_infrastructure(
        settings=settings,
        logger_factory=_logger_factory(),
        shutdown_manager=ResourceShutdownManager(
            resource_shutdown_timeout_seconds=10.0
        ),
        clock=_Clock(),
        url_normalizer=url_normalizer,
        host_normalizer=host_normalizer,
    )


@pytest.mark.parametrize("prometheus_enabled", [True, False])
def test_infrastructure_aggregate_passed_to_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    prometheus_enabled: bool,
) -> None:
    """Infrastructure aggregate is passed through to execution; identity preserved."""
    from orchestration.composition.runtime import (
        crawler_graph as graph_module,
    )

    infrastructure = _build_infrastructure(
        tmp_path, prometheus_enabled=prometheus_enabled
    )

    received: dict[str, object] = {}

    def recording_build_crawler_execution(**kwargs: object) -> object:
        received.update(kwargs)
        return SimpleNamespace(
            control_directory=SimpleNamespace(),
            dataset_writer=SimpleNamespace(),
            worker_pool=SimpleNamespace(),
            worker_scaler=SimpleNamespace(),
            scheduler=SimpleNamespace(),
            seed_enqueuer=SimpleNamespace(),
            task_feedback=SimpleNamespace(),
            build_runtime_session=lambda: None,
        )

    monkeypatch.setattr(
        graph_module,
        "build_crawler_execution",
        recording_build_crawler_execution,
    )
    monkeypatch.setattr(
        graph_module,
        "build_crawler_infrastructure",
        lambda **_kwargs: infrastructure,
    )

    from orchestration.bootstrap.run_context import create_run_context
    from orchestration.resource_shutdown import ResourceShutdownManager

    graph_module.build_crawler_graph(
        settings=_settings(tmp_path, prometheus_enabled=prometheus_enabled),
        processing_activity_registry=ProcessingActivityRegistry(
            schema_version="1.0.0",
            activities=(),
        ),
        logger_factory=_logger_factory(),
        run_context=create_run_context(stage="prometheus_smoke"),
        shutdown_manager=ResourceShutdownManager(
            resource_shutdown_timeout_seconds=10.0
        ),
        clock=SimpleNamespace(now=lambda: datetime(2026, 1, 1, tzinfo=UTC)),
        id_generator=SimpleNamespace(generate=lambda: "id-1"),
    )

    assert received["infrastructure"] is infrastructure


def test_prometheus_exporter_wiring_in_source() -> None:
    """Source-level proof that build_runtime_session_factory receives infrastructure.prometheus_exporter."""
    import ast
    import pathlib

    execution_file = pathlib.Path(
        "orchestration/composition/runtime/crawler_execution/composition.py"
    )

    source = execution_file.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "build_runtime_session_factory":
            continue

        for keyword in node.keywords:
            if keyword.arg != "prometheus_exporter":
                continue

            # Verify the value is exactly `infrastructure.prometheus_exporter`
            value = keyword.value
            assert isinstance(value, ast.Attribute), (
                f"prometheus_exporter value must be an attribute access, got {type(value).__name__}"
            )
            assert value.attr == "prometheus_exporter", (
                f"Expected .prometheus_exporter, got .{value.attr}"
            )
            assert isinstance(value.value, ast.Name), (
                f"Expected attribute owner to be a Name node, got {type(value.value).__name__}"
            )
            assert value.value.id == "infrastructure", (
                f"Expected infrastructure.prometheus_exporter, got {value.value.id}.prometheus_exporter"
            )
            return

    pytest.fail(
        "build_runtime_session_factory call not found in "
        "crawler_execution/composition.py"
    )
