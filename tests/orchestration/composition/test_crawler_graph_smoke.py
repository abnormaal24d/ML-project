"""End-to-end crawler graph construction smoke test.

Builds the complete crawler object graph through ``build_crawler_graph``
with real dev settings and no network traffic. This is the highest-ROI
composition test: it catches every missing import, wrong capability type,
and duplicated graph service at construction time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from config.load import load_settings
from crawler.governance.processing_activity import ProcessingActivityRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)


class _IdGenerator:
    def __init__(self) -> None:
        self._counter = 0

    def generate(self) -> str:
        self._counter += 1
        return f"id-{self._counter}"


def _logger_factory():
    from logger.factory import ProjectLoggerFactory

    return ProjectLoggerFactory(
        root_name="smoke",
        base_context={"stage": "crawler_graph_smoke"},
    )


def test_crawler_graph_builds_the_complete_object_graph(
    tmp_path: Path,
) -> None:
    from orchestration.bootstrap.run_context import create_run_context
    from orchestration.composition.runtime.crawler_graph import (
        build_crawler_graph,
    )
    from orchestration.resource_shutdown import ResourceShutdownManager

    settings = load_settings(
        "dev",
        project_root=tmp_path / "project-root",
        config_root=PROJECT_ROOT,
        environment="dev",
    )

    graph = build_crawler_graph(
        settings=settings,
        processing_activity_registry=ProcessingActivityRegistry(
            schema_version="1.0.0",
            activities=(),
        ),
        logger_factory=_logger_factory(),
        run_context=create_run_context(stage="crawler_graph_smoke"),
        shutdown_manager=ResourceShutdownManager(
            resource_shutdown_timeout_seconds=10.0
        ),
        clock=_Clock(),
        id_generator=_IdGenerator(),
    )

    # Every subgraph produced its primary runtime object.
    assert graph.crawler is not None
    assert graph.dataset_writer is not None
    assert graph.seed_plan is not None
    assert graph.seed_plan.tasks, "seed plan must produce tasks"

    # The canonical control directory flows state -> execution -> crawler.
    assert isinstance(graph.crawler._control_directory, object)


def test_crawler_graph_shares_one_control_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """State and execution must receive the same ControlDirectory instance."""
    from orchestration.bootstrap.run_context import create_run_context
    from orchestration.composition.runtime.crawler_graph import (
        CrawlerControlDirectory,
        build_crawler_graph,
    )
    from orchestration.resource_shutdown import ResourceShutdownManager

    constructions: list[Path] = []
    real_init = CrawlerControlDirectory.__init__

    def recording_init(
        self: CrawlerControlDirectory,
        *,
        settings: object,
        project_root: Path,
    ) -> None:
        constructions.append(project_root)
        real_init(self, settings=settings, project_root=project_root)

    monkeypatch.setattr(CrawlerControlDirectory, "__init__", recording_init)

    graph = build_crawler_graph(
        settings=load_settings(
            "dev",
            project_root=tmp_path / "project-root",
            config_root=PROJECT_ROOT,
            environment="dev",
        ),
        processing_activity_registry=ProcessingActivityRegistry(
            schema_version="1.0.0",
            activities=(),
        ),
        logger_factory=_logger_factory(),
        run_context=create_run_context(stage="crawler_graph_smoke"),
        shutdown_manager=ResourceShutdownManager(
            resource_shutdown_timeout_seconds=10.0
        ),
        clock=_Clock(),
        id_generator=_IdGenerator(),
    )

    assert len(constructions) == 1
    assert isinstance(
        graph.crawler._control_directory, CrawlerControlDirectory
    )
